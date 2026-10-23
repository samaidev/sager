// engine_main.cpp - SAGER resident engine: JSON-lines on stdin/stdout.
//
// The Python serving layer (serve/engine_proc.py) spawns this process and
// speaks the protocol below. Logs go to STDERR ONLY; stdout is protocol.
//
// Requests  (one JSON object per line):
//   {"cmd":"load","model":"/path/model.gguf","ctx":16384,"vram_mb":10240,
//    "threads":0,"graphs":true,"mtp":true,"chunk":8192,"kv_q8":true}
//   {"cmd":"generate","ids":[...],"max_tokens":512,"temperature":1.0,
//    "top_k":20,"top_p":0.95,"min_p":0.0,"presence_penalty":1.5,
//    "repeat_penalty":1.0,"greedy":false,"seed":0,"stop_ids":[151645],
//    "stream":true}
//   {"cmd":"stats"} | {"cmd":"reset"} | {"cmd":"shutdown"}
//
// Responses:
//   {"type":"loaded",...} {"type":"token","id":N} {"type":"done",...}
//   {"type":"stats",...} {"type":"error","message":"..."}
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "sager/common.hpp"
#include "sager/core/config.hpp"
#include "sager/core/gguf.hpp"
#include "sager/core/session.hpp"
#include "sager/core/weights.hpp"
#include "sager/core/expert_cache.hpp"
#include "sager/core/expert_source.hpp"
#include "sager/log.hpp"
#include "sager/util/json.hpp"
#include "generate.hpp"

using namespace sager;  // NOLINT - program file

namespace {

struct EngineState {
  std::unique_ptr<gguf::File> file;
  std::unique_ptr<Weights> weights;
  std::unique_ptr<ExpertCache> cache;
  std::unique_ptr<CpuPool> pool;
  std::unique_ptr<Session> session;
  std::unique_ptr<SpecController> ctrl;
  std::unique_ptr<MtpDrafter> mtp;
  std::unique_ptr<SuffixDrafter> suffix;
  ModelConfig mc;
  CtxParams cp;
};

void emit(const json::Value& v) {
  std::fputs(v.dump().c_str(), stdout);
  fputc('\n', stdout);
  std::fflush(stdout);
}

json::Value jerr(const std::string& msg) {
  json::Value v = json::Value::object();
  v["type"] = "error";
  v["message"] = msg;
  return v;
}

json::Value load_cmd(EngineState& es, const json::Value& req) {
  const std::string path =
      req.find("model") ? req.find("model")->as_string() : "";
  if (path.empty()) return jerr("load: missing 'model'");

  // Fresh state for a fresh file.
  es.session.reset();
  es.weights.reset();
  es.file.reset();

  es.file = std::make_unique<gguf::File>();
  Status s = es.file->open(path);
  if (s != Status::kOk) return jerr("gguf open failed: " + path);

  s = ModelConfig::load(path, &es.mc);
  if (s != Status::kOk) return jerr("config load failed");

  CtxParams& cp = es.cp;
  if (const auto* v = req.find("ctx"))
    cp.n_ctx = static_cast<int>(v->as_int(16384));
  if (const auto* v = req.find("chunk"))
    cp.chunk_prefill = static_cast<int>(v->as_int(8192));
  if (const auto* v = req.find("vram_mb"))
    cp.vram_budget_mb = static_cast<size_t>(v->as_int(10240));
  if (const auto* v = req.find("threads"))
    cp.batch_threads = static_cast<int>(v->as_int(0));
  if (const auto* v = req.find("kv_q8")) cp.kv_q8 = v->as_bool(true);
  if (const auto* v = req.find("graphs")) cp.capture_graphs = v->as_bool(true);
  if (const auto* v = req.find("mtp")) cp.enable_mtp = v->as_bool(true);
  if (const auto* v = req.find("suffix"))
    cp.enable_suffix_draft = v->as_bool(true);
  if (const auto* v = req.find("mtp_max_draft"))
    cp.mtp_max_draft = static_cast<int>(v->as_int(3));

  es.weights = std::make_unique<Weights>();
  s = es.weights->load(*es.file, es.mc, cp.vram_budget_mb << 20);
  if (s != Status::kOk)
    return jerr(std::string("weights load failed: ") + status_str(s));

  es.cache = std::make_unique<ExpertCache>();
  es.pool = make_cpu_pool();
  es.session = std::make_unique<Session>(*es.weights, *es.cache, *es.pool,
                                         es.mc, cp);
  s = es.session->prepare();
  if (s != Status::kOk) {
    es.session.reset();
    return jerr(std::string("prepare failed: ") + status_str(s));
  }

  es.ctrl = std::make_unique<SpecController>(SpecController::Config{});
  es.mtp = std::make_unique<MtpDrafter>(*es.session,
                                        cp.enable_mtp ? cp.mtp_max_draft : 0);
  es.mtp->set_ready(cp.enable_mtp && es.weights->mtp_present);
  es.suffix = std::make_unique<SuffixDrafter>();
  es.suffix->set_max_draft(cp.enable_suffix_draft ? cp.suffix_max_draft : 0);

  const EngineDims& d = es.mc.dims;
  json::Value out = json::Value::object();
  out["type"] = "loaded";
  out["model"] = path;
  out["arch"] = es.mc.arch;
  out["quant"] = es.mc.quant_name;
  out["layers"] = static_cast<int64_t>(d.n_layer);
  out["experts"] = static_cast<int64_t>(d.experts_total());
  out["vocab"] = static_cast<int64_t>(d.n_vocab);
  out["ctx"] = static_cast<int64_t>(cp.n_ctx);
  out["mtp"] = es.mtp->ready();
  out["graphs"] = es.session->graphs().valid();
  out["hot_experts"] = static_cast<int64_t>(es.cache->stats().resident_count);
  out["hot_capacity"] = static_cast<int64_t>(es.cache->capacity());
  return out;
}

json::Value generate_cmd(EngineState& es, const json::Value& req,
                         const std::atomic<bool>& abort) {
  if (!es.session) return jerr("no model loaded");

  GenParams gp;
  if (const auto* v = req.find("ids")) {
    for (const auto& t : v->as_array()) {
      gp.prompt_ids.push_back(static_cast<int>(t.as_int()));
    }
  }
  if (gp.prompt_ids.empty()) return jerr("generate: empty prompt");

  if (const auto* v = req.find("max_tokens"))
    gp.max_tokens = static_cast<int>(v->as_int(512));
  if (const auto* v = req.find("temperature"))
    gp.temperature = static_cast<float>(v->as_double(1.0));
  if (const auto* v = req.find("top_k"))
    gp.top_k = static_cast<int>(v->as_int(20));
  if (const auto* v = req.find("top_p"))
    gp.top_p = static_cast<float>(v->as_double(0.95));
  if (const auto* v = req.find("min_p"))
    gp.min_p = static_cast<float>(v->as_double(0.0));
  if (const auto* v = req.find("presence_penalty"))
    gp.presence_penalty = static_cast<float>(v->as_double(1.5));
  if (const auto* v = req.find("repeat_penalty"))
    gp.repeat_penalty = static_cast<float>(v->as_double(1.0));
  if (const auto* v = req.find("greedy")) gp.greedy = v->as_bool(false);
  if (const auto* v = req.find("seed"))
    gp.seed = static_cast<uint64_t>(v->as_int(0));
  if (const auto* v = req.find("stop_ids")) {
    for (const auto& t : v->as_array()) {
      gp.stop_ids.push_back(static_cast<int>(t.as_int()));
    }
  }

  GenResult result;
  GenCallbacks cb;
  cb.on_token = [](int id) {
    json::Value line = json::Value::object();
    line["type"] = "token";
    line["id"] = static_cast<int64_t>(id);
    emit(line);
    return true;   // client abort arrives via the atomic flag
  };
  Status s = run_generation(*es.session, *es.ctrl, *es.mtp, *es.suffix, gp,
                            &abort, cb, &result);
  if (s != Status::kOk && result.tokens == 0)
    return jerr(std::string("generate failed: ") + status_str(s));

  const ExpertCache::Stats cs = es.cache->stats();
  json::Value out = json::Value::object();
  out["type"] = "done";
  out["tokens"] = static_cast<int64_t>(result.tokens);
  out["prefill_tps"] = result.prefill_tps;
  out["decode_tps"] = result.decode_tps;
  out["drafted"] = static_cast<int64_t>(result.drafted);
  out["accepted"] = static_cast<int64_t>(result.accepted);
  out["cache_hit_rate"] = cs.hit_rate();
  out["pos"] = static_cast<int64_t>(es.session->pos());
  return out;
}

json::Value stats_cmd(EngineState& es) {
  const ExpertCache::Stats cs =
      es.cache ? es.cache->stats() : ExpertCache::Stats{};
  json::Value out = json::Value::object();
  out["type"] = "stats";
  out["cache_hits"] = static_cast<int64_t>(cs.hits);
  out["cache_misses"] = static_cast<int64_t>(cs.misses);
  out["cache_hit_rate"] = cs.hit_rate();
  out["hot_experts"] = static_cast<int64_t>(cs.resident_count);
  out["hot_capacity"] = static_cast<int64_t>(cs.capacity);
  out["rebalances"] = static_cast<int64_t>(cs.rebalances);
  if (es.session) {
    out["decode_tps"] = es.session->perf.decode_tps;
    out["prefill_tps"] = es.session->perf.prefill_tps;
    out["steps"] = static_cast<int64_t>(es.session->perf.steps);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  EngineState es;
  std::atomic<bool> abort{false};
  std::string cli_model;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc)
      cli_model = argv[++i];
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    json::Value req;
    if (!json::Value::parse(line, &req)) {
      emit(jerr("bad json"));
      continue;
    }
    const std::string cmd =
        req.find("cmd") ? req.find("cmd")->as_string() : "";

    if (cmd == "load") {
      emit(load_cmd(es, req));
    } else if (cmd == "generate") {
      emit(generate_cmd(es, req, abort));
    } else if (cmd == "stats") {
      emit(stats_cmd(es));
    } else if (cmd == "reset") {
      if (es.session) es.session->reset();
      if (es.suffix) es.suffix->reset();
      json::Value ok = json::Value::object();
      ok["type"] = "ok";
      emit(ok);
    } else if (cmd == "shutdown") {
      json::Value bye = json::Value::object();
      bye["type"] = "bye";
      emit(bye);
      break;
    } else {
      emit(jerr("unknown cmd: " + cmd));
    }
  }

  // Persist the expert usage profile for the next start (warm ramp).
  if (es.cache) {
    if (const auto p = env_get("SAGER_PROFILE_PATH")) es.cache->save_profile(*p);
  }
  return 0;
}
