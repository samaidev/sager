// common.cpp - env/time/ram helpers declared in common.hpp.
#include "sager/common.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <fstream>
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace sager {

std::optional<std::string> env_get(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) return std::nullopt;
  return std::string(v);
}

bool env_flag(const char* name) {
  auto v = env_get(name);
  if (!v) return false;
  return *v == "1" || *v == "true" || *v == "on" || *v == "yes";
}

int env_int(const char* name, int fallback) {
  auto v = env_get(name);
  if (!v) return fallback;
  char* end = nullptr;
  long parsed = std::strtol(v->c_str(), &end, 10);
  if (end == v->c_str()) return fallback;
  return static_cast<int>(parsed);
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint64_t host_ram_bytes() {
#if defined(__linux__)
  struct sysinfo si{};
  if (::sysinfo(&si) == 0) return static_cast<uint64_t>(si.totalram) * si.mem_unit;
  return 0;
#elif defined(__APPLE__)
  int64_t mem = 0;
  size_t len = sizeof(mem);
  sysctlbyname("hw.memsize", &mem, &len, nullptr, 0);
  return static_cast<uint64_t>(mem);
#elif defined(_WIN32)
  MEMORYSTATUSEX ms{};
  ms.dwLength = sizeof(ms);
  GlobalMemoryStatusEx(&ms);
  return ms.ullTotalPhys;
#else
  return 0;
#endif
}

}  // namespace sager
