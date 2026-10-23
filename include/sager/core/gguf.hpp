// gguf.hpp - minimal, mmap-based GGUF v3 reader (the container SAGER ships
// and consumes). We only need: header parse, typed KV metadata, tensor
// directory, and stable offsets into the mmap so weights can be used without
// an intermediate copy (cold experts go straight from page cache into the
// pinned pool on load).
//
// Spec summary implemented here:
//   magic "GGUF" | version u32 (3) | n_tensors u64 | n_kv u64
//   kv: string key + u32 value type + payload
//   tensor info: string name + u32 n_dims + u64 dims[] + u32 type + u64 offset
//   alignment: general.alignment kv (default 32)
#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "sager/common.hpp"

namespace sager::gguf {

enum class Type : uint32_t {
  u8 = 0, i8, u16, i16, u32, i32, u64, i64, f32, f64, boolean,
  string, array, f16,
};

struct Value {
  Type type = Type::u8;
  // Scalars live in a union-ish pair; arrays/strings own their bytes.
  uint64_t u64v = 0;
  double   f64v = 0.0;
  std::string str;
  Type elem_type = Type::u8;          // for arrays
  std::vector<uint8_t> raw;           // for arrays (packed elements)
  size_t elem_count = 0;              // for arrays

  int64_t  as_int(int64_t d = 0) const {
    switch (type) {
      case Type::i8: case Type::i16: case Type::i32: case Type::i64:
        return static_cast<int64_t>(u64v);
      case Type::u8: case Type::u16: case Type::u32: case Type::u64:
        return static_cast<int64_t>(u64v);
      case Type::f16: case Type::f32: case Type::f64:
        return static_cast<int64_t>(f64v);
      default: return d;
    }
  }
  double   as_double(double d = 0) const {
    if (type == Type::f32 || type == Type::f64 || type == Type::f16) return f64v;
    if (type == Type::boolean) return u64v ? 1.0 : 0.0;
    if (type == Type::string) return d;
    return static_cast<double>(static_cast<int64_t>(u64v));
  }
  bool     as_bool(bool d = false) const { return type == Type::boolean ? (u64v != 0) : d; }
  const std::string& as_string() const { static const std::string kEmpty; return type == Type::string ? str : kEmpty; }
};

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> dims;   // gguf order: dims[0] = fastest (ne[0])
  uint32_t type = 0;            // ggml tensor type enum (6 = Q4_K, 0 = F32, 1 = F16)
  uint64_t offset = 0;          // relative to data section start
  uint64_t nbytes = 0;          // computed from dims/type
  // row stride convenience: size of one dim[1] row (q4_K: dim[0]/256*144)
  uint64_t row_bytes = 0;
};

class File {
 public:
  File() = default;
  ~File() { close(); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  Status open(const std::string& path);
  void close();

  const std::map<std::string, Value>& kv() const { return kv_; }
  const Value* kv_find(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
  }
  int64_t kv_int(const std::string& key, int64_t dflt) const {
    const Value* v = kv_find(key); return v ? v->as_int(dflt) : dflt;
  }
  double  kv_double(const std::string& key, double dflt) const {
    const Value* v = kv_find(key); return v ? v->as_double(dflt) : dflt;
  }
  std::string kv_str(const std::string& key, const std::string& dflt = "") const {
    const Value* v = kv_find(key); return v ? v->as_string() : dflt;
  }

  const TensorInfo* tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
  }
  const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }

  // Raw pointer to tensor data inside the mmap (valid until close()).
  const uint8_t* tensor_data(const TensorInfo& t) const {
    return data_base_ ? data_base_ + t.offset : nullptr;
  }
  uint64_t alignment() const { return alignment_; }

 private:
  bool read_exact(void* dst, size_t n);
  bool read_str(std::string* out);

  void* map_ = nullptr;
  size_t map_len_ = 0;
  const uint8_t* cursor_ = nullptr;
  const uint8_t* data_base_ = nullptr;
  std::map<std::string, Value> kv_;
  std::unordered_map<std::string, TensorInfo> tensors_;
  uint64_t alignment_ = 32;
};

}  // namespace sager::gguf
