// gguf.cpp - GGUF v3 reader implementation (mmap based).
#include "sager/core/gguf.hpp"

#include <cstring>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace sager::gguf {

namespace {

// ggml tensor type sizes (only the types the engine consumes).
struct TypeInfo { const char* name; uint64_t block_size; uint64_t type_size; };
const std::unordered_map<uint32_t, TypeInfo> kTypeTable = {
    {0,  {"F32",  1, 4}},        // 4 B / weight
    {1,  {"F16",  1, 2}},        // 2 B / weight
    {8,  {"Q8_0", 32, 34}},      // 34 B / 32 weights
    {12, {"Q4_K", 256, 144}},    // 144 B / 256 weights (this engine's 4-bit)
    {14, {"Q6_K", 256, 210}},    // 210 B / 256 weights
    {30, {"BF16", 1, 2}},
};

uint64_t type_row_bytes(uint32_t type, uint64_t ne0) {
  auto it = kTypeTable.find(type);
  if (it == kTypeTable.end()) return 0;
  return (ne0 / it->second.block_size) * it->second.type_size;
}

uint64_t type_total_bytes(uint32_t type, const std::vector<uint64_t>& dims) {
  auto it = kTypeTable.find(type);
  if (it == kTypeTable.end() || dims.empty()) return 0;
  uint64_t n = dims[0] / it->second.block_size * it->second.type_size;
  for (size_t i = 1; i < dims.size(); ++i) n *= dims[i];
  return n;
}

}  // namespace

bool File::read_exact(void* dst, size_t n) {
  if (cursor_ + n > map_ + map_len_) return false;  // bounds-checked mmap read
  std::memcpy(dst, cursor_, n);
  cursor_ += n;
  return true;
}

bool File::read_str(std::string* out) {
  uint64_t len = 0;
  if (!read_exact(&len, 8)) return false;
  if (len > (1ull << 24)) return false;  // 16 MB: no legitimate key/name here
  out->resize(len);
  return read_exact(out->data(), len);
}

void File::close() {
  if (map_) {
#ifdef _WIN32
    UnmapViewOfFile(map_);
#else
    munmap(map_, map_len_);
#endif
    map_ = nullptr;
    map_len_ = 0;
    cursor_ = nullptr;
    data_base_ = nullptr;
  }
}

Status File::open(const std::string& path) {
  close();

#ifdef _WIN32
  HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return Status::kFileNotFound;
  LARGE_INTEGER sz;
  GetFileSizeEx(h, &sz);
  map_len_ = static_cast<size_t>(sz.QuadPart);
  HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  CloseHandle(h);
  if (!m) return Status::kGgufParseError;
  map_ = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(m);
  if (!map_) return Status::kGgufParseError;
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return Status::kFileNotFound;
  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < 32) { ::close(fd); return Status::kGgufParseError; }
  map_len_ = static_cast<size_t>(st.st_size);
  map_ = ::mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map_ == MAP_FAILED) { map_ = nullptr; return Status::kGgufParseError; }
#endif

  cursor_ = static_cast<const uint8_t*>(map_);

  char magic[4] = {0};
  if (!read_exact(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    close(); return Status::kGgufParseError;
  }
  uint32_t version = 0;
  if (!read_exact(&version, 4) || version != 3) {
    close(); return Status::kGgufParseError;  // v1/v2 files: re-export with the converter
  }

  uint64_t n_tensors = 0, n_kv = 0;
  if (!read_exact(&n_tensors, 8) || !read_exact(&n_kv, 8)) { close(); return Status::kGgufParseError; }

  // --- KV metadata ---
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    if (!read_str(&key)) { close(); return Status::kGgufParseError; }
    uint32_t vtype = 0;
    if (!read_exact(&vtype, 4)) { close(); return Status::kGgufParseError; }

    Value v; v.type = static_cast<Type>(vtype);
    switch (v.type) {
      case Type::u8: case Type::i8: case Type::boolean: {
        uint8_t x; if (!read_exact(&x, 1)) { close(); return Status::kGgufParseError; }
        v.u64v = x; break;
      }
      case Type::u16: case Type::i16: {
        uint16_t x; if (!read_exact(&x, 2)) { close(); return Status::kGgufParseError; }
        v.u64v = x; break;
      }
      case Type::u32: case Type::i32: {
        uint32_t x; if (!read_exact(&x, 4)) { close(); return Status::kGgufParseError; }
        v.u64v = x; break;
      }
      case Type::u64: case Type::i64: {
        if (!read_exact(&v.u64v, 8)) { close(); return Status::kGgufParseError; }
        break;
      }
      case Type::f32: {
        float x; if (!read_exact(&x, 4)) { close(); return Status::kGgufParseError; }
        v.f64v = x; break;
      }
      case Type::f64: {
        if (!read_exact(&v.f64v, 8)) { close(); return Status::kGgufParseError; }
        break;
      }
      case Type::f16: {
        uint16_t x; if (!read_exact(&x, 2)) { close(); return Status::kGgufParseError; }
        // fp16 -> double (host half conversion from q4_k.hpp is internal; use a
        // small inline here to avoid a circular include).
        uint32_t s = (x & 0x8000u) << 16, e = (x & 0x7C00u) >> 10, m = x & 0x3FFu;
        uint32_t bits = (e == 0) ? (s | ((127 - 15 + 1) << 23))
                                 : (s | ((e + 112u) << 23) | (m << 13));
        float f; std::memcpy(&f, &bits, 4);
        v.f64v = (x & 0x7FFF) == 0 ? 0.0 : f; break;
      }
      case Type::string: {
        if (!read_str(&v.str)) { close(); return Status::kGgufParseError; }
        break;
      }
      case Type::array: {
        uint32_t et = 0; uint64_t count = 0;
        if (!read_exact(&et, 4) || !read_exact(&count, 8)) { close(); return Status::kGgufParseError; }
        v.elem_type = static_cast<Type>(et);
        v.elem_count = count;
        size_t elem_sz = 0;
        switch (static_cast<Type>(et)) {
          case Type::u8: case Type::i8: case Type::boolean: elem_sz = 1; break;
          case Type::u16: case Type::i16: case Type::f16:   elem_sz = 2; break;
          case Type::u32: case Type::i32: case Type::f32:   elem_sz = 4; break;
          case Type::u64: case Type::i64: case Type::f64:   elem_sz = 8; break;
          case Type::string: elem_sz = 0; break;  // handled element-wise
          default: close(); return Status::kGgufParseError;
        }
        if (elem_sz == 0) {
          // array of strings: length-prefixed each
          for (uint64_t j = 0; j < count; ++j) {
            std::string s;
            if (!read_str(&s)) { close(); return Status::kGgufParseError; }
            uint64_t len = s.size();
            const uint8_t* p = reinterpret_cast<const uint8_t*>(&len);
            v.raw.insert(v.raw.end(), p, p + 8);
            v.raw.insert(v.raw.end(), s.begin(), s.end());
          }
        } else {
          v.raw.resize(static_cast<size_t>(count) * elem_sz);
          if (!read_exact(v.raw.data(), v.raw.size())) { close(); return Status::kGgufParseError; }
        }
        break;
      }
      default:
        close(); return Status::kGgufParseError;
    }
    kv_.emplace(std::move(key), std::move(v));
  }

  // --- Tensor directory ---
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TensorInfo t;
    if (!read_str(&t.name)) { close(); return Status::kGgufParseError; }
    uint32_t n_dims = 0;
    if (!read_exact(&n_dims, 4) || n_dims == 0 || n_dims > 4) { close(); return Status::kGgufParseError; }
    t.dims.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      if (!read_exact(&t.dims[d], 8)) { close(); return Status::kGgufParseError; }
    }
    if (!read_exact(&t.type, 4) || !read_exact(&t.offset, 8)) { close(); return Status::kGgufParseError; }
    t.nbytes = type_total_bytes(t.type, t.dims);
    t.row_bytes = type_row_bytes(t.type, t.dims.empty() ? 0 : t.dims[0]);
    if (t.nbytes == 0) { close(); return Status::kGgufParseError; }  // unsupported dtype
    tensors_.emplace(t.name, std::move(t));
  }

  // --- Alignment & data section ---
  alignment_ = static_cast<uint64_t>(kv_int("general.alignment", 32));
  if (alignment_ == 0) alignment_ = 32;
  uint64_t data_start = static_cast<uint64_t>(cursor_ - static_cast<const uint8_t*>(map_));
  data_start = (data_start + alignment_ - 1) / alignment_ * alignment_;
  if (data_start >= map_len_) { close(); return Status::kGgufParseError; }
  data_base_ = static_cast<const uint8_t*>(map_) + data_start;

  return Status::kOk;
}

}  // namespace sager::gguf
