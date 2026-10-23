// json.hpp - a tiny, dependency-free JSON value + serializer used by the
// engine's stdin/stdout protocol and the expert-profile persistence.
//
// It is intentionally small: the protocol surface is tiny and fixed
// (see docs/architecture.md "Engine protocol"). Supports the full value
// model (null/bool/number/string/array/object) with 64-bit integers and
// doubles kept distinct.
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sager::json {

class Value;
using Object = std::map<std::string, Value>;
using Array  = std::vector<Value>;

class Value {
 public:
  enum class Type : int { kNull, kBool, kInt, kDouble, kString, kArray, kObject };

  Value() : type_(Type::kNull) {}
  Value(std::nullptr_t) : type_(Type::kNull) {}
  Value(bool b) : type_(Type::kBool), bool_(b) {}
  Value(int v) : type_(Type::kInt), int_(v) {}
  Value(int64_t v) : type_(Type::kInt), int_(v) {}
  Value(uint32_t v) : type_(Type::kInt), int_(static_cast<int64_t>(v)) {}
  Value(size_t v) : type_(Type::kInt), int_(static_cast<int64_t>(v)) {}
  Value(double d) : type_(Type::kDouble), dbl_(d) {}
  Value(const char* s) : type_(Type::kString), str_(s) {}
  Value(std::string s) : type_(Type::kString), str_(std::move(s)) {}
  Value(Array a) : type_(Type::kArray), arr_(std::make_shared<Array>(std::move(a))) {}
  Value(Object o) : type_(Type::kObject), obj_(std::make_shared<Object>(std::move(o))) {}

  Type type() const { return type_; }
  static Value object() { Value v; v.ensure_object(); return v; }
  static Value array()  { Value v; v.ensure_array();  return v; }
  bool is_null()   const { return type_ == Type::kNull; }
  bool is_bool()   const { return type_ == Type::kBool; }
  bool is_int()    const { return type_ == Type::kInt; }
  bool is_double() const { return type_ == Type::kDouble; }
  bool is_number() const { return type_ == Type::kInt || type_ == Type::kDouble; }
  bool is_string() const { return type_ == Type::kString; }
  bool is_array()  const { return type_ == Type::kArray; }
  bool is_object() const { return type_ == Type::kObject; }

  bool        as_bool(bool dflt = false) const { return is_bool() ? bool_ : dflt; }
  int64_t     as_int(int64_t dflt = 0) const {
    if (is_int()) return int_;
    if (is_double()) return static_cast<int64_t>(dbl_);
    return dflt;
  }
  double      as_double(double dflt = 0.0) const {
    if (is_double()) return dbl_;
    if (is_int()) return static_cast<double>(int_);
    return dflt;
  }
  const std::string& as_string() const { static const std::string kEmpty; return is_string() ? str_ : kEmpty; }
  const Array&  as_array()  const { static const Array kEmpty; return is_array() ? *arr_ : kEmpty; }
  const Object& as_object() const { static const Object kEmpty; return is_object() ? *obj_ : kEmpty; }

  // Object convenience
  const Value* find(const std::string& key) const {
    if (!is_object()) return nullptr;
    auto it = obj_->find(key);
    return it == obj_->end() ? nullptr : &it->second;
  }
  Value& operator[](const std::string& key) { ensure_object(); return (*obj_)[key]; }
  void set(const std::string& key, Value v) { ensure_object(); (*obj_)[key] = std::move(v); }

  // Array convenience
  void push(Value v) { ensure_array(); arr_->push_back(std::move(v)); }
  size_t size() const { return is_array() ? arr_->size() : (is_object() ? obj_->size() : 0); }

  std::string dump() const {
    std::string out;
    dump_to(out);
    return out;
  }

  static bool parse(const std::string& text, Value* out);

 private:
  void ensure_object() { if (type_ != Type::kObject) { type_ = Type::kObject; obj_ = std::make_shared<Object>(); } }
  void ensure_array()  { if (type_ != Type::kArray)  { type_ = Type::kArray;  arr_ = std::make_shared<Array>(); } }
  void dump_to(std::string& out) const;

  Type type_;
  bool bool_ = false;
  int64_t int_ = 0;
  double dbl_ = 0.0;
  std::string str_;
  std::shared_ptr<Array> arr_;
  std::shared_ptr<Object> obj_;
};

// ---- serialization details -------------------------------------------------
inline void dump_string(const std::string& s, std::string& out) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

inline void Value::dump_to(std::string& out) const {
  char buf[40];
  switch (type_) {
    case Type::kNull:   out += "null"; break;
    case Type::kBool:   out += bool_ ? "true" : "false"; break;
    case Type::kInt:
      std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(int_));
      out += buf; break;
    case Type::kDouble: {
      std::snprintf(buf, sizeof(buf), "%.6f", dbl_);
      out += buf; break;
    }
    case Type::kString: dump_string(str_, out); break;
    case Type::kArray: {
      out += '[';
      bool first = true;
      for (const auto& v : *arr_) {
        if (!first) out += ',';
        first = false;
        v.dump_to(out);
      }
      out += ']'; break;
    }
    case Type::kObject: {
      out += '{';
      bool first = true;
      for (const auto& [k, v] : *obj_) {
        if (!first) out += ',';
        first = false;
        dump_string(k, out);
        out += ':';
        v.dump_to(out);
      }
      out += '}'; break;
    }
  }
}

// ---- parser ----------------------------------------------------------------
namespace detail {

struct Parser {
  const char* p;
  const char* end;
  bool ok = true;
  Value* out_val = nullptr;

  void skip_ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }

  bool parse_value(Value* out, int depth) {
    if (depth > 64 || p >= end) { ok = false; return false; }
    skip_ws();
    if (p >= end) { ok = false; return false; }
    char c = *p;
    if (c == '{') return parse_object(out, depth);
    if (c == '[') return parse_array(out, depth);
    if (c == '"') { std::string s; if (!parse_string(&s)) return false; *out = Value(std::move(s)); return true; }
    if (c == 't') { return parse_lit("true",  Value(true)); }
    if (c == 'f') { return parse_lit("false", Value(false)); }
    if (c == 'n') { return parse_lit("null",  Value()); }
    return parse_number(out);
  }

  bool parse_lit(const char* lit, Value v) {
    size_t n = std::char_traits<char>::length(lit);
    if (static_cast<size_t>(end - p) < n || std::char_traits<char>::compare(p, lit, n) != 0) { ok = false; return false; }
    p += n; *out_val = std::move(v); return true;
  }

  bool parse_string(std::string* out) {
    ++p;  // opening quote
    while (p < end) {
      unsigned char c = static_cast<unsigned char>(*p);
      if (c == '"') { ++p; return true; }
      if (c == '\\') {
        ++p;
        if (p >= end) break;
        switch (*p) {
          case '"':  out->push_back('"');  break;
          case '\\': out->push_back('\\'); break;
          case '/':  out->push_back('/');  break;
          case 'b':  out->push_back('\b'); break;
          case 'f':  out->push_back('\f'); break;
          case 'n':  out->push_back('\n'); break;
          case 'r':  out->push_back('\r'); break;
          case 't':  out->push_back('\t'); break;
          case 'u': {
            if (end - p < 5) { ok = false; return false; }
            unsigned code = 0;
            for (int i = 1; i <= 4; ++i) {
              char h = p[i]; code <<= 4;
              if (h >= '0' && h <= '9') code |= static_cast<unsigned>(h - '0');
              else if (h >= 'a' && h <= 'f') code |= static_cast<unsigned>(h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') code |= static_cast<unsigned>(h - 'A' + 10);
              else { ok = false; return false; }
            }
            p += 4;
            // UTF-8 encode (BMP only; surrogates are not used by our protocol)
            if (code < 0x80) out->push_back(static_cast<char>(code));
            else if (code < 0x800) {
              out->push_back(static_cast<char>(0xC0 | (code >> 6)));
              out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              out->push_back(static_cast<char>(0xE0 | (code >> 12)));
              out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default: ok = false; return false;
        }
        ++p;
      } else {
        out->push_back(static_cast<char>(c));
        ++p;
      }
    }
    ok = false;
    return false;
  }

  bool parse_number(Value* out) {
    bool is_double = false;
    const char* start = p;
    if (p < end && (*p == '-' || *p == '+')) ++p;
    while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) {
      if (*p == '.' || *p == 'e' || *p == 'E') is_double = true;
      ++p;
    }
    if (p == start) { ok = false; return false; }
    std::string num(start, p);
    if (!is_double) {
      try { *out = Value(static_cast<int64_t>(std::stoll(num))); return true; }
      catch (...) { is_double = true; }
    }
    try { *out = Value(std::stod(num)); return true; }
    catch (...) { ok = false; return false; }
  }

  bool parse_object(Value* out, int depth) {
    Object o; ++p;  // {
    skip_ws();
    if (p < end && *p == '}') { ++p; *out = Value(std::move(o)); return true; }
    while (p < end) {
      skip_ws();
      if (p >= end || *p != '"') { ok = false; return false; }
      std::string key;
      if (!parse_string(&key)) return false;
      skip_ws();
      if (p >= end || *p != ':') { ok = false; return false; }
      ++p;
      Value v;
      if (!parse_value(&v, depth + 1)) return false;
      o.emplace(std::move(key), std::move(v));
      skip_ws();
      if (p < end && *p == ',') { ++p; continue; }
      if (p < end && *p == '}') { ++p; *out = Value(std::move(o)); return true; }
      ok = false; return false;
    }
    ok = false; return false;
  }

  bool parse_array(Value* out, int depth) {
    Array a; ++p;  // [
    skip_ws();
    if (p < end && *p == ']') { ++p; *out = Value(std::move(a)); return true; }
    while (p < end) {
      Value v;
      if (!parse_value(&v, depth + 1)) return false;
      a.push_back(std::move(v));
      skip_ws();
      if (p < end && *p == ',') { ++p; continue; }
      if (p < end && *p == ']') { ++p; *out = Value(std::move(a)); return true; }
      ok = false; return false;
    }
    ok = false; return false;
  }
};

}  // namespace detail

inline bool Value::parse(const std::string& text, Value* out) {
  detail::Parser ps{text.data(), text.data() + text.size(), true, nullptr};
  Value v;
  ps.out_val = &v;
  if (!ps.parse_value(&v, 0)) return false;
  ps.skip_ws();
  if (ps.p != ps.end) return false;  // trailing garbage is an error
  *out = std::move(v);
  return true;
}

}  // namespace sager::json
