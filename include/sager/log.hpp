// log.hpp - minimal leveled logging. Header-only, no dependencies.
//
// Level is controlled with SAGER_LOG (debug|info|warn|error|off, default info)
// and JSON-lines protocol messages on stdout are NEVER mixed with logs: the
// engine logs to stderr only. This separation is load-bearing for serve/.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <utility>

namespace sager::log {

enum class Level : int { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3, kOff = 4 };

inline Level level_from_env() {
  const char* e = std::getenv("SAGER_LOG");
  if (!e) return Level::kInfo;
  switch (e[0]) {
    case 'd': case 'D': return Level::kDebug;
    case 'i': case 'I': return Level::kInfo;
    case 'w': case 'W': return Level::kWarn;
    case 'e': case 'E': return Level::kError;
    default:            return Level::kOff;
  }
}

inline Level min_level() {
  static Level lvl = level_from_env();
  return lvl;
}

template <typename... Args>
inline void write(Level lvl, const char* tag, const char* fmt, Args&&... args) {
  if (lvl < min_level()) return;
  static const char* names[] = {"DEBUG", "INFO", "WARN", "ERROR"};
  std::fprintf(stderr, "[%s sager/%s] ", names[static_cast<int>(lvl)], tag);
  std::fprintf(stderr, fmt, std::forward<Args>(args)...);
  std::fprintf(stderr, "\n");
}

}  // namespace sager::log

#define SLOG_D(tag, ...) ::sager::log::write(::sager::log::Level::kDebug, tag, __VA_ARGS__)
#define SLOG_I(tag, ...) ::sager::log::write(::sager::log::Level::kInfo,  tag, __VA_ARGS__)
#define SLOG_W(tag, ...) ::sager::log::write(::sager::log::Level::kWarn,  tag, __VA_ARGS__)
#define SLOG_E(tag, ...) ::sager::log::write(::sager::log::Level::kError, tag, __VA_ARGS__)
