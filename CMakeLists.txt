cmake_minimum_required(VERSION 3.24)

project(sager
  VERSION 0.1.0
  DESCRIPTION "SAGER: heterogeneous expert-tiered inference engine for Sage-A1.5 (35B MoE) on a single T4"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SAGER_ENABLE_CUDA   "Build the CUDA (sm_75 / T4) backend"  ON)
option(SAGER_ENABLE_AVX2   "Build the AVX2 CPU expert pool"       ON)
option(SAGER_BUILD_TESTS   "Build host-side unit tests"           ON)
option(SAGER_WERROR        "Treat warnings as errors"             OFF)

if(SAGER_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    # Host TUs (sager_core) include <cuda_runtime.h> under SAGER_HAVE_CUDA,
    # so the toolkit's include dirs + libcudart must reach plain CXX targets.
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # T4 = compute capability 7.5 (Turing). Keep the fatbin small and targeted;
    # add more -gencode entries to cover other cards.
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "")
    message(STATUS "SAGER: CUDA enabled, arch=${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "SAGER: SAGER_ENABLE_CUDA=ON but no CUDA compiler found - building CPU-only")
    set(SAGER_ENABLE_CUDA OFF)
  endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Core library (host side: GGUF reader, config, expert cache, session loop,
# speculative controller, CPU expert pool). CUDA is linked separately.
# ---------------------------------------------------------------------------
add_library(sager_core STATIC
  src/core/common.cpp
  src/core/config.cpp
  src/core/gguf.cpp
  src/core/weights.cpp
  src/core/expert_cache.cpp
  src/core/expert_pump.cpp
  src/core/session.cpp
  src/core/layer.cpp
  src/core/verify.cpp
  src/core/mtp.cpp
  src/core/conversation.cpp
  src/spec/controller.cpp
  src/spec/suffix_drafter.cpp
  src/prefill/prefill.cpp
  src/kernels/q4_k.cpp
  src/kernels/cpu/q4_avx2.cpp
  src/kernels/cpu/pool.cpp
  src/kernels/cpu/gdn_cpu.cpp
)

target_include_directories(sager_core PUBLIC include)
target_link_libraries(sager_core PUBLIC Threads::Threads)

if(SAGER_ENABLE_AVX2)
  if(MSVC)
    target_compile_options(sager_core PRIVATE /arch:AVX2)
  else()
    target_compile_options(sager_core PRIVATE -mavx2 -mfma)
  endif()
  target_compile_definitions(sager_core PUBLIC SAGER_HAVE_AVX2=1)
endif()

if(MSVC)
  target_compile_options(sager_core PRIVATE /W4)
else()
  target_compile_options(sager_core PRIVATE -Wall -Wextra -Wpedantic)
  if(SAGER_WERROR)
    target_compile_options(sager_core PRIVATE -Werror)
  endif()
endif()

# ---------------------------------------------------------------------------
# CUDA backend (Q4_K kernels, router/MoE, GQA attention, RoPE/norm, sampling).
# Every kernel is tuned first for sm_75 (Tesla T4): dp4a int8 dots for the
# 4-bit dequant-GEMV path, with an optional Turing int4-tensor-core path.
# ---------------------------------------------------------------------------
if(SAGER_ENABLE_CUDA)
  add_library(sager_cuda STATIC
    src/kernels/cuda/q4_k.cu
    src/kernels/cuda/moe.cu
    src/kernels/cuda/attention.cu
    src/kernels/cuda/norm_rotary.cu
    src/kernels/cuda/sampling.cu
    src/kernels/cuda/gdn.cu
  )
  target_include_directories(sager_cuda PUBLIC include)
  target_compile_definitions(sager_cuda PUBLIC SAGER_HAVE_CUDA=1)
  # Whole-program per-TU compilation: the kernel files share no device
  # symbols, and skipping the device-link step avoids the RDC registration
  # pitfall when the final executable is CXX-only.
  set_property(TARGET sager_cuda PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(sager_core PUBLIC sager_cuda CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# Engine executable: resident process speaking JSON-lines on stdin/stdout
# (the Python serving layer in serve/ spawns and talks to it).
# ---------------------------------------------------------------------------
add_executable(sager_engine
  src/program/engine_main.cpp
  src/program/generate.cpp
)
target_link_libraries(sager_engine PRIVATE sager_core)

# Host-side tests (pure CPU, no GPU required in CI).
if(SAGER_BUILD_TESTS)
  enable_testing()
  add_executable(sager_test_q4
    tests/test_q4_host.cpp
  )
  target_link_libraries(sager_test_q4 PRIVATE sager_core)
  add_test(NAME q4_host_roundtrip COMMAND sager_test_q4)

  # v0.2: GDN reference math vs numpy golden vectors + delta-rule properties.
  add_executable(sager_test_gdn
    tests/test_gdn_host.cpp
  )
  target_link_libraries(sager_test_gdn PRIVATE sager_core)
  add_test(NAME gdn_host_reference COMMAND sager_test_gdn)

  # v0.2: expert pump residency policy (LRU, reuse, in-flight safety).
  add_executable(sager_test_pump
    tests/test_expert_pump.cpp
  )
  target_link_libraries(sager_test_pump PRIVATE sager_core)
  add_test(NAME expert_pump_policy COMMAND sager_test_pump)
endif()

install(TARGETS sager_engine RUNTIME DESTINATION bin)
install(DIRECTORY serve/ DESTINATION share/sager/serve FILES_MATCHING PATTERN "*.py")
install(DIRECTORY serve/web DESTINATION share/sager/serve)
