#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "data.hpp"

inline auto AmendArgs(int argc, char* argv[]) {
  std::vector<char*> new_argv(argv, argv + argc);
  new_argv.push_back(const_cast<char*>("--benchmark_counters_tabular=true"));
  new_argv.push_back(const_cast<char*>("--benchmark_out_format=json"));
  return new_argv;
}

/// Calls `fn<T>()` for every benchmarked data type, e.g. to generate data
/// before the benchmarks run.
#define FOR_ALL_BENCHMARK_TYPES(fn) \
  fn<int16_t>();                    \
  fn<int32_t>();                    \
  fn<int64_t>();                    \
  fn<__int128_t>();                 \
  fn<float>();                      \
  fn<double>();                     \
  fn<std::string>()

inline int RunBenchmarks(int argc, char** argv) {
  auto new_argv = AmendArgs(argc, argv);
  argc = static_cast<int>(new_argv.size());
  ::benchmark::Initialize(&argc, new_argv.data());
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}

#define CUSTOM_BENCHMARK_MAIN(cache_data, cache_hashes) \
  int main(int argc, char** argv) {                     \
    if (cache_data) {                                   \
      FOR_ALL_BENCHMARK_TYPES(GetData);                 \
    }                                                   \
    if (cache_hashes) {                                 \
      FOR_ALL_BENCHMARK_TYPES(GetHashes);               \
    }                                                   \
    return RunBenchmarks(argc, argv);                   \
  }
