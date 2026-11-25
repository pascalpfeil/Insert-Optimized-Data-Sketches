// Same as bm_insert.cpp, but on Zipf(1.0) distributed data (see GetZipfData).
// Skewed data changes the picture for the sketches whose work depends on the
// values seen before: SpaceSaving mostly hits its monitored set, and the
// counting sketches repeatedly touch the same counters.
#include "benchmark.hpp"
#include "benchmark/benchmark.h"
#include "cs/cs_datasketches.hpp"
#include "cs/cs_fastrange.hpp"
#include "cs/cs_final.hpp"
#include "cs/cs_final_no_murmur_unroll.hpp"
#include "cs/cs_fixed_size.hpp"
#include "cs/cs_naive.hpp"
#include "data.hpp"
#include "kll/kll_branchless.hpp"
#include "kll/kll_cached_level_capacities.hpp"
#include "kll/kll_datasketches.hpp"
#include "kll/kll_final.hpp"
#include "kll/kll_naive.hpp"
#include "kll/kll_no_min_max.hpp"
#include "kll/kll_no_self_move_protection.hpp"
#include "kll/kll_pcg_random.hpp"
#include "ss/ss_datasketches.hpp"
#include "ss/ss_final.hpp"
#include "ss/ss_fingerprint.hpp"
#include "ss/ss_heap.hpp"
#include "ss/ss_map.hpp"
#include "ss/ss_naive.hpp"
#include "types.hpp"

template <typename Sketch, typename T>
void BM_InsertZipf(benchmark::State& state) {
  const auto& data = GetZipfData<T>();
  for (auto _ : state) {
    Sketch sketch;
    for (const auto& value : data) {
      sketch.Insert(value);
    }
    ::benchmark::DoNotOptimize(sketch);
    ::benchmark::ClobberMemory();
  }

  int64_t num_items = state.iterations() * data.size();
  state.SetItemsProcessed(num_items);
  int64_t item_size = sizeof(T);
  if constexpr (detail::is_string_v<T>) {
    item_size = data[0].size() * sizeof(char);
  }
  state.SetBytesProcessed(num_items * item_size);
  state.counters["item_size"] = item_size;
}

#define BENCHMARK_INSERT_ZIPF_TYPE(sketch, type) \
  BENCHMARK_TEMPLATE(BM_InsertZipf, sketch<type>, type)

#define BENCHMARK_INSERT_ZIPF_ALL_TYPES(sketch)   \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, int16_t);    \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, int32_t);    \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, int64_t);    \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, __int128_t); \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, float);      \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, double);     \
  BENCHMARK_INSERT_ZIPF_TYPE(sketch, std::string)

BENCHMARK_INSERT_ZIPF_ALL_TYPES(datasketches::SpaceSaving);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(naive::SpaceSaving);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(map::SpaceSaving);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(heap::SpaceSaving);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(final::SpaceSaving);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(fingerprint::SpaceSaving);

BENCHMARK_INSERT_ZIPF_ALL_TYPES(datasketches::CountMinSketch);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(naive::CountSketch);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(fastrange::CountSketch);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(fixed_size::CountSketch);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(final_no_murmur_unroll::CountSketch);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(final::CountSketch);

BENCHMARK_INSERT_ZIPF_ALL_TYPES(naive::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(datasketches::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(no_min_max::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(no_self_move_protection::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(cached_level_capacities::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(pcg_random::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(final::KarninLangLiberty);
BENCHMARK_INSERT_ZIPF_ALL_TYPES(branchless::KarninLangLiberty);

int main(int argc, char** argv) {
  FOR_ALL_BENCHMARK_TYPES(GetZipfData);
  return RunBenchmarks(argc, argv);
}
