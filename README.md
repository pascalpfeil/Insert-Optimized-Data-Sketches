# Insert-Optimized Implementation of Streaming Data Sketches
This repository contains the implementations accompanying our paper "Insert-Optimized Implementation of Streaming Data Sketches."

The codebase includes our optimized implementations alongside baseline versions and Apache DataSketches implementations. You'll also find the microbenchmarking code, results from our experiments, and visualization scripts for analyzing the results.

*Note: Due to licensing restrictions, implementations from compared papers are not included in this repository.*

## Build
To build the project, simply run:
```bash
./build.sh
```
This script uses CMake with GCC in release mode  by default, and creates all build artifacts in the `cmake-build-release`.

## Run Benchmarks
After building, execute the following commands to run the benchmarks:

```
cmake-build-release/bm_insert --benchmark_out="results/bm_insert.json" --benchmark_min_time=10s

cmake-build-release/bm_hash --benchmark_out="results/bm_hash.json" --benchmark_min_time=10s

cmake-build-release/bm_hash_insert --benchmark_out="results/bm_hash_insert.json" --benchmark_min_time=10s
```

### Zipf-distributed data
`bm_insert` uses uniformly distributed values, so nearly every value is distinct. `bm_insert_zipf` runs the same sketches on Zipf(1.0) data (1M values over 1M distinct ones; the most frequent value is ~7% of the stream and ~217K distinct values appear), where SpaceSaving mostly hits its monitored set and the counting sketches repeatedly touch the same counters. The generator is `GetZipfData<T>()` in `include/data.hpp`.

```
cmake-build-release/bm_insert_zipf --benchmark_out="results/bm_insert_zipf.json" --benchmark_min_time=10s
```

## Plot
Execute the Jupyter notebook `analysis.ipynb` to generate the plots. `RESULTS_DIR` at its top selects the measurements:
- `results/`: the measurements from the paper; plots are saved to `figures/`.
- `results/gcc16_xeon8259cl/` (default) and `results/gcc7_xeon8259cl/`: measured later with GCC 16 and GCC 7.3 on a Xeon Platinum 8259CL, including the implementations below; plots are saved to `figures/<name>/`.

`analysis_zipf.ipynb` puts the uniform and Zipf results of `RESULTS_DIR` next to each other.

## Additional Implementations
Two implementations added after the paper. Both keep the sketch parameters and semantics of the paper's `final` versions.

### Fingerprint SpaceSaving
`src/ss/ss_fingerprint.hpp` (`fingerprint::SpaceSaving`): K = 96 monitored items, a miss replaces a minimum-count item and gives it count min + 1, as in `final::SpaceSaving`, but with a different data structure:
- The scanned array holds 16-bit fingerprints of the keys instead of the keys, so a lookup is three AVX-512 compares (six with AVX2) for every key type, including strings and 128-bit integers. A fingerprint match is verified against the full key, so results are exact.
- No heap. The sketch keeps the minimum count and a list of the slots that have it. A hit increments one counter; a miss pops a slot off the list (slots whose count has grown are skipped). When the list runs empty it is rebuilt with one SIMD min-reduction and compare, at most once per K misses.
- A miss rewrites the whole fingerprint vector, so the next lookup's vector load can take it from the store buffer (a 2-byte store followed by a 64-byte load of the same line cannot forward).
- Strings are fingerprinted with CRC32C instead of MurmurHash3. The string benchmark is then bound by reading 1M separately allocated strings of about 320 bytes, not by hashing.

`Estimate()` and `Items()` allow checking against exact counts. `-DSS_NO_AVX512` selects the AVX2 code path on an AVX-512 machine.

### Branchless KLL
`src/kll/kll_branchless.hpp` (`branchless::KarninLangLiberty`) is `final::KarninLangLiberty` with the data-dependent branches taken out of compaction. With k = 200 and m = 8, level 0 holds 8 to 16 items most of the time, so a compaction sorts and merges a handful of items every ~14 inserts, and the comparisons in `std::sort`'s insertion sort and in the merge mispredict about half the time. For arithmetic types up to 64 bits, level 0 is sorted with padded sorting networks (8 to 64 items; compare-exchanges compile to cmov or vminsd/vmaxsd), and compactions are merged with merging networks (up to 32 + 32 items) that read the kept half straight from the sorted level, or with a two-ended branch-free merge for larger levels. Other types keep `std::sort` and the DataSketches merge. `get_rank()` was added so the estimates can be checked against exact ranks.

## Benchmarking Your Own Sketch Implementation
1. Add your implementation to the `src` directory
2. Integrate it into one of the existing benchmarks in `bench/`
3. Ensure your implementation uses these sketch parameters:
   - Count Sketch: `t = 2048`, `d = 5`
   - SpaceSaving: `k = 96`
   - Karnin-Lang-Liberty: `k = 200`
4. Follow the build, run, and plot instructions above

# Citation

Our paper is available [here](https://www.amazon.science/publications/insert-optimized-implementation-of-streaming-data-sketches).
```
@Article{Pfeil2025,
 author = {Pascal Pfeil and Dominik Horn and Orestis Polychroniou and George Erickson and Zhe Heng Eng and Mengchu Cai and Tim Kraska},
 title = {Insert-Optimized Implementation of Streaming Data Sketches},
 year = {2025},
 url = {https://www.amazon.science/publications/insert-optimized-implementation-of-streaming-data-sketches},
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

