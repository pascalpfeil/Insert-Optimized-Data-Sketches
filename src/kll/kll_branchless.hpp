// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// Based on final::KarninLangLiberty (itself based on Apache DataSketches 4.1.0,
// see kll_final.hpp). Same k, m, level capacities, compaction rule and random
// source, so the sketch is statistically the same. What changed:
//
// - Compaction has no data-dependent branches for arithmetic types up to 64
//   bits under std::less. Level 0 is sorted with padded sorting networks (8 to
//   64 items); the kept half of a level is merged into the level above with a
//   merging network (up to 32 + 32 items) or a two-ended branch-free merge.
//   Compare-exchanges compile to cmov, or to vminsd/vmaxsd for floats, which
//   run through one SSE register each (see Lane). Other types keep std::sort
//   and the DataSketches merge.
//   Why: the capacity of level 0 is k * (2/3)^(levels - 1) clamped to m = 8,
//   so after a few thousand inserts level 0 holds 8 to 16 items, every ~14
//   inserts trigger a compaction, and sorting and merging those few items with
//   data-dependent branches cost about 2.5 mispredictions per insert.
// - levels_ is a fixed array inside the object instead of a std::vector.
// - Random bits come from a buffered pcg32_fast output, one call per 32 bits.
// - n_ and is_level_zero_sorted_ are gone: n() is the total weight of the
//   levels, and nothing ever set is_level_zero_sorted_.
// - get_rank() was added to check the estimates against exact ranks.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "compiler.hpp"
#include "const.hpp"
#include "pcg_random.hpp"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace branchless {

namespace kll_constants {
const uint16_t DEFAULT_K = 200;
const uint8_t DEFAULT_M = 8;
const uint16_t MIN_K = DEFAULT_M;
const uint16_t MAX_K = (1 << 16) - 1;
}  // namespace kll_constants

namespace kll_detail {

/// c ? t : f, computed with a mask so that no compiler turns it into a branch
/// (GCC does so for `?:` on doubles, which have no cmov, and often on int64).
template <typename T>
ALWAYS_INLINE T select(bool c, T t, T f) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
  using U = std::conditional_t<sizeof(T) == 2, uint16_t,
                               std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>;
  U ut, uf;
  std::memcpy(&ut, &t, sizeof(T));
  std::memcpy(&uf, &f, sizeof(T));
  const U mask = U(0) - U(c);
  const U r = (ut & mask) | (uf & ~mask);
  T out;
  std::memcpy(&out, &r, sizeof(T));
  return out;
}

/// Compare-exchange: a <= b afterwards, no branch. Integers get the larger
/// value by xor so that GCC does not turn the min/max pair into a branch.
template <typename T>
ALWAYS_INLINE void cx(T& a, T& b) {
  if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    const T lo = b < a ? b : a;
    b = static_cast<T>(static_cast<U>(a) ^ static_cast<U>(b) ^
                       static_cast<U>(lo));
    a = lo;
#if defined(__SSE2__)
  } else if constexpr (std::is_same_v<T, __m128d>) {
    // One double per register (see Lane). NaNs never get here, update()
    // drops them, so min/max operand order does not matter.
    const __m128d lo = _mm_min_sd(a, b);
    b = _mm_max_sd(a, b);
    a = lo;
  } else if constexpr (std::is_same_v<T, __m128>) {
    const __m128 lo = _mm_min_ss(a, b);
    b = _mm_max_ss(a, b);
    a = lo;
#endif
  } else {
    static_assert(std::is_floating_point_v<T>);
    // GCC 16 compiles this to vminsd/vmaxsd, GCC 7 to a branch; on x86 the
    // networks therefore run on Lane<T>::type instead of on T.
    const T lo = b < a ? b : a;
    const T hi = b < a ? a : b;
    a = lo;
    b = hi;
  }
}

/// Batcher's odd-even merge sort network for 8 items, 19 comparators.
template <typename T>
ALWAYS_INLINE void sort8(T* v) {
  cx(v[0], v[1]); cx(v[2], v[3]); cx(v[4], v[5]); cx(v[6], v[7]);
  cx(v[0], v[2]); cx(v[1], v[3]); cx(v[4], v[6]); cx(v[5], v[7]);
  cx(v[1], v[2]); cx(v[5], v[6]); cx(v[0], v[4]); cx(v[3], v[7]);
  cx(v[1], v[5]); cx(v[2], v[6]);
  cx(v[1], v[4]); cx(v[3], v[6]);
  cx(v[2], v[4]); cx(v[3], v[5]);
  cx(v[3], v[4]);
}

template <unsigned I, unsigned Stop, unsigned Step, unsigned R, typename T>
ALWAYS_INLINE void oe_merge_cx(T* a) {
  if constexpr (I < Stop) {
    cx(a[I], a[I + R]);
    oe_merge_cx<I + Step, Stop, Step, R>(a);
  }
}

/// Batcher's odd-even merging network: merges the two sorted halves of
/// a[Lo..Hi] (inclusive bounds, elements R apart) in place.
template <unsigned Lo, unsigned Hi, unsigned R, typename T>
ALWAYS_INLINE void oe_merge(T* a) {
  constexpr unsigned step = R * 2;
  if constexpr (step < Hi - Lo) {
    oe_merge<Lo, Hi, step>(a);
    oe_merge<Lo + R, Hi, step>(a);
    oe_merge_cx<Lo + R, Hi - R, step, R>(a);
  } else {
    cx(a[Lo], a[Lo + R]);
  }
}

/// Sorting network for N = 8, 16, 32, ... items.
template <unsigned N, typename T>
ALWAYS_INLINE void sort_network(T* v) {
  static_assert(N >= 8 && (N & (N - 1)) == 0, "N must be a power of two >= 8");
  if constexpr (N == 8) {
    sort8(v);
  } else {
    sort_network<N / 2>(v);
    sort_network<N / 2>(v + N / 2);
    oe_merge<0, N - 1, 1>(v);
  }
}

/// Merge sorted a[0, na) and b[0, nb) into out[0, na + nb), a before b on
/// ties, without data-dependent branches. `out` must not overlap a or b.
/// The smallest half is merged from the front and the largest half from the
/// back at the same time; the two chains are independent, so the core
/// overlaps them (a single merge chain is bound by its load -> compare ->
/// index update latency). Indices are clamped instead of checked, so an
/// exhausted input is never read out of bounds.
template <typename T>
void merge(const T* a, uint32_t na, const T* b, uint32_t nb, T* out) {
  if (nb == 0) {
    std::copy(a, a + na, out);
    return;
  }
  if (na == 0) {
    std::copy(b, b + nb, out);
    return;
  }
  const uint32_t n = na + nb;
  const uint32_t front = n / 2;  // outputs taken from the front
  uint32_t i = 0, j = 0;         // next unconsumed from the front
  uint32_t ri = na, rj = nb;     // one past the next unconsumed from the back
  for (uint32_t k = 0; k < front; ++k) {
    const T x = a[std::min(i, na - 1)];
    const T y = b[std::min(j, nb - 1)];
    const bool take_b = (i == na) | ((j != nb) & (y < x));
    out[k] = select(take_b, y, x);
    i += !take_b;
    j += take_b;

    const T rx = a[ri - (ri > 0)];
    const T ry = b[rj - (rj > 0)];
    const bool rtake_b = (ri == 0) | ((rj != 0) & !(ry < rx));
    out[n - 1 - k] = select(rtake_b, ry, rx);
    ri -= !rtake_b;
    rj -= rtake_b;
  }
  if (n & 1) {  // exactly one element is left, in a (i < ri) or in b
    out[front] = select(i < ri, a[std::min(i, na - 1)], b[std::min(j, nb - 1)]);
  }
}

/// Largest value of T, used to pad partial blocks of the networks. (+inf items
/// tie with the padding, which is harmless: equal items may be in any order.)
template <typename T>
constexpr T pad_value() {
  if constexpr (std::is_floating_point_v<T>) {
    return std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::max();
  }
}

/// The register type the networks run on. Floats and doubles use an SSE
/// register each, so every compare-exchange is one vminsd and one vmaxsd
/// (GCC 7 compiles the scalar ternary form into a branch).
template <typename T>
struct Lane {
  using type = T;
  static ALWAYS_INLINE type load(const T* p) { return *p; }
  static ALWAYS_INLINE void store(T* p, type v) { *p = v; }
  static ALWAYS_INLINE type pad() { return pad_value<T>(); }
};
#if defined(__SSE2__)
template <>
struct Lane<double> {
  using type = __m128d;
  static ALWAYS_INLINE type load(const double* p) { return _mm_load_sd(p); }
  static ALWAYS_INLINE void store(double* p, type v) { _mm_store_sd(p, v); }
  static ALWAYS_INLINE type pad() { return _mm_set_sd(pad_value<double>()); }
};
template <>
struct Lane<float> {
  using type = __m128;
  static ALWAYS_INLINE type load(const float* p) { return _mm_load_ss(p); }
  static ALWAYS_INLINE void store(float* p, type v) { _mm_store_ss(p, v); }
  static ALWAYS_INLINE type pad() { return _mm_set_ss(pad_value<float>()); }
};
#endif

/// Sorts v[0, n), n <= N, with the N-item network; v is padded with the
/// largest value in a temporary when n < N.
template <unsigned N, typename T>
ALWAYS_INLINE void sort_padded(T* v, uint32_t n) {
  using L = Lane<T>;
  if constexpr (std::is_same_v<typename L::type, T>) {
    if (n == N) {
      sort_network<N>(v);
      return;
    }
  }
  typename L::type tmp[N];
  for (uint32_t i = 0; i < n; ++i) tmp[i] = L::load(v + i);
  for (uint32_t i = n; i < N; ++i) tmp[i] = L::pad();
  sort_network<N>(tmp);
  for (uint32_t i = 0; i < n; ++i) L::store(v + i, tmp[i]);
}

/// Merges the sorted items a[0], a[StrideA], ... (na of them) and b[0, nb),
/// na and nb <= N / 2, into out with the N-item merging network, padding both
/// inputs with the largest value. StrideA = 2 merges every other item of a
/// sorted level, which is what compaction keeps.
template <unsigned N, unsigned StrideA, typename T>
ALWAYS_INLINE void merge_padded(const T* a, uint32_t na, const T* b, uint32_t nb, T* out) {
  using L = Lane<T>;
  typename L::type tmp[N];
  for (uint32_t i = 0; i < na; ++i) tmp[i] = L::load(a + i * StrideA);
  for (uint32_t i = na; i < N / 2; ++i) tmp[i] = L::pad();
  for (uint32_t i = 0; i < nb; ++i) tmp[N / 2 + i] = L::load(b + i);
  for (uint32_t i = N / 2 + nb; i < N; ++i) tmp[i] = L::pad();
  oe_merge<0, N - 1, 1>(tmp);
  for (uint32_t i = 0; i < na + nb; ++i) L::store(out + i, tmp[i]);
}

/// Sorts v[0, n) without data-dependent branches. Up to 64 items (99.5% of
/// level-0 compactions with k = 200) a single padded sorting network does it;
/// beyond that, 64-item blocks are sorted with the network and merged bottom-up
/// through `scratch` (room for n items).
template <typename T>
void sort(T* v, uint32_t n, T* scratch) {
  if (n <= 8) return sort_padded<8>(v, n);
  if (n <= 16) return sort_padded<16>(v, n);
  if (n <= 32) return sort_padded<32>(v, n);
  if (n <= 64) return sort_padded<64>(v, n);
  constexpr uint32_t kBlock = 64;
  for (uint32_t i = 0; i < n; i += kBlock) sort_padded<kBlock>(v + i, std::min(kBlock, n - i));
  T* src = v;
  T* dst = scratch;
  for (uint32_t width = kBlock; width < n; width *= 2) {
    for (uint32_t lo = 0; lo < n; lo += 2 * width) {
      const uint32_t mid = std::min(lo + width, n);
      const uint32_t hi = std::min(lo + 2 * width, n);
      merge(src + lo, mid - lo, src + mid, hi - mid, dst + lo);
    }
    std::swap(src, dst);
  }
  if (src != v) std::copy(src, src + n, v);
}

}  // namespace kll_detail

template <typename T, typename C = std::less<T>, typename A = std::allocator<T>>
class KarninLangLiberty {
  /// Branch-free sorting and merging need a strict weak order the hardware
  /// can evaluate with one compare: arithmetic types under std::less. 128-bit
  /// integers qualify but lose: 16 of them need 32 registers, and the spilled
  /// network is slower than std::sort.
  static constexpr bool kBranchless =
      std::is_arithmetic_v<T> && sizeof(T) <= 8 && std::is_same_v<C, std::less<T>>;

 public:
  using value_type = T;
  using comparator = C;
  using allocator_type = A;

  explicit KarninLangLiberty(uint16_t k = 200, const C& comparator = C(),
                             const A& allocator = A())
      : comparator_(comparator),
        allocator_(allocator),
        k_(k),
        level_capacities_(compute_level_capacities(k, m_)),
        max_capacity_(compute_total_capacity(kMaxNumLevels)) {
    if (k < kll_constants::MIN_K || k > kll_constants::MAX_K) {
      throw std::invalid_argument(
          "K must be >= " + std::to_string(kll_constants::MIN_K) + " and <= " +
          std::to_string(kll_constants::MAX_K) + ": " + std::to_string(k));
    }
    levels_.fill(0);
    levels_[0] = levels_[1] = k;
    // The first half of the allocation is scratch space for sorting level 0
    // (which can hold up to the whole sketch); the sketch itself grows
    // downwards from the end of the second half.
    storage_ = allocator_.allocate(2 * max_capacity_);
    items_ = storage_ + 2 * max_capacity_ - k;
  }

  ~KarninLangLiberty() {
    if (storage_ != nullptr) {
      if constexpr (!std::is_trivially_destructible_v<T>) {
        for (uint32_t i = levels_[0]; i < levels_[num_levels_]; i++) {
          items_[i].~T();
        }
      }
      allocator_.deallocate(storage_, 2 * max_capacity_);
    }
  }

  KarninLangLiberty(const KarninLangLiberty&) = delete;
  KarninLangLiberty& operator=(const KarninLangLiberty&) = delete;
  KarninLangLiberty(KarninLangLiberty&&) = delete;
  KarninLangLiberty& operator=(KarninLangLiberty&&) = delete;

  /// Insert a value into the sketch.
  void Insert(const T& x) noexcept { update(x); }
  void Insert(T&& x) noexcept { update(std::move(x)); }

  /// Number of inserted items. Compaction preserves the total weight of the
  /// levels (2^i per item on level i), so this needs no per-insert counter.
  uint64_t n() const noexcept {
    uint64_t n = 0;
    for (uint8_t level = 0; level < num_levels_; ++level) {
      n += uint64_t(levels_[level + 1] - levels_[level]) << level;
    }
    return n;
  }

  /// Estimated fraction of inserted items that are less than x. Items on
  /// level i carry weight 2^i; level 0 is unsorted, the others are sorted.
  double get_rank(const T& x) const {
    const uint64_t n_ = n();
    if (n_ == 0) return 0;
    uint64_t below = 0;
    for (uint8_t level = 0; level < num_levels_; ++level) {
      const T* beg = items_ + levels_[level];
      const T* end = items_ + levels_[level + 1];
      const uint64_t count =
          level == 0 ? std::count_if(beg, end, [&](const T& v) { return comparator_(v, x); })
                     : static_cast<uint64_t>(std::lower_bound(beg, end, x, comparator_) - beg);
      below += count << level;
    }
    return static_cast<double>(below) / static_cast<double>(n_);
  }

 private:
  /// 60 levels are enough to fit std::numeric_limits<size_t>::max() elements.
  static constexpr size_t kMaxNumLevels = 60;
  static constexpr uint8_t m_ = 8;  // minimum level capacity

  C comparator_;
  A allocator_;
  uint16_t k_;
  uint8_t num_levels_ = 1;
  std::array<uint16_t, kMaxNumLevels> level_capacities_;
  size_t max_capacity_;
  /// levels_[i] is the index of the first item of level i; levels_[0] is also
  /// the amount of free space, which sits below level 0.
  std::array<uint32_t, kMaxNumLevels + 2> levels_;
  T* storage_ = nullptr;  // max_capacity_ scratch slots, then the sketch
  T* items_ = nullptr;    // the current sketch region, see constructor
  pcg32_fast rng_;
  uint32_t random_bits_ = 0;
  uint8_t num_random_bits_ = 0;

  bool random_bit() noexcept {
    if (num_random_bits_ == 0) {
      random_bits_ = rng_();
      num_random_bits_ = 32;
    }
    const bool bit = random_bits_ & 1;
    random_bits_ >>= 1;
    --num_random_bits_;
    return bit;
  }

  uint16_t level_capacity(uint8_t num_levels, uint8_t height) const noexcept {
    return level_capacities_[num_levels - height - 1];
  }

  static uint16_t int_cap_aux(uint16_t k, uint8_t depth) {
    if (depth <= 30) return int_cap_aux_aux(k, depth);
    const uint8_t half = depth / 2;
    const uint8_t rest = depth - half;
    return int_cap_aux_aux(int_cap_aux_aux(k, half), rest);
  }

  static uint16_t int_cap_aux_aux(uint16_t k, uint8_t depth) {
    const uint64_t twok = k << 1;  // pre-multiplied by 2 for rounding
    const uint64_t tmp = (twok << depth) / detail::powers_of_three[depth];
    return static_cast<uint16_t>((tmp + 1) >> 1);
  }

  static std::array<uint16_t, kMaxNumLevels> compute_level_capacities(
      uint16_t k, uint8_t min_wid) {
    std::array<uint16_t, kMaxNumLevels> caps;
    caps.fill(min_wid);
    for (size_t depth = 0; depth < caps.size(); ++depth) {
      caps[depth] = std::max<uint16_t>(min_wid, int_cap_aux(k, depth));
      if (caps[depth] == min_wid) break;
    }
    return caps;
  }

  size_t compute_total_capacity(uint8_t num_levels) const noexcept {
    size_t total = 0;
    for (uint8_t h = 0; h < num_levels; ++h) total += level_capacity(num_levels, h);
    return total;
  }

  template <typename TT = T>
  static bool check_update_item(const TT& item) noexcept {
    if constexpr (std::is_floating_point_v<TT>) return !std::isnan(item);
    return true;
  }

  uint8_t find_level_to_compact() const noexcept {
    for (uint8_t level = 0;; ++level) {
      const uint32_t pop = levels_[level + 1] - levels_[level];
      if (pop >= level_capacity(num_levels_, level)) return level;
    }
  }

  void add_empty_top_level_to_completely_full_sketch() noexcept {
    const uint32_t cur_total_cap = levels_[num_levels_];
    const uint32_t delta_cap = level_capacity(num_levels_ + 1, 0);
    const uint32_t new_total_cap = cur_total_cap + delta_cap;
    items_ = storage_ + 2 * max_capacity_ - new_total_cap;
    for (uint8_t i = 0; i <= num_levels_; i++) levels_[i] += delta_cap;
    num_levels_++;
    levels_[num_levels_] = new_total_cap;
  }

  /// buf[i] = move(buf[j]). When the random offset is 0 the first step has
  /// i == j, and self-move-assignment leaves e.g. a long libstdc++ string
  /// empty, so non-trivial types need the check.
  static void move_item(T* buf, uint32_t i, uint32_t j) noexcept {
    if constexpr (std::is_trivially_copyable_v<T>) {
      buf[i] = buf[j];
    } else {
      if (i != j) buf[i] = std::move(buf[j]);
    }
  }

  void randomly_halve_down(T* buf, uint32_t start, uint32_t length) noexcept {
    const uint32_t half_length = length / 2;
    uint32_t j = start + random_bit();
    for (uint32_t i = start; i < start + half_length; i++, j += 2) move_item(buf, i, j);
  }

  void randomly_halve_up(T* buf, uint32_t start, uint32_t length) noexcept {
    const uint32_t half_length = length / 2;
    uint32_t j = start + length - 1 - random_bit();
    for (uint32_t i = start + length - 1; i >= start + half_length; i--, j -= 2) move_item(buf, i, j);
  }

  void sort_level_zero(uint32_t beg, uint32_t len) {
    if constexpr (kBranchless) {
      kll_detail::sort(items_ + beg, len, storage_);
    } else {
      std::sort(items_ + beg, items_ + beg + len, comparator_);
    }
  }

  /// Halves the sorted level [beg, beg + len) (len even) and merges the kept
  /// half with the sorted level above, [beg + len, beg + len + len_above),
  /// into [beg + len / 2, beg + len + len_above). Equivalent to
  /// randomly_halve_down followed by DataSketches' merge_sorted_arrays.
  void halve_and_merge_up(uint32_t beg, uint32_t len, uint32_t len_above) {
    const uint32_t half = len / 2;
    if constexpr (kBranchless) {
      // The merging networks read every other item straight from the sorted
      // level, so there is no separate halving pass. Sizes up to 32 + 32
      // cover 95% of the merges; the two-ended merge handles the rest. (A
      // 64 + 64 network is 6 KB of code and made everything 10% slower.)
      const T* a = items_ + beg + random_bit();
      const T* b = items_ + beg + len;
      T* out = items_ + beg + half;
      const uint32_t longest = std::max(half, len_above);
      if (longest <= 8) return kll_detail::merge_padded<16, 2>(a, half, b, len_above, out);
      if (longest <= 16) return kll_detail::merge_padded<32, 2>(a, half, b, len_above, out);
      if (longest <= 32) return kll_detail::merge_padded<64, 2>(a, half, b, len_above, out);
      // Output overlaps the level above: halve in place, merge from a copy.
      randomly_halve_down(items_, beg, len);
      std::copy(b, b + len_above, storage_);
      kll_detail::merge(items_ + beg, half, storage_, len_above, out);
    } else {
      randomly_halve_down(items_, beg, len);
      merge_sorted(beg, half, beg + len, len_above, beg + half);
    }
  }

  /// DataSketches kll_helper::merge_sorted_arrays: merges [beg_a, +len_a) and
  /// [beg_b, +len_b) into [beg_c, +len_a + len_b), beg_c = beg_b - len_a.
  void merge_sorted(uint32_t beg_a, uint32_t len_a, uint32_t beg_b,
                    uint32_t len_b, uint32_t beg_c) {
    uint32_t a = beg_a, b = beg_b;
    const uint32_t lim_a = beg_a + len_a, lim_b = beg_b + len_b;
    for (uint32_t c = beg_c; c < beg_c + len_a + len_b; ++c) {
      if (a == lim_a) {
        if (b != c) items_[c] = std::move(items_[b]);
        ++b;
      } else if (b == lim_b || comparator_(items_[a], items_[b])) {
        items_[c] = std::move(items_[a]);
        ++a;
      } else {
        if (b != c) items_[c] = std::move(items_[b]);
        ++b;
      }
    }
  }

  void compress_while_updating() {
    const uint8_t level = find_level_to_compact();
    // Adding the top level grows the region and shifts all indices.
    if (level == num_levels_ - 1) add_empty_top_level_to_completely_full_sketch();

    const uint32_t raw_beg = levels_[level];
    const uint32_t raw_lim = levels_[level + 1];
    const uint32_t pop_above = levels_[level + 2] - raw_lim;
    const uint32_t raw_pop = raw_lim - raw_beg;
    const bool odd_pop = raw_pop & 1;
    const uint32_t adj_beg = raw_beg + odd_pop;  // leaves the first item alone
    const uint32_t adj_pop = raw_pop - odd_pop;
    const uint32_t half_adj_pop = adj_pop / 2;
    const uint32_t destroy_beg = levels_[0];

    if (level == 0) sort_level_zero(adj_beg, adj_pop);
    if (pop_above == 0) {
      randomly_halve_up(items_, adj_beg, adj_pop);
    } else {
      halve_and_merge_up(adj_beg, adj_pop, pop_above);
    }
    levels_[level + 1] -= half_adj_pop;  // the level above grew downwards
    if (odd_pop) {
      levels_[level] = levels_[level + 1] - 1;  // the leftover item
      if (levels_[level] != raw_beg) items_[levels_[level]] = std::move(items_[raw_beg]);
    } else {
      levels_[level] = levels_[level + 1];  // the level is now empty
    }

    // Shift the levels below up so that the freed space ends up below level 0.
    if (level > 0) {
      const uint32_t amount = raw_beg - levels_[0];
      T* first = items_ + levels_[0];
      if constexpr (std::is_trivially_copyable_v<T>) {
        // Mostly a few dozen items: a backwards loop beats the memmove call.
        for (uint32_t i = amount; i-- > 0;) first[i + half_adj_pop] = first[i];
      } else {
        std::move_backward(first, first + amount, first + half_adj_pop + amount);
      }
      for (uint8_t lvl = 0; lvl < level; lvl++) levels_[lvl] += half_adj_pop;
    }
    if constexpr (!std::is_trivially_destructible_v<T>) {
      for (uint32_t i = 0; i < half_adj_pop; i++) items_[i + destroy_beg].~T();
    }
  }

  template <typename FwdT>
  void update(FwdT&& item) noexcept {
    if (!check_update_item(item)) return;
    if (levels_[0] == 0) compress_while_updating();
    new (&items_[--levels_[0]]) T(std::forward<FwdT>(item));
  }
};

}  // namespace branchless
