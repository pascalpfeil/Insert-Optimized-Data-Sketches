#pragma once

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>
#include <utility>

#include "compiler.hpp"
#include "helpers.hpp"
#include "types.hpp"

#if !defined(__AVX2__) || !defined(__BMI__) || !defined(__BMI2__) || \
    !defined(__POPCNT__) || !defined(__SSE4_2__)
#error "fingerprint::SpaceSaving needs AVX2, BMI1/2, POPCNT and SSE4.2 (build with -march=native)"
#endif

/// AVX-512BW scans 32 fingerprints per compare straight into a mask register;
/// the AVX2 path needs twice the compares plus a mask extraction each.
/// -DSS_NO_AVX512 forces the AVX2 path on an AVX-512 machine.
#if defined(__AVX512BW__) && defined(__AVX512VL__) && !defined(SS_NO_AVX512)
#define SS_FINGERPRINT_AVX512 1
#else
#define SS_FINGERPRINT_AVX512 0
#endif

namespace fingerprint {

/// SpaceSaving sketch for frequent item estimation (Metwally, Agrawal, El
/// Abbadi 2005) with the semantics of final::SpaceSaving: K monitored items, a
/// miss replaces an item of minimum count and gives it count min + 1.
///
/// What differs from final::SpaceSaving:
///  * The scanned array holds 16-bit fingerprints of the keys, not the keys:
///    96 slots are 192 bytes, three AVX-512 (six AVX2) compares, for every key
///    type. A fingerprint match is verified against the full key, so the
///    result is exact; a false match (about 0.15% of lookups on random keys)
///    only costs the verification. Adversarial keys sharing a fingerprint make
///    a lookup verify up to K keys: slower, still exact.
///  * No heap. The minimum count and a list of the slots that have it are kept
///    instead. A hit increments one counter and nothing else; a miss pops a
///    slot off the list, skipping slots whose count has grown since the list
///    was built, and gives it count min + 1. When the list is empty, the
///    minimum is recomputed with a SIMD min-reduction over the counts and the
///    list with one SIMD compare. After warm-up that happens at most once per
///    K misses, since every miss removes one slot from the minimum.
///  * A miss rewrites the whole fingerprint vector of the slot, so the next
///    insert's vector load can take it from the store buffer; a 2-byte store
///    followed by a 64-byte load of the same line cannot forward and stalls
///    until the store retires.
///  * Strings are fingerprinted with CRC32C (see FingerprintBytes), other keys
///    with one multiply, not with MurmurHash3.
///
/// @tparam T the data type the sketch summarizes.
/// @tparam K number of elements the sketch can store, a multiple of 32.
template <typename T, size_t K = 96>
class SpaceSaving {
  static_assert(K % 32 == 0 && K <= 128, "K must be a multiple of 32, at most 128");
  using Fp = uint16_t;
  /// Fingerprints per vector of the scan.
  static constexpr size_t kLanes = SS_FINGERPRINT_AVX512 ? 32 : 16;
  /// Items up to 16 bits are their own fingerprint: no verification needed.
  static constexpr bool kExact = std::is_integral_v<T> && sizeof(T) <= sizeof(Fp);
  static constexpr bool kIsString = detail::is_string_v<T> || std::is_same_v<T, std::string_view>;
  /// Floats are normalized by value (+-0.0), everything else passed by reference.
  using NormalizedT = std::conditional_t<std::is_floating_point_v<T>, T, const T&>;

 public:
  SpaceSaving() {
    if constexpr (std::is_arithmetic_v<T>) values_ = detail::sequence<T, K>();
    for (size_t i = 0; i < K; ++i) fps_[i] = Fingerprint(values_[i]);
    weights_.fill(0);
    RebuildMinList();
  }

  /// Update the weight of a given value.
  void Insert(const T& v) noexcept {
    const NormalizedT key = Normalized(v);
    const Fp fp = Fingerprint(key);
    uint64_t lo, hi;
    Scan(fp, lo, hi);
    // Hit or miss stays a branch. A branch-free version (same stores on both
    // paths, selects instead of jumps) was 2x slower on uniform and 1.5x on
    // Zipf data: it chains every insert through scan -> key verification ->
    // slot select -> fingerprint store -> next scan, while with the branch
    // the miss path runs ahead speculatively.
    if (UNLIKELY((lo | hi) != 0)) {
      const size_t i = FirstBit(lo, hi);
      if (kExact || LIKELY(values_[i] == key)) {
        ++weights_[i];
        return;
      }
      if (ResolveCollision(key, lo, hi)) return;
    }
    StoreFingerprint(Miss(key), fp);
  }

  /// Estimated count of a value: its counter if monitored, else the minimum
  /// counter (an upper bound, as in SpaceSaving). The minimum is recomputed
  /// because min_weight_ may lag behind it until the next rebuild.
  uint64_t Estimate(const T& v) const noexcept {
    const NormalizedT key = Normalized(v);
    for (size_t i = 0; i < K; ++i) {
      if (values_[i] == key) return weights_[i];
    }
    return *std::min_element(weights_.begin(), weights_.end());
  }

  /// Monitored items with their counts.
  std::array<std::pair<T, uint64_t>, K> Items() const {
    std::array<std::pair<T, uint64_t>, K> out;
    for (size_t i = 0; i < K; ++i) out[i] = {values_[i], weights_[i]};
    return out;
  }

 private:
  /// Equal values must have equal bits: +0.0 and -0.0 both become +0.0.
  OPT_INLINE static NormalizedT Normalized(const T& value) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
      using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
      Bits bits;
      std::memcpy(&bits, &value, sizeof bits);
      const Bits magnitude = bits & (~Bits{0} >> 1);
      bits &= Bits{0} - Bits(magnitude != 0);  // all ones unless +-0
      T out;
      std::memcpy(&out, &bits, sizeof out);
      return out;
    } else {
      return value;
    }
  }

  OPT_INLINE static uint64_t Load64(const char* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
  }

  /// 64x64 -> 128-bit multiply folded to 64 bits.
  OPT_INLINE static uint64_t Mum(uint64_t a, uint64_t b) noexcept {
    const __uint128_t r = static_cast<__uint128_t>(a) * b;
    return static_cast<uint64_t>(r) ^ static_cast<uint64_t>(r >> 64);
  }

  OPT_INLINE static Fp Fingerprint(const T& key) noexcept {
    if constexpr (kExact) {
      return static_cast<Fp>(static_cast<std::make_unsigned_t<T>>(key));
    } else if constexpr (kIsString) {
      return FingerprintBytes(key.data(), key.size());
    } else if constexpr (std::is_same_v<T, __int128_t> || std::is_same_v<T, __uint128_t>) {
      const auto u = static_cast<__uint128_t>(key);
      return static_cast<Fp>(Mum(static_cast<uint64_t>(u) ^ 0xe7037ed1a0b428dbULL,
                                 static_cast<uint64_t>(u >> 64) ^ 0x8ebc6af09c88c6e3ULL) >> 48);
    } else {
      // Multiply-shift: the top 16 bits of key * odd constant. One multiply,
      // and unlike the key's own top bits it spreads sequential keys.
      static_assert(sizeof(T) <= 8);
      using Bits = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
      Bits bits;
      std::memcpy(&bits, &key, sizeof bits);
      return static_cast<Fp>((static_cast<uint64_t>(bits) * 0x9E3779B97F4A7C15ULL) >> 48);
    }
  }

  /// Fingerprint of a byte string with CRC32C: 8 bytes per cycle, three
  /// independent streams to cover the 3-cycle latency. Every match is verified
  /// against the key, so the fingerprint only has to spread real data, not
  /// resist an adversary. MurmurHash3 costs 3 to 5 times more and was most of
  /// the insert time for the 320-character benchmark strings.
  OPT_INLINE static Fp FingerprintBytes(const char* p, size_t n) noexcept {
    uint64_t a = n, b = 0x9E3779B97F4A7C15ULL, c = 0xBF58476D1CE4E5B9ULL;
    const char* const end = p + n;
    for (; p + 24 <= end; p += 24) {
      a = _mm_crc32_u64(a, Load64(p));
      b = _mm_crc32_u64(b, Load64(p + 8));
      c = _mm_crc32_u64(c, Load64(p + 16));
    }
    for (; p + 8 <= end; p += 8) a = _mm_crc32_u64(a, Load64(p));
    for (; p < end; ++p) a = _mm_crc32_u8(static_cast<uint32_t>(a), static_cast<uint8_t>(*p));
    // The three CRCs are 32-bit; one multiply mixes them into the top bits.
    return static_cast<Fp>(((a | (b << 32)) * 0x9E3779B97F4A7C15ULL ^ c * 0xD6E8FEB86659FD93ULL) >> 48);
  }

  /// Index of the lowest set bit of the 128-bit mask (hi:lo), which must not
  /// be zero. tzcnt of zero is 64, so there is no branch on which half holds
  /// the bit (a coin flip on skewed data).
  OPT_INLINE static size_t FirstBit(uint64_t lo, uint64_t hi) noexcept {
    const uint64_t in_hi = uint64_t{0} - uint64_t(lo == 0);  // all ones if lo == 0
    return _tzcnt_u64(lo) + (_tzcnt_u64(hi) & in_hi);
  }

  /// Bit i of (hi:lo) is set iff fps_[i] == fp.
  OPT_INLINE void Scan(Fp fp, uint64_t& lo, uint64_t& hi) const noexcept {
    uint64_t m[K / kLanes];
#if SS_FINGERPRINT_AVX512
    const __m512i v = _mm512_set1_epi16(static_cast<short>(fp));
    for (size_t j = 0; j < K / kLanes; ++j) {
      m[j] = _mm512_cmpeq_epi16_mask(_mm512_load_si512(fps_.data() + j * kLanes), v);
    }
#else
    const __m256i v = _mm256_set1_epi16(static_cast<short>(fp));
    for (size_t j = 0; j < K / kLanes; ++j) {
      const __m256i eq = _mm256_cmpeq_epi16(
          _mm256_load_si256(reinterpret_cast<const __m256i*>(fps_.data() + j * kLanes)), v);
      // Two mask bits per 16-bit lane; keep one of each pair.
      m[j] = _pext_u32(static_cast<uint32_t>(_mm256_movemask_epi8(eq)), 0xAAAAAAAAu);
    }
#endif
    lo = hi = 0;
    for (size_t j = 0; j < K / kLanes; ++j) {
      const size_t bit = j * kLanes;
      if (bit < 64) lo |= m[j] << bit;
      else hi |= m[j] << (bit - 64);
    }
  }

  /// The first fingerprint match was a false positive (about 0.15% of
  /// lookups): check the other matches. Returns whether key was found and
  /// counted. Out of line so the common path stays small.
  __attribute__((noinline)) bool ResolveCollision(const T& key, uint64_t lo, uint64_t hi) noexcept {
    for (;;) {
      if (lo) {
        lo &= lo - 1;
      } else {
        hi &= hi - 1;
      }
      if ((lo | hi) == 0) return false;
      const size_t i = FirstBit(lo, hi);
      if (values_[i] == key) {
        ++weights_[i];
        return true;
      }
    }
  }

  /// Replaces a minimum slot with key and returns the slot. Pops the list
  /// until a slot that still has the minimum count comes up: slots that were
  /// hit since the list was built are stale and skipped (rare). An empty list
  /// means the minimum has grown, so it is recomputed.
  OPT_INLINE size_t Miss(const T& key) noexcept {
    size_t n = min_count_;
    size_t i;
    for (;;) {
      if (UNLIKELY(n == 0)) {
        RebuildMinList();
        n = min_count_;
      }
      i = min_list_[--n];
      if (LIKELY(weights_[i] == min_weight_)) break;
    }
    min_count_ = static_cast<uint32_t>(n);
    values_[i] = key;
    weights_[i] = min_weight_ + 1;
    return i;
  }

  /// fps_[i] = fp, written as the whole vector holding slot i (see the class
  /// comment on store forwarding). The asm barrier keeps GCC from folding
  /// load + blend + store into a single-lane masked store, which would not
  /// forward either.
  OPT_INLINE void StoreFingerprint(size_t i, Fp fp) noexcept {
    Fp* vec = fps_.data() + (i / kLanes) * kLanes;
#if SS_FINGERPRINT_AVX512
    const __mmask32 lane = __mmask32{1} << (i % kLanes);
    __m512i v = _mm512_mask_set1_epi16(_mm512_load_si512(vec), lane, static_cast<short>(fp));
    asm("" : "+v"(v));
    _mm512_store_si512(vec, v);
#else
    const __m256i lanes = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i lane = _mm256_cmpeq_epi16(lanes, _mm256_set1_epi16(static_cast<short>(i % kLanes)));
    const __m256i old = _mm256_load_si256(reinterpret_cast<const __m256i*>(vec));
    __m256i v = _mm256_blendv_epi8(old, _mm256_set1_epi16(static_cast<short>(fp)), lane);
    asm("" : "+x"(v));
    _mm256_store_si256(reinterpret_cast<__m256i*>(vec), v);
#endif
  }

  /// Recomputes the minimum count and lists the slots that have it. Runs once
  /// per K misses after warm-up, so it is kept simple rather than fast.
  void RebuildMinList() noexcept {
    uint64_t min = weights_[0];
    uint32_t n = 0;
#if SS_FINGERPRINT_AVX512
    __m512i m = _mm512_load_si512(weights_.data());
    for (size_t j = 1; j < K / 8; ++j) m = _mm512_min_epu64(m, _mm512_load_si512(weights_.data() + j * 8));
    min = _mm512_reduce_min_epu64(m);
    const __m512i v = _mm512_set1_epi64(static_cast<long long>(min));
    const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (size_t j = 0; j < K / 8; ++j) {
      const __mmask8 eq = _mm512_cmpeq_epi64_mask(_mm512_load_si512(weights_.data() + j * 8), v);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(min_list_.data() + n),
                          _mm256_maskz_compress_epi32(eq, _mm256_add_epi32(iota, _mm256_set1_epi32(8 * j))));
      n += static_cast<uint32_t>(_mm_popcnt_u32(eq));
    }
#else
    for (size_t i = 1; i < K; ++i) min = std::min(min, weights_[i]);
    for (uint32_t i = 0; i < K; ++i) {
      min_list_[n] = i;
      n += weights_[i] == min;
    }
#endif
    min_weight_ = min;
    min_count_ = n;
  }

  /// Fingerprints, the only array the lookup scans.
  alignas(64) std::array<Fp, K> fps_;
  /// Counts.
  alignas(64) std::array<uint64_t, K> weights_;
  /// Full keys, read to verify a fingerprint match and written on a miss.
  std::array<T, K> values_{};
  /// Minimum count and the slots that had it when the list was built; the
  /// first min_count_ entries are live. 8 spare entries: the compress store
  /// writes a full vector.
  uint64_t min_weight_ = 0;
  uint32_t min_count_ = 0;
  std::array<uint32_t, K + 8> min_list_;
};

}  // namespace fingerprint
