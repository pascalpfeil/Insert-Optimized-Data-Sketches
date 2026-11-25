#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "hash.hpp"
#include "types.hpp"

template <typename T>
std::string to_string(const T& value) {
  std::stringstream ss;
  ss << std::fixed;
  int width = 0;

  if constexpr (std::is_same_v<T, float>) {
    ss << std::setprecision(std::numeric_limits<float>::max_digits10);
    width = 40;  // Enough to represent any float value
  } else if constexpr (std::is_same_v<T, double>) {
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);
    width = 320;  // Enough to represent any double value
  } else if constexpr (std::is_same_v<T, int16_t>) {
    width = 6;  // -32768 to 32767
  } else if constexpr (std::is_same_v<T, int32_t>) {
    width = 11;  // -2147483648 to 2147483647
  } else if constexpr (std::is_same_v<T, int64_t>) {
    width = 20;  // -9223372036854775808 to 9223372036854775807
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    width = 5;  // 0 to 65535
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    width = 10;  // 0 to 4294967295
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    width = 20;  // 0 to 18446744073709551615
  } else {
    static_assert(!sizeof(T), "Unsupported type");
  }

  // Handle negative numbers for signed types
  if constexpr (std::is_signed_v<T>) {
    if (value < 0) {
      ss << '-' << std::setfill('0') << std::setw(width - 1) << -value;
    } else {
      ss << std::setfill('0') << std::setw(width) << value;
    }
  } else {
    ss << std::setfill('0') << std::setw(width) << value;
  }

  return std::move(ss).str();
}

template <typename T>
const std::vector<T>& GetData() {
  static std::vector<T> data;
  if (data.empty()) {
    using TT = std::conditional_t<
        detail::is_string_v<T>, double,
        std::conditional_t<
            std::is_same_v<T, __int128_t>, int64_t,
            std::conditional_t<std::is_same_v<T, __uint128_t>, uint64_t, T>>>;
    using distribution = std::conditional_t<std::is_integral_v<TT>,
                                            std::uniform_int_distribution<TT>,
                                            std::uniform_real_distribution<TT>>;
    constexpr size_t kSeed = 42;
    std::mt19937 gen(kSeed);
    // pcg32_fast gen(kSeed);
    distribution dist(std::numeric_limits<TT>::min(),
                      std::numeric_limits<TT>::max());
    constexpr size_t kNumValues = 1'000'000;
    data.reserve(kNumValues);
    for (size_t i = 0; i < kNumValues; ++i) {
      if constexpr (detail::is_string_v<T>) {
        data.emplace_back(to_string(dist(gen)));
      } else if constexpr (std::is_same_v<T, __int128_t> ||
                           std::is_same_v<T, __uint128_t>) {
        data.push_back((static_cast<T>(dist(gen)) << 64) |
                       static_cast<T>(dist(gen)));
      } else {
        data.emplace_back(dist(gen));
      }
    }
  }
  return data;
}

/// Bijection on [0, 2^bits): odd multiplications and xor-shifts are each
/// invertible, so distinct inputs below 2^bits give distinct outputs.
inline uint64_t MixBits(uint64_t x, unsigned bits) {
  const uint64_t mask = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
  x = (x * 0x9E3779B97F4A7C15ULL) & mask;
  x ^= x >> (bits / 2);
  x = (x * 0xBF58476D1CE4E5B9ULL) & mask;
  x ^= x >> (bits / 2 + 1);
  return x;
}

/// Value of type T for Zipf rank `rank` (0 = most frequent). Ranks are spread
/// over the value space by a bijection, so the frequent values are not simply
/// the small integers. Floating point values are integers below 2^24 / 2^53,
/// which both types represent exactly.
template <typename T>
T ZipfValue(uint64_t rank) {
  if constexpr (std::is_same_v<T, int16_t>) {
    return static_cast<T>(static_cast<uint16_t>(MixBits(rank, 16)));
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return static_cast<T>(static_cast<uint32_t>(MixBits(rank, 32)));
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<T>(MixBits(rank, 64));
  } else if constexpr (std::is_same_v<T, __int128_t>) {
    return static_cast<T>((static_cast<__uint128_t>(MixBits(rank, 64)) << 64) |
                          MixBits(rank ^ 0x5555555555555555ULL, 64));
  } else if constexpr (std::is_same_v<T, float>) {
    return static_cast<float>(MixBits(rank, 24));
  } else if constexpr (std::is_same_v<T, double>) {
    return static_cast<double>(MixBits(rank, 53));
  } else if constexpr (detail::is_string_v<T>) {
    return to_string(static_cast<double>(MixBits(rank, 53)));
  } else {
    static_assert(!sizeof(T), "Unsupported type");
  }
}

/// Zipf(kZipfExponent) distributed data: 1M values drawn over 1M distinct
/// values (65536 for int16_t), rank r having probability proportional to
/// 1 / (r + 1)^s. With s = 1 the most frequent value makes up about 7% of the
/// data and about 217K distinct values appear (59K for int16_t). Sampling
/// inverts the exact CDF.
constexpr double kZipfExponent = 1.0;

template <typename T>
const std::vector<T>& GetZipfData() {
  static std::vector<T> data;
  if (data.empty()) {
    constexpr size_t kNumValues = 1'000'000;
    const size_t distinct = std::is_same_v<T, int16_t> ? 65536 : kNumValues;
    std::vector<double> cdf(distinct);
    double acc = 0;
    for (size_t i = 0; i < distinct; ++i) {
      cdf[i] = (acc += std::pow(static_cast<double>(i + 1), -kZipfExponent));
    }
    for (double& c : cdf) c /= acc;
    constexpr size_t kSeed = 42;
    std::mt19937 gen(kSeed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    data.reserve(kNumValues);
    for (size_t i = 0; i < kNumValues; ++i) {
      const auto it = std::upper_bound(cdf.begin(), cdf.end(), dist(gen));
      const size_t rank =
          std::min<size_t>(static_cast<size_t>(it - cdf.begin()), distinct - 1);
      data.push_back(ZipfValue<T>(rank));
    }
  }
  return data;
}

template <typename T>
const std::vector<__uint128_t>& GetHashes() {
  static std::vector<__uint128_t> hashes;
  if (hashes.empty()) {
    const auto& data = GetData<T>();
    hashes.reserve(data.size());
    for (const auto& value : data) {
      hashes.push_back(detail::Hash(value));
    }
  }
  return hashes;
}
