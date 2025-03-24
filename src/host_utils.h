

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

 * @file host_utils.h
 * @brief Common host utlitites.
 */
#pragma once
#include <acl/acl.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace host_utils {

/// @brief Default device id used when the `DEVICE_ID` variable is not
/// specified.
constexpr int32_t DEFAULT_DEVICE_ID = 0;

/**
 * @brief Get the device id to use.
 *
 * The value of the `DEVICE_ID` environment variable is used, if specified.
 * Otherwise, a default value is used.
 *
 * @return The id of the device.
 */
int32_t get_device_id() {
  const char *env_device_id = std::getenv("DEVICE_ID");
  if (env_device_id != nullptr) {
    return std::stoi(env_device_id);
  }
  return DEFAULT_DEVICE_ID;
}

/// @brief Maximum number of vector cores per block.
constexpr uint16_t MAX_AIV_PER_BLOCK = 2;
/// @brief Required alignment in UB, in bytes.
constexpr uint16_t UB_ALIGNMENT = 32;

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

/**
 * @brief Rounds an integral value up to the nearest multiple of a given
 * alignment.
 *
 * @tparam T Data type of the value.
 * @param [in] value Input value.
 * @param [in] alignment Alignment to use.
 * @return Aligned value.
 */
template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
constexpr T AlignUp(T value, uint32_t alignment) {
  const T tail = value % alignment;
  if (!tail) {
    return value;
  }
  const T padding = alignment - tail;
  return value + padding;
}

/**
 * @brief Calculates a median value of the values from a given vector.
 *
 * @tparam T Data type of the values.
 * @param [in] v Input vector.
 * @return Median value.
 */
template <typename T>
double GetMedian(const std::vector<T> &v) {
  assert(v.size() > 0);
  std::vector<T> v_sorted = v;
  std::sort(v_sorted.begin(), v_sorted.end());
  const size_t n = v_sorted.size();
  if (n % 2 == 1) {
    return v_sorted[n / 2];
  }
  return (v_sorted[n / 2 - 1] + v_sorted[n / 2]) / 2.;
}

/**
 * @brief A type metafunction for Cube's input / output types. For input type
 * half returns float, otherwise int32_t.
 *
 * @tparam InputT Input cube type. Must be half or int8_t.
 */
template <typename InputT>
struct CubeOutType {
  /// Output cube unit type
  using type = typename std::conditional<
      sizeof(InputT) == 2, float,
      typename std::conditional<std::is_same_v<InputT, int8_t>, int32_t,
                                uint32_t>::type>::type;
};

/**
 * @brief Syntactic sugar for `CubeOutType`
 * @tparam T Input type
 */
template <typename T>
using CubeOutType_t = typename CubeOutType<T>::type;

/// Maximum size of the L2 cache.
constexpr int32_t L2_SIZE = 192 * 1024 * 1024;
/// Global memory allocation alignment, added only for performance.
constexpr int32_t GM_ALIGNMENT = 256;

}  // namespace host_utils
