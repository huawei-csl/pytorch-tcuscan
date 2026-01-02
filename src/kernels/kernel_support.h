/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_support.h
 * @brief Kernel implementing a support debug operation.
 */
#pragma once

namespace tcuscan {

/**
 * @brief Prints an amout of data N from the GM memory
 *
 * It takes as input the size N and the GM Address to read from.
 * It fetches memory and puts values in LocalTesnor to be printed
 * @param size : amount of data to be read from memory
 * @tparam T Data type.
 */
template <typename T>
class KernelSupport {
 public:
  /**
   * @brief Class constructor.
   * @param [in] size Number of elements to be read.
   */
  __aicore__ inline KernelSupport(uint32_t size) : size_(size) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input GM address to read from
   *
   * */
  __aicore__ inline void Init(GM_ADDR input) {
    global_input_.SetGlobalBuffer((__gm__ T *)input, size_);

    pipe.InitBuffer(vecin_q_, 1, size_ * sizeof(T));
  }

  /**
   * @brief Fetches memory and prints the values
   *
   * */
  __aicore__ inline void Process() {
    copy::CopyGmToVec(vecin_q_, global_input_);
    LocalTensor<T> vec_lt = vecin_q_.DeQue<T>();
    for (int i = 0; i < size_; i++) {
      printf("[SUPPORT KERNEL] value at idx %d is %f\n", i, vec_lt.GetValue(i));
    }
  }

 private:
  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  GlobalTensor<T> global_input_;
  const uint32_t size_;
};

}  // namespace tcuscan
