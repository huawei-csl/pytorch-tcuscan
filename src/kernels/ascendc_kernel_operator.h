/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
// The CANN system-variable intrinsics (`GetBlockNum`, `GetBlockIdx`, ...) are
// declared `inline` unconditionally, but only defined when `__NPU_ARCH__` is
// set.
#pragma GCC diagnostic ignored "-Wundefined-inline"
#include "kernel_operator.h"
#pragma GCC diagnostic pop
