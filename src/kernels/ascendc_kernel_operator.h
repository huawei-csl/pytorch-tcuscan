/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
// AscendC declares the system-variable intrinsics (GetBlockNum, GetBlockIdx,
// ...) as `__aicore__ inline` but only defines them when __NPU_ARCH__ is set,
// i.e. never for host-only compilations or for clang-tidy. The diagnostic is
// reported at the declaration, so ignoring it here also covers the uses.
#pragma GCC diagnostic ignored "-Wundefined-inline"
#include "kernel_operator.h"
#pragma GCC diagnostic pop
