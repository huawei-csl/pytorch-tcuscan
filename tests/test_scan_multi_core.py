import torch
import torch_npu  # noqa

import tcuscan_ops

torch.npu.config.allow_internal_format = False

NUM_AI_CORES = 20


def _test_fp16(s: int, max_iters: int):

    U_s = torch.tril(torch.ones(s, s)).half().npu()

    for multiplier in range(1, max_iters):
        vec_len = multiplier * NUM_AI_CORES * s * s
        x = torch.randn(vec_len).half().npu()

        expected = torch.cumsum(x, dim=-1, dtype=torch.float32)
        actual = tcuscan_ops.run_scan_multi_core(x, U_s)
        assert torch.allclose(
            actual, expected, atol=1e-02
        ), f"multi-core scan (fp16) wrong. s={s}, vec_len={vec_len},  multiplier={multiplier}"
        torch.synchronize()


def test_mcscan_fp16_s_16():
    s = 16
    max_iters = 2
    _test_fp16(s, max_iters)


def test_mcscan_fp16_s_64():
    s = 64
    max_iters = 1
    _test_fp16(s, max_iters)


def test_mcscan_fp16_s_128():
    s = 128
    max_iters = 1
    _test_fp16(s, max_iters)
