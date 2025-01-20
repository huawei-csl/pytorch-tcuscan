import torch
import torch_npu  # noqa
import numpy as np
import numpy.typing as npt

import tcuscan_ops
import sys
import numpy

torch.npu.config.allow_internal_format = False

NUM_AI_CORES = 20

count = 0


def _segsum_expiremental(x, f):
    scan_x = np.cumsum(x)
    n = len(x)
    assert len(f) == len(x), "Input data and flags vector lengths must be equal."
    positions = np.append(f[1:], 1).astype(bool)
    compact_vals = np.compress(positions, scan_x)
    diffs = np.diff(compact_vals, prepend=[0])
    return diffs


def print_file(filename, array):
    np.savetxt(filename, array)


def tcuscan_compress(x, mask, s: int, output_size: int):
    "Implementes torch.masked_select using split operator."
    z = tcuscan_ops.run_compress(x, mask, s)
    return z[:output_size]


def segmented_sum(x, f, s):
    global count

    scan_x = tcuscan_ops.run_scan_multi_core(x, s)
    torch.npu.synchronize()

    mask = torch.cat([f[1:], torch.ones(1)]).to(torch.int8).npu()
    output_size = torch.sum(mask)
    compact_vals = tcuscan_compress(scan_x, mask, s, output_size)
    torch.npu.synchronize()

    result = tcuscan_ops.run_diff(compact_vals)
    torch.npu.synchronize()

    return result


def _test_fp32(s: int, max_iters: int):
    numpy.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(threshold=sys.maxsize)
    num_iterations = 1
    global count

    # torch.manual_seed(0)
    for _ in range(0, num_iterations):
        for multiplier in range(1, max_iters):
            vec_len = multiplier * NUM_AI_CORES * s * s

            for segments_density in {
                0.2,
                0.4,
            }:  # 0.6 - TODO: 0.6 must be further tested

                data = torch.randn(vec_len).half()
                f = (torch.rand(size=(vec_len,)) < segments_density).to(torch.int8)
                golden = _segsum_expiremental(data.numpy(), f.numpy())
                tensor_golden = torch.from_numpy(golden)

                x = data.npu()
                torch.npu.synchronize()
                first = segmented_sum(x, f, s)
                torch.npu.synchronize()
                second = segmented_sum(x, f, s)
                torch.npu.synchronize()

                count = count + 1

                print(
                    f"{count}: size of the golden tensor: {len(tensor_golden)} while the size of the actual tensor is {len(second.cpu())}"
                )
                diff_tensor = tensor_golden - second.cpu()
                norm_avg = torch.norm(diff_tensor) / vec_len

                print(f"{count}: norm value is {norm_avg}")
                print(f"maximum norm is {torch.norm(diff_tensor, torch.inf) }")
                assert (
                    norm_avg < 1e-02
                ), f"segmented sum (fp16) wrong. s={s}, vec_len={vec_len},  multiplier={multiplier}, at the execution {count}"


def test_segmented_sum_fp32_s_32():
    s = 32
    max_iters = 2
    _test_fp32(s, max_iters)


def test_segmented_sum_fp32_s_64():
    s = 64
    max_iters = 2
    _test_fp32(s, max_iters)


def test_segmented_sum_fp32_s_128():
    s = 128
    max_iters = 2
    _test_fp32(s, max_iters)
