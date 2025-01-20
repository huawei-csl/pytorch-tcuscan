import torch
import torch_npu  # noqa
import numpy as np
import tcuscan_ops
import pytest

torch.npu.config.allow_internal_format = False

NUM_CORES = 20


def print_file(filename, array):
    sourceFile = open(filename, "w")
    print(array, file=sourceFile)
    sourceFile.close()


def get_lengths(s: int, max_iters: int):
    NUM_AI_CORES = 20
    for multiplier in range(1, max_iters):
        yield multiplier * NUM_AI_CORES * s * s


def masked_select_from_tcuscan(x, mask, s: int):
    "Implementes torch.masked_select using split operator."
    z = tcuscan_ops.run_compress(x, mask, s)
    # TODO: the torch.sum below should be done in PyTorch. WIP
    output_size = torch.sum(mask)
    return z[:output_size]


#
# def test_tcuscan_masked_select_s_16():
#    s = 16
#    vec_len = 8 * NUM_CORES * s * s
#    sparsity=0.2
#
#    x = torch.randn(vec_len).half().npu()
#    mask = (torch.rand(size=(vec_len,)) < sparsity).to(torch.int8).npu()
#    output_size = torch.sum(mask)
#
#    expected = torch.masked_select(x, mask.to(torch.uint8))
#    actual = masked_select_from_tcuscan(x, mask, s, output_size)
#
#    assert len(expected) == len(actual)
#    assert torch.allclose(expected, actual)
#


def _test_compress(vec_len: int, s: int, dtype: torch.dtype):
    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(dtype).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()

    expected = torch.masked_select(x, mask.to(torch.uint8))
    actual = masked_select_from_tcuscan(x, mask, s)

    assert len(expected) == len(actual)
    assert torch.allclose(expected, actual)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_compress_fp16_s_16(vec_len: int):
    _test_compress(vec_len, 32, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_compress_fp16_s_64(vec_len: int):
    _test_compress(vec_len, 64, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_compress_fp16_s_128(vec_len: int):
    _test_compress(vec_len, 128, torch.float16)


@pytest.mark.parametrize("vec_len", get_lengths(s=32, max_iters=12))
def test_tcuscan_compress_fp32_s_16(vec_len: int):
    _test_compress(vec_len, 32, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=64, max_iters=8))
def test_tcuscan_compress_fp32_s_64(vec_len: int):
    _test_compress(vec_len, 64, torch.float32)


@pytest.mark.parametrize("vec_len", get_lengths(s=128, max_iters=6))
def test_tcuscan_compress_fp32_s_128(vec_len: int):
    _test_compress(vec_len, 128, torch.float32)


# def test_custom():
#    s=32
#    #array_positions = np.loadtxt("./temp_mask.txt")
#    array_positions = np.loadtxt("./positions_0.txt")
#
#    mask = torch.tensor(array_positions).to(torch.int8).npu()
#    output_size = torch.sum(mask)
#    #array_scan_x = np.loadtxt("./temp_x.txt")
#    array_scan_x = np.loadtxt("./hardware_scan_0.txt")
#    x = torch.tensor(array_scan_x).half().npu()
#    #output_size = torch.sum(mask).cpu().long().numpy()
#
#    expected = torch.masked_select(x, mask.to(torch.uint8))
#    mask = mask.npu()
#    x = x.npu()
#    torch.npu.synchronize()
#    actual = masked_select_from_tcuscan(x, mask, s, output_size)
#    torch.npu.synchronize()
#
#    np.savetxt("unit_test_expected.txt", expected.cpu().numpy())
#    np.savetxt("unit_test_actual.txt", actual.cpu().numpy())
#
#    assert len(expected) == len(actual)
#    assert torch.allclose(expected, actual)
