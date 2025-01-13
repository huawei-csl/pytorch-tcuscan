import torch
import torch_npu  # noqa

import tcuscan_ops

torch.npu.config.allow_internal_format = False

NUM_CORES = 20


def masked_select_from_tcuscan(x, mask, s: int, output_size: int):
    "Implementes torch.masked_select using split operator."
    # Setup TCUSCAN additional inputs
    L_s = torch.tril(torch.ones((s, s))).to(torch.int8).npu()
    z = tcuscan_ops.run_compress(x, mask, L_s)
    # TODO: the torch.sum below should be done in PyTorch. WIP
    return z[:output_size]


def test_tcuscan_masked_select_s_32():
    s = 32
    vec_len = 8 * NUM_CORES * s * s

    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(torch.float16).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()
    output_size = torch.sum(mask)

    expected = torch.masked_select(x, mask.to(torch.uint8))
    actual = masked_select_from_tcuscan(x, mask, s, output_size)

    assert len(expected) == len(actual)
    assert torch.allclose(expected, actual)


def test_tcuscan_masked_select_s_64():
    s = 64
    vec_len = 8 * NUM_CORES * s * s

    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(torch.float16).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()
    output_size = torch.sum(mask)

    expected = torch.masked_select(x, mask.to(torch.uint8))
    actual = masked_select_from_tcuscan(x, mask, s, output_size)

    assert len(expected) == len(actual)
    assert torch.allclose(expected, actual)


def test_tcuscan_masked_select_s_128():
    s = 128
    vec_len = 8 * NUM_CORES * s * s

    x = torch.randint(0, 2**7 - 1, (vec_len,)).to(torch.float16).npu()
    mask = (torch.randn(vec_len) > 0).to(torch.int8).npu()
    output_size = torch.sum(mask)

    expected = torch.masked_select(x, mask.to(torch.uint8))
    actual = masked_select_from_tcuscan(x, mask, s, output_size)

    assert len(expected) == len(actual)
    assert torch.allclose(expected, actual)
