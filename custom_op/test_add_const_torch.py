import torch
from pathlib import Path
from max.torch import CustomOpLibrary
from typing import Callable

mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)

add_one = op_library.add_constant_custom[
    {
        "value": 1,
    }
]

add_two = op_library.add_constant_custom[
    {
        "value": 2,
    }
]


def test_add_const(mojo_fn: Callable, value: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    x = torch.randn((2, 2), dtype=dtype, device=device)
    output = torch.zeros((2, 2), dtype=x.dtype, device=x.device)
    mojo_fn(output, x)
    torch_res = x + value
    all_close = torch.allclose(output, torch_res)
    print(f"\nadd_constant {value} test results:")
    print(f"torch.allclose result: {all_close}")


def tests():
    test_add_const(add_one, 1)
    test_add_const(add_two, 2)
    test_add_const(add_one, 1)
