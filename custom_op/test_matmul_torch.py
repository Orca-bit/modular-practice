import torch
from pathlib import Path
from max.torch import CustomOpLibrary
from typing import Callable

mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)

naive_matmul = op_library.mojo_matmul[
    {
        "algorithm": "naive",
    }
]

tiled_matmul = op_library.mojo_matmul[
    {
        "algorithm": "tiled",
    }
]


def test_matmul(mojo_fn: Callable, algorithm: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    x = torch.randn((64, 64), dtype=dtype, device=device)
    y = torch.randn((64, 64), dtype=dtype, device=device)
    output = torch.zeros((64, 64), dtype=x.dtype, device=x.device)
    mojo_fn(output, x, y)
    torch_res = x @ y
    all_close = torch.allclose(output, torch_res)
    print(f"\n{algorithm} matmul test results:")
    print(f"torch.allclose result: {all_close}")


def tests():
    test_matmul(naive_matmul, "naive")
    test_matmul(tiled_matmul, "tiled")
    test_matmul(naive_matmul, "naive")


if __name__ == "__main__":
    tests()
