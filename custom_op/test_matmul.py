import torch
from pathlib import Path
from max.torch import CustomOpLibrary

mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)
mojo_matmul_naive = op_library.mojo_matmul[
    {
        "algorithm": "naive",
    }
]


def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    _, n = b.shape
    result = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    mojo_matmul_naive(result, a, b)
    return result


def test_naive():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    lhs = torch.randn((128, 64), dtype=dtype, device=device)
    rhs = torch.randn((64, 128), dtype=dtype, device=device)
    mojo_res = matmul_naive(lhs, rhs)
    torch_res = torch.matmul(lhs, rhs)
    print(mojo_res)
    print(torch_res)
    print(torch.allclose(mojo_res, torch_res))


if __name__ == "__main__":
    test_naive()
