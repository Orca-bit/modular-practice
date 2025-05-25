import torch
from pathlib import Path
from max.torch import CustomOpLibrary
from typing import Callable


class MojoKernels:
    def __init__(self):
        mojo_kernels = Path(__file__).parent / "operations"
        self.op_library = CustomOpLibrary(mojo_kernels)
        self.kernels = dict()

    def __getitem__(self, algorithm: str):
        if algorithm not in self.kernels:
            self.kernels[algorithm] = self.op_library.mojo_matmul[
                {
                    "algorithm": algorithm,
                }
            ]
        return self.kernels[algorithm]


mojo_kernels_factory = MojoKernels()


def matmul_mojo(func: Callable, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, _ = a.shape
    _, n = b.shape
    result = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    func(result, a, b)
    return result


def check_close(
    mojo_res: torch.Tensor,
    torch_res: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    # Calculate the absolute difference
    abs_diff = torch.abs(mojo_res - torch_res)

    # Find elements where the difference is greater than the tolerance
    # Formula for allclose: absolute(input - other) <= atol + rtol * absolute(other)
    not_close_mask = abs_diff > (atol + rtol * torch.abs(torch_res))

    indices = torch.nonzero(not_close_mask)

    if indices.numel() > 0:
        print(f"Found {indices.shape[0]} elements that are not close.")
        for i in range(
            min(indices.shape[0], 10)
        ):  # Print at most 10 differing elements
            idx_tuple = tuple(indices[i].tolist())
            mojo_val = mojo_res[idx_tuple]
            torch_val = torch_res[idx_tuple]
            diff_val = abs_diff[idx_tuple]
            print(
                f"  Index {idx_tuple}: Mojo={mojo_val:.6f}, Torch={torch_val:.6f}, Diff={diff_val:.6f}"
            )
    else:
        # This case might happen if allclose fails due to NaN or Inf, but not numerical difference
        # or if the custom tolerance check above is stricter/different than torch.allclose's internal one.
        print(
            "Allclose is False, but no specific differing elements found with custom tolerance check."
        )
        print(
            "This might be due to NaN/Inf values or differences in tolerance parameters."
        )
        print(f"mojo_res contains NaN: {torch.isnan(mojo_res).any()}")
        print(f"torch_res contains NaN: {torch.isnan(torch_res).any()}")
        print(f"mojo_res contains Inf: {torch.isinf(mojo_res).any()}")
        print(f"torch_res contains Inf: {torch.isinf(torch_res).any()}")


def test_matmul(algorithm: str):
    mojo_matmul_fn = mojo_kernels_factory[algorithm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    lhs = torch.randn((128, 64), dtype=dtype, device=device)
    rhs = torch.randn((64, 128), dtype=dtype, device=device)
    mojo_res = matmul_mojo(mojo_matmul_fn, lhs, rhs)
    torch_res = torch.matmul(lhs, rhs)

    all_close = torch.allclose(mojo_res, torch_res)
    print(f"\n{algorithm} matmul test results:")
    print(f"torch.allclose result: {all_close}")

    if not all_close:
        print("\nElements not close:")
        check_close(mojo_res, torch_res)


def perf_matmul(algorithm: str):
    mojo_matmul_fn = mojo_kernels_factory[algorithm]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    lhs = torch.randn((128, 64), dtype=dtype, device=device)
    rhs = torch.randn((64, 128), dtype=dtype, device=device)
    _res = matmul_mojo(mojo_matmul_fn, lhs, rhs)

    import time

    print(f"\nStarting {algorithm} matmul perf test...")
    start = time.time()
    for _ in range(1000):
        _res = matmul_mojo(mojo_matmul_fn, lhs, rhs)
    end = time.time()
    print(f"{algorithm} matmul perf results:")
    print(f"time: {end - start}s")


def test():
    test_matmul("naive")
    perf_matmul("naive")
    test_matmul("tiled")
    perf_matmul("tiled")


def perf():
    perf_matmul("naive")
    perf_matmul("tiled")


if __name__ == "__main__":
    test()
    perf()
