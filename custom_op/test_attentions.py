import torch

from pathlib import Path
from max.torch import CustomOpLibrary

mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)
mojo_fused_attention = op_library.fused_attention_custom[
    {
        "BN": 8,
        "BD": 16,
    }
]


def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    m, _ = q.shape
    _, n = v.shape
    result = torch.zeros((m, n), dtype=q.dtype, device=q.device)
    mojo_fused_attention(result, q, k, v)
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    q = torch.randn((32, 32), dtype=torch.float32, device=device)
    k = torch.randn((32, 32), dtype=torch.float32, device=device)
    v = torch.randn((32, 32), dtype=torch.float32, device=device)
    print(fused_attention(q, k, v))


if __name__ == "__main__":
    main()
