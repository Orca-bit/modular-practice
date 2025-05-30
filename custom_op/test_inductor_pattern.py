import torch
from torch._inductor.pattern_matcher import (
    fwd_only,
    PatternMatcherPass,
    register_replacement,
)

from pathlib import Path
from max.torch import CustomOpLibrary


mojo_kernels = Path(__file__).parent / "operations"
op_library = CustomOpLibrary(mojo_kernels)
custom_pow2_add = op_library.custom_pow2_add


def custom_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    print("custum_op")
    result = torch.zeros_like(a)
    custom_pow2_add(result, a, b)
    return result


def pattern(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a**2
    return c + b


patterns = PatternMatcherPass()
inputs = (torch.randn(10, 10), torch.randn(10, 10))
register_replacement(pattern, custom_op, inputs, fwd_only, patterns)

count = 0


def custom_pass(graph: torch.fx.graph):
    global count
    count = patterns.apply(graph)


@torch.compile(backend="inductor")
def f_mojo(x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
    return x**2 + y


def f_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x**2 + y


if __name__ == "__main__":
    import torch._inductor.config as inductor_config

    inductor_config.post_grad_custom_post_pass = custom_pass
    inp1 = torch.rand(3, 5)
    inp2 = torch.rand(3, 5)
    print(torch.allclose(f_torch(inp1, inp2), f_mojo(inp1, inp2)))
    print(f"Patterns applied: {count}")
