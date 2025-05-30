# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from pathlib import Path

import click
import numpy as np
from max.driver import CPU, Accelerator, Tensor, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@click.command()
@click.option(
    "--constant", "-c", type=int, default=5, help="常量值，将被添加到输入张量"
)
def main(constant):
    mojo_kernels = Path(__file__).parent / "operations"

    rows = 5
    columns = 10
    dtype = DType.float32

    # Place the graph on a GPU, if available. Fall back to CPU if not.
    device = CPU() if accelerator_count() == 0 else Accelerator()

    print(f"使用常量值: {constant}")
    print(f"运行设备: {device}")

    # Configure our simple one-operation graph.
    graph = Graph(
        "addition",
        # The custom Mojo operation is referenced by its string name, and we
        # need to provide inputs as a list as well as expected output types.
        forward=lambda x: ops.custom(
            name="add_constant_custom",
            values=[x],
            parameters={"value": constant},  # 使用命令行传入的常量
            out_types=[
                TensorType(
                    dtype=x.dtype,
                    shape=x.tensor.shape,
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor,
        input_types=[
            TensorType(
                dtype,
                shape=[rows, columns],
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    )

    # Set up an inference session for running the graph.
    session = InferenceSession(devices=[device])

    # Compile the graph.
    model = session.load(graph)

    # Fill an input matrix with random values.
    x_values = np.random.uniform(size=(rows, columns)).astype(np.float32)

    # Create a driver tensor from this, and move it to the accelerator.
    x = Tensor.from_numpy(x_values).to(device)

    # Perform the calculation on the target device.
    result = model.execute(x)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    result = result.to(CPU())

    print("Graph result:")
    print(result.to_numpy())
    print()

    print("Expected result:")
    print(x_values + constant)  # 使用命令行传入的常量

    isclose = np.all(
        np.isclose(result.to_numpy(), x_values + constant)
    )  # 使用命令行传入的常量
    print("Are the results close to each other?: %s" % (isclose))


if __name__ == "__main__":
    main()
