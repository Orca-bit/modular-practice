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

import compiler
from gpu import block_idx, thread_idx, barrier
from gpu.memory import async_copy_wait_all
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from math import ceildiv
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import OutputTensor, InputTensor


fn naive_matmul_gpu[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
](
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    col = thread_idx.x
    row = thread_idx.y
    bidx = block_idx.x
    bidy = block_idx.y

    dst = c.tile[BM, BN](bidy, bidx)
    var dst_reg: c.element_type = 0

    @parameter
    for k in range(b.shape[0]()):
        a_tile = a.tile[BM, 1](bidy, k)
        b_tile = b.tile[1, BN](k, bidx)
        dst_reg += a_tile[row, 0] * b_tile[0, col]

    dst[row, col] = dst_reg


fn tiled_matmul[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    BM: Int,
    BN: Int,
    BK: Int,
    NUM_THREADS: Int,
](
    a: LayoutTensor[dtype, a_layout, MutableAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutableAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutableAnyOrigin],
):
    col = thread_idx.x % BN
    row = thread_idx.x // BN
    bidx = block_idx.x
    bidy = block_idx.y

    dst = c.tile[BM, BN](bidy, bidx)
    a_smem = tb[dtype=dtype]().row_major[BM, BK]().shared().alloc()
    b_smem = tb[dtype=dtype]().row_major[BK, BN]().shared().alloc()

    var dst_reg: c.element_type = 0

    @parameter
    for k_block in range(b.shape[0]() // BK):
        a_tile = a.tile[BM, BK](bidy, k_block)
        b_tile = b.tile[BK, BN](k_block, bidx)

        alias load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        alias load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)
        async_copy_wait_all()

        barrier()

        @parameter
        for k in range(BK):
            dst_reg += a_smem[row, k] * b_smem[k, col]
        barrier()

    dst[row, col] = dst_reg


@compiler.register("mojo_matmul")
struct MatMul[algorithm: StaticString]:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int, //, target: StaticString
    ](
        output: OutputTensor[type=dtype, rank=rank],
        lhs: InputTensor[type=dtype, rank=rank],
        rhs: InputTensor[type=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "gpu":
            a = lhs.to_layout_tensor()
            b = rhs.to_layout_tensor()
            c = output.to_layout_tensor()

            M, N = a.shape[0](), b.shape[1]()
            device_ctx = ctx.get_device_context()

            @parameter
            if algorithm == "naive":
                alias BM = 32
                alias BN = 32
                device_ctx.enqueue_function[
                    naive_matmul_gpu[
                        dtype, a.layout, b.layout, c.layout, BM, BN
                    ]
                ](
                    a,
                    b,
                    c,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BN, BM),
                )
            elif algorithm == "tiled":
                alias BM = 32
                alias BN = 32
                alias BK = 32
                alias NUM_THREADS = BM * BN
                device_ctx.enqueue_function[
                    tiled_matmul[
                        dtype,
                        a.layout,
                        b.layout,
                        c.layout,
                        BM,
                        BN,
                        BK,
                        NUM_THREADS,
                    ]
                ](
                    a,
                    b,
                    c,
                    grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                    block_dim=(BM * BN),
                )
