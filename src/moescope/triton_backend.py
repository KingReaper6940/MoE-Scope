"""Masked static persistent grouped GEMM for CUDA float16 matrices.

Scheduling follows the design in Triton's official Group GEMM tutorial:
https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
This implementation adds arbitrary-dimension masks and explicit prepared inputs.
It is a GEMM microbenchmark backend, not a fused MoE layer.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped(a_ptrs, b_ptrs, c_ptrs, sizes, GROUPS: tl.constexpr,
             WORKERS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    base = 0
    for group in range(GROUPS):
        m = tl.load(sizes + group * 3)
        n = tl.load(sizes + group * 3 + 1)
        k = tl.load(sizes + group * 3 + 2)
        columns = tl.cdiv(n, BN)
        count = tl.cdiv(m, BM) * columns
        while (tile >= base) & (tile < base + count):
            local = tile - base
            rows = (local // columns) * BM + tl.arange(0, BM)
            cols = (local % columns) * BN + tl.arange(0, BN)
            ks = tl.arange(0, BK)
            ap = tl.load(a_ptrs + group).to(tl.pointer_type(tl.float16))
            bp = tl.load(b_ptrs + group).to(tl.pointer_type(tl.float16))
            cp = tl.load(c_ptrs + group).to(tl.pointer_type(tl.float16))
            acc = tl.zeros((BM, BN), tl.float32)
            for step in range(tl.cdiv(k, BK)):
                kk = step * BK + ks
                a = tl.load(ap + rows[:, None] * k + kk[None, :],
                            (rows[:, None] < m) & (kk[None, :] < k), other=0.)
                b = tl.load(bp + kk[:, None] * n + cols[None, :],
                            (kk[:, None] < k) & (cols[None, :] < n), other=0.)
                acc += tl.dot(a, b)
            tl.store(cp + rows[:, None] * n + cols[None, :], acc.to(tl.float16),
                     (rows[:, None] < m) & (cols[None, :] < n))
            tile += WORKERS
        base += count


class PreparedGroup:
    """Own input/output tensors and pointer tables outside the timed region."""
    def __init__(self, matrices_a, matrices_b):
        if not matrices_a or len(matrices_a) != len(matrices_b):
            raise ValueError("provide matching, nonempty matrix groups")
        device = matrices_a[0].device
        self.a, self.b = list(matrices_a), list(matrices_b)
        self.c = []
        sizes = []
        for a, b in zip(self.a, self.b):
            if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]
                    or a.dtype != torch.float16 or b.dtype != torch.float16
                    or not a.is_cuda or not b.is_cuda or a.device != device or b.device != device
                    or not a.is_contiguous() or not b.is_contiguous()
                    or a.shape[1] == 0 or b.shape[1] == 0):
                raise ValueError("expected contiguous float16 CUDA matrices with compatible nonzero K,N")
            self.c.append(torch.empty((a.shape[0], b.shape[1]), device=device, dtype=a.dtype))
            sizes.extend([a.shape[0], b.shape[1], a.shape[1]])
        self.ap, self.bp, self.cp = [torch.tensor([t.data_ptr() for t in group],
                                                 dtype=torch.int64, device=device)
                                    for group in (self.a, self.b, self.c)]
        self.sizes = torch.tensor(sizes, dtype=torch.int32, device=device)
        self.workers = torch.cuda.get_device_properties(device).multi_processor_count

    def triton(self, *, block_m=32, block_n=64, block_k=32):
        for value in (block_m, block_n, block_k):
            if type(value) is not int or value < 16 or value & (value - 1):
                raise ValueError("Triton block sizes must be powers of two >=16")
        with torch.cuda.device(self.a[0].device):
            _grouped[(self.workers,)](self.ap, self.bp, self.cp, self.sizes,
                                      len(self.a), self.workers, block_m, block_n, block_k,
                                      num_warps=4)
        return self.c

    def torch(self):
        for a, b, c in zip(self.a, self.b, self.c):
            torch.mm(a, b, out=c)
        return self.c
