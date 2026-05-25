import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp, min_out, max_out, M, BLOCK_SIZE: tl.constexpr, UPCAST: tl.constexpr
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    
    dtype = inp.type.element_ty
    acc_type = tl.float32 if UPCAST else dtype

    first_val = tl.load(inp).to(acc_type)
    
    min_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(acc_type)
    max_val = tl.load(inp_ptrs, mask=mask, other=0.0).to(acc_type)
    
    min_val = tl.where(mask, min_val, first_val)
    max_val = tl.where(mask, max_val, first_val)

    min_val = tl.min(min_val).to(dtype)
    max_val = tl.max(max_val).to(dtype)

    min_ptr = min_out + pid
    max_ptr = max_out + pid
    tl.store(min_ptr, min_val)
    tl.store(max_ptr, max_val)


@libentry()
@triton.jit
def aminmax_kernel_2(
    min_inp, max_inp, min_out, max_out, mid_size, BLOCK_MID: tl.constexpr, UPCAST: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset
    mask = offset < mid_size
    
    dtype = min_inp.type.element_ty
    acc_type = tl.float32 if UPCAST else dtype

    first_min = tl.load(min_inp).to(acc_type)
    first_max = tl.load(max_inp).to(acc_type)
    
    min_val = tl.load(min_ptrs, mask=mask, other=0.0).to(acc_type)
    max_val = tl.load(max_ptrs, mask=mask, other=0.0).to(acc_type)
    
    min_val = tl.where(mask, min_val, first_min)
    max_val = tl.where(mask, max_val, first_max)

    min_val = tl.min(min_val).to(dtype)
    max_val = tl.max(max_val).to(dtype)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)

@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def aminmax_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UPCAST: tl.constexpr,
):
    dtype = inp.type.element_ty

    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    
    inp_ptrs = inp + rows * N
    row_mask = rows < M

    acc_type = tl.float32 if UPCAST else dtype
    
    # 提取每一行的合法首元素作为初始占位符
    first_a = tl.load(inp_ptrs, mask=row_mask, other=0.0).to(acc_type)
    
    # [核心修复 1] 显式加法实例化物理寄存器
    # 强制打破 Triton 的 tl.broadcast_to 带来的惰性编译异常，
    # 确保 tl.max 和 tl.min 可以正常寻址二维张量块，杜绝 float32 max 的溢出返回。
    zeros = tl.zeros([BLOCK_M, BLOCK_N], dtype=acc_type)
    _min = first_a + zeros
    _max = first_a + zeros

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp_ptrs + cols, mask=mask, other=0.0).to(acc_type)
        #a_safe = tl.where(mask, a, first_a) 
        # 掩码外的元素一律保留 first_a，保证归约的绝对正确
        _min = tl.where(mask, tl.minimum(_min, a), _min)
        _max = tl.where(mask, tl.maximum(_max, a), _max)
        #print("_min",_min)
        if pid==0:
            #tl.device_print("_max=", _max)
            #print("_max",_max)
            pass
        #_min=tl.minimum(_min,a_safe)
        #_max=tl.maximum(_max,a_safe)
        #tl.debug_barrier()
    min_result = tl.min(_min, axis=1)[:, None].to(dtype)
    max_result = tl.max(_max, axis=1)[:, None].to(dtype)
    #print("min_result",min_result)
    #print("max_result",max_result)
    tl.store(min_out + rows, min_result, row_mask)
    tl.store(max_out + rows, max_result, row_mask)

def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS AMINMAX")

    is_upcast = (inp.dtype == torch.bfloat16)

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            aminmax_kernel_1[(mid_size, 1)](
                inp, min_mid, max_mid, M, block_size, UPCAST=is_upcast
            )
            aminmax_kernel_2[(1, 1)](
                min_mid, max_mid, min_out, max_out, mid_size, block_mid, UPCAST=is_upcast
            )
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        inp = dim_compress(inp, dim)

        # [核心修复 2] 弃用 clone()，强制转为真正的标准连续内存
        # clone() 默认 memory_format=torch.preserve_format，如果张量本就被视为 contiguous 
        # (例如某些维度为1的情况)，即使 stride 错乱它依然保留怪异 stride，导致越界。
        if not inp.is_contiguous() or (inp.ndim > 0 and inp.stride(-1) != 1):
            inp = inp.contiguous()

        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            aminmax_kernel[grid](inp, min_out, max_out, M, N, UPCAST=is_upcast)
            
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out
