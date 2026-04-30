"""
Token Sparse Attention Backend for vLLM

基于 Token-Sparse-Attention
与 vLLM FlashAttention Backend 接口规范集成

核心思想：在 Prefill 阶段基于累积注意力权重选择 top-k tokens，
在压缩空间内执行 FlashAttention，显著降低长上下文推理的计算成本。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# ============================================================================
# Triton Kernel: 计算 Prefix Attention Cache
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["L", "D", "W"],
)
@triton.jit
def _stats_m_l_kernel(
    Q_ptr,
    K_ptr,
    M_ptr,
    L_ptr,  # (B,H,W) - max and sum for online softmax
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_ql: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kl: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_mw: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lw: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    """计算每个 query 在 window 内的 max 和 sum (online softmax stats)"""
    pid_bh = tl.program_id(0)
    pid_qc = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    # Window query indices
    q_off = pid_qc * BQ + tl.arange(0, BQ)
    q_mask = q_off < W
    q_pos = (L - W) + q_off

    # Load Q
    d = tl.arange(0, D)
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + q_pos[:, None] * stride_ql
        + d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax
    m = tl.full([BQ], -float("inf"), tl.float32)
    l_acc = tl.zeros([BQ], tl.float32)

    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))
    win_start = L - W

    k0 = 0
    while k0 < L:
        k_idx = k0 + tl.arange(0, BK)
        k_mask = k_idx < L

        k_ptrs = (
            K_ptr
            + b * stride_kb
            + h * stride_kh
            + k_idx[None, :] * stride_kl
            + d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d

        # Causal masking only inside window
        in_window_k = k_idx[None, :] >= win_start
        future = k_idx[None, :] > q_pos[:, None]
        scores = tl.where(in_window_k & future, -float("inf"), scores)

        # Online softmax update
        row_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, row_max)
        l_acc = l_acc * tl.exp(m - m_new) + tl.sum(
            tl.exp(scores - m_new[:, None]), axis=1
        )
        m = m_new

        k0 += BK

    # Store results
    m_ptrs = M_ptr + b * stride_mb + h * stride_mh + q_off * stride_mw
    l_ptrs = L_ptr + b * stride_lb + h * stride_lh + q_off * stride_lw
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l_acc, mask=q_mask)


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["L", "D", "W"],
)
@triton.jit
def _prefix_meanprob_kernel(
    Q_ptr,
    K_ptr,
    M_ptr,
    L_ptr,
    OUT_ptr,  # (B,H,L-W) - prefix attention cache
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_ql: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kl: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_mh: tl.constexpr,
    stride_mw: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_lw: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ok: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    """计算 prefix token 的累积注意力权重"""
    pid_bh = tl.program_id(0)
    pid_kb = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    prefix_len = L - W
    k_idx = pid_kb * BK + tl.arange(0, BK)
    k_mask = k_idx < prefix_len

    d = tl.arange(0, D)
    k_ptrs = (
        K_ptr
        + b * stride_kb
        + h * stride_kh
        + k_idx[None, :] * stride_kl
        + d[:, None] * stride_kd
    )
    k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros([BK], tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))

    q0 = 0
    while q0 < W:
        q_off = q0 + tl.arange(0, BQ)
        q_mask = q_off < W
        q_pos = (L - W) + q_off

        q_ptrs = (
            Q_ptr
            + b * stride_qb
            + h * stride_qh
            + q_pos[:, None] * stride_ql
            + d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        m_ptrs = M_ptr + b * stride_mb + h * stride_mh + q_off * stride_mw
        l_ptrs = L_ptr + b * stride_lb + h * stride_lh + q_off * stride_lw
        m = tl.load(m_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)
        l_sum = tl.load(l_ptrs, mask=q_mask, other=1.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d
        probs = tl.exp(scores - m[:, None]) / l_sum[:, None]
        acc += tl.sum(probs, axis=0)

        q0 += BQ

    acc *= 1.0 / tl.full([], W, tl.float32)

    out_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + k_idx * stride_ok
    tl.store(out_ptrs, acc, mask=k_mask)


# ============================================================================
# Packed Triton Kernels: 支持 cu_seqlens 变长序列
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["D", "MAX_W"],
)
@triton.jit
def _stats_m_l_kernel_packed(
    Q_ptr,
    K_ptr,
    M_ptr,
    L_ptr,  # (num_seqs*H, W) - max and sum for online softmax
    cu_seqlens,  # (num_seqs + 1,) int32 tensor
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_W: tl.constexpr,
    stride_q0: tl.constexpr,
    stride_q1: tl.constexpr,
    stride_q2: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_m0: tl.constexpr,
    stride_m1: tl.constexpr,
    stride_l0: tl.constexpr,
    stride_l1: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    """
    计算每个 query 在 window 内的 max 和 sum (online softmax stats)
    支持变长序列，通过 cu_seqlens 查找每个序列的边界。

    Grid: (num_seqs * H, triton.cdiv(MAX_W, 16))
    - program_id(0): seq * H + h
    - program_id(1): query chunk within window
    """
    pid_seq_h = tl.program_id(0)
    pid_qc = tl.program_id(1)

    h = pid_seq_h % H
    seq_idx = pid_seq_h // H

    # 加载序列边界（cast 到 int64 用于 pointer offset）
    seq_start = tl.cast(tl.load(cu_seqlens + seq_idx), tl.int64)
    seq_end = tl.cast(tl.load(cu_seqlens + seq_idx + 1), tl.int64)
    seq_len = tl.cast(seq_end - seq_start, tl.int32)

    # 动态 window 大小（不超过序列长度）
    W = MAX_W if MAX_W < seq_len else seq_len
    prefix_len = tl.cast(seq_len - W, tl.int32)

    # Window query indices（相对位置）
    q_off = pid_qc * BQ + tl.arange(0, BQ)
    q_mask = q_off < W

    # 在 packed tensor 中的绝对位置
    q_pos = seq_start + prefix_len + q_off

    # 加载 Q: (total_tokens, H, D), stride = (H*D, D, 1)
    d = tl.arange(0, D)
    q_ptrs = Q_ptr + q_pos[:, None] * stride_q0 + h * stride_q1 + d[None, :] * stride_q2
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax
    m = tl.full([BQ], -float("inf"), tl.float32)
    l_acc = tl.zeros([BQ], tl.float32)

    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))
    win_start = seq_start + prefix_len

    k0 = 0
    while k0 < seq_len:
        k_idx = k0 + tl.arange(0, BK)
        k_mask = k_idx < seq_len
        k_pos = seq_start + tl.cast(k_idx, tl.int64)

        k_ptrs = (
            K_ptr + k_pos[None, :] * stride_k0 + h * stride_k1 + d[:, None] * stride_k2
        )
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d

        # Causal masking only inside window
        in_window_k = k_idx[None, :] >= prefix_len
        future = k_pos[None, :] > (seq_start + prefix_len + q_off)[:, None]
        scores = tl.where(in_window_k & future, -float("inf"), scores)

        # Online softmax update
        row_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, row_max)
        l_acc = l_acc * tl.exp(m - m_new) + tl.sum(
            tl.exp(scores - m_new[:, None]), axis=1
        )
        m = m_new

        k0 += BK

    # 存储结果
    m_ptrs = M_ptr + pid_seq_h * stride_m0 + q_off * stride_m1
    l_ptrs = L_ptr + pid_seq_h * stride_l0 + q_off * stride_l1
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l_acc, mask=q_mask)


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["D", "MAX_W"],
)
@triton.jit
def _prefix_meanprob_kernel_packed(
    Q_ptr,
    K_ptr,
    M_ptr,
    L_ptr,
    OUT_ptr,  # (total_tokens, H) - prefix attention cache
    cu_seqlens,  # (num_seqs + 1,) int32 tensor
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_W: tl.constexpr,
    stride_q0: tl.constexpr,
    stride_q1: tl.constexpr,
    stride_q2: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_m0: tl.constexpr,
    stride_m1: tl.constexpr,
    stride_l0: tl.constexpr,
    stride_l1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    BK: tl.constexpr,
    BQ: tl.constexpr,
):
    """
    计算 prefix token 的累积注意力权重
    支持变长序列。

    Grid: (num_seqs * H, triton.cdiv(max_prefix, 64))
    """
    pid_seq_h = tl.program_id(0)
    pid_kb = tl.program_id(1)

    h = pid_seq_h % H
    seq_idx = pid_seq_h // H

    # 加载序列边界（cast 到 int64 用于 pointer offset）
    seq_start = tl.cast(tl.load(cu_seqlens + seq_idx), tl.int64)
    seq_end = tl.cast(tl.load(cu_seqlens + seq_idx + 1), tl.int64)
    seq_len = tl.cast(seq_end - seq_start, tl.int32)

    # 动态 prefix_len: W 是 window_size (MAX_W) 与 seq_len 的较小值
    W = MAX_W if MAX_W < seq_len else seq_len
    prefix_len = tl.cast(seq_len - W, tl.int32)

    k_idx = pid_kb * BK + tl.arange(0, BK)
    k_mask = k_idx < prefix_len
    k_pos = seq_start + k_idx

    d = tl.arange(0, D)
    k_ptrs = K_ptr + k_pos[None, :] * stride_k0 + h * stride_k1 + d[:, None] * stride_k2
    k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros([BK], tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))

    q0 = 0
    while q0 < W:
        q_off = q0 + tl.arange(0, BQ)
        q_mask = q_off < W
        q_pos = seq_start + prefix_len + q_off

        q_ptrs = (
            Q_ptr + q_pos[:, None] * stride_q0 + h * stride_q1 + d[None, :] * stride_q2
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

        m_ptrs = M_ptr + pid_seq_h * stride_m0 + q_off * stride_m1
        l_ptrs = L_ptr + pid_seq_h * stride_l0 + q_off * stride_l1
        m = tl.load(m_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)
        l_sum = tl.load(l_ptrs, mask=q_mask, other=1.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d
        probs = tl.exp(scores - m[:, None]) / l_sum[:, None]
        acc += tl.sum(probs, axis=0)

        q0 += BQ

    acc *= 1.0 / tl.full([], W, tl.float32)

    out_ptrs = OUT_ptr + k_pos * stride_o0 + h * stride_o1
    tl.store(out_ptrs, acc, mask=k_mask)


def triton_get_attn_cache(
    query_states: torch.Tensor,  # (B,H,L,D)
    key_states: torch.Tensor,  # (B,H,L,D)
    window_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算累积注意力权重: mean_q softmax(Q_window @ K^T)[prefix_keys]
    返回 Shape: (B, H, L-W)
    """
    assert query_states.is_cuda and key_states.is_cuda
    assert query_states.ndim == 4 and key_states.ndim == 4
    B, H, L, D = query_states.shape
    B2, H2, L2, D2 = key_states.shape
    assert (B, H, L, D) == (B2, H2, L2, D2)

    W = window_size
    prefix_len = L - W
    if prefix_len <= 0:
        return query_states.new_zeros((B, H, 0), dtype=torch.float32)

    # Allocate output buffers
    m = torch.empty((B, H, W), device=query_states.device, dtype=torch.float32)
    l_sum = torch.empty((B, H, W), device=query_states.device, dtype=torch.float32)
    out = torch.empty(
        (B, H, prefix_len), device=query_states.device, dtype=torch.float32
    )

    # Kernel 1: 计算 online softmax stats
    grid_stats = (B * H, triton.cdiv(W, 16))
    _stats_m_l_kernel[grid_stats](
        query_states,
        key_states,
        m,
        l_sum,
        B=B,
        H=H,
        L=L,
        D=D,
        W=W,
        stride_qb=query_states.stride(0),
        stride_qh=query_states.stride(1),
        stride_ql=query_states.stride(2),
        stride_qd=query_states.stride(3),
        stride_kb=key_states.stride(0),
        stride_kh=key_states.stride(1),
        stride_kl=key_states.stride(2),
        stride_kd=key_states.stride(3),
        stride_mb=m.stride(0),
        stride_mh=m.stride(1),
        stride_mw=m.stride(2),
        stride_lb=l_sum.stride(0),
        stride_lh=l_sum.stride(1),
        stride_lw=l_sum.stride(2),
    )

    # Kernel 2: 计算 prefix 平均注意力
    grid_prob = (B * H, triton.cdiv(prefix_len, 64))
    _prefix_meanprob_kernel[grid_prob](
        query_states,
        key_states,
        m,
        l_sum,
        out,
        B=B,
        H=H,
        L=L,
        D=D,
        W=W,
        stride_qb=query_states.stride(0),
        stride_qh=query_states.stride(1),
        stride_ql=query_states.stride(2),
        stride_qd=query_states.stride(3),
        stride_kb=key_states.stride(0),
        stride_kh=key_states.stride(1),
        stride_kl=key_states.stride(2),
        stride_kd=key_states.stride(3),
        stride_mb=m.stride(0),
        stride_mh=m.stride(1),
        stride_mw=m.stride(2),
        stride_lb=l_sum.stride(0),
        stride_lh=l_sum.stride(1),
        stride_lw=l_sum.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ok=out.stride(2),
    )
    return out.to(out_dtype)


def triton_get_attn_cache_packed(
    query_states: torch.Tensor,  # (total_tokens, H, D) packed
    key_states: torch.Tensor,  # (total_tokens, H, D) packed
    cu_seqlens: torch.Tensor,  # (num_seqs + 1,) int32
    window_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算累积注意力权重: mean_q softmax(Q_window @ K^T)[prefix_keys]
    支持变长序列，一次性处理所有序列。

    Args:
        query_states: (total_tokens, H, D) packed tensor
        key_states: (total_tokens, H, D) packed tensor
        cu_seqlens: (num_seqs + 1,) int32, cumulative sequence lengths
        window_size: attention window size
    Returns:
        Shape: (total_tokens, H) — 仅 prefix 位置有有效值，
               decode/token 位置（窗口内）为 0
    """
    assert query_states.is_cuda and key_states.is_cuda
    assert query_states.ndim == 3 and key_states.ndim == 3
    total_tokens, H, D = query_states.shape
    total_tokens_k, H2, D2 = key_states.shape
    assert (total_tokens, H, D) == (total_tokens_k, H2, D2)

    num_seqs = cu_seqlens.shape[0] - 1

    # 计算每个序列的 prefix_len
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # (num_seqs,)
    W = window_size
    prefix_lens = torch.clamp(seq_lens - W, min=0)  # (num_seqs,)

    max_W = W  # window_size 是上界
    max_prefix = int(prefix_lens.max().item())

    if max_prefix <= 0:
        return query_states.new_zeros((total_tokens, H), dtype=torch.float32)

    # Allocate output buffers
    # m, l_sum: (num_seqs * H, W) — 每个 seq*head 组合有 W 个元素
    m = torch.empty((num_seqs * H, W), device=query_states.device, dtype=torch.float32)
    l_sum = torch.empty(
        (num_seqs * H, W), device=query_states.device, dtype=torch.float32
    )
    # out: (total_tokens, H) — 仅 prefix 位置有值，window 位置自动为 0
    out = torch.zeros(
        (total_tokens, H), device=query_states.device, dtype=torch.float32
    )

    # Strides for packed (total_tokens, H, D): (H*D, D, 1)
    stride_q0 = H * D
    stride_q1 = D
    stride_q2 = 1
    stride_k0 = H * D
    stride_k1 = D
    stride_k2 = 1

    # Kernel 1: 计算 online softmax stats
    grid_stats = (num_seqs * H, triton.cdiv(max_W, 16))
    _stats_m_l_kernel_packed[grid_stats](
        query_states,
        key_states,
        m,
        l_sum,
        cu_seqlens,
        H=H,
        D=D,
        MAX_W=max_W,
        stride_q0=stride_q0,
        stride_q1=stride_q1,
        stride_q2=stride_q2,
        stride_k0=stride_k0,
        stride_k1=stride_k1,
        stride_k2=stride_k2,
        stride_m0=m.stride(0),
        stride_m1=m.stride(1),
        stride_l0=l_sum.stride(0),
        stride_l1=l_sum.stride(1),
    )

    # Kernel 2: 计算 prefix 平均注意力
    grid_prob = (num_seqs * H, triton.cdiv(max_prefix, 64))
    _prefix_meanprob_kernel_packed[grid_prob](
        query_states,
        key_states,
        m,
        l_sum,
        out,
        cu_seqlens,
        H=H,
        D=D,
        MAX_W=max_W,
        stride_q0=stride_q0,
        stride_q1=stride_q1,
        stride_q2=stride_q2,
        stride_k0=stride_k0,
        stride_k1=stride_k1,
        stride_k2=stride_k2,
        stride_m0=m.stride(0),
        stride_m1=m.stride(1),
        stride_l0=l_sum.stride(0),
        stride_l1=l_sum.stride(1),
        stride_o0=out.stride(0),
        stride_o1=out.stride(1),
    )

    return out.to(out_dtype)


# ============================================================================
# Gather 和 Scatter 操作 (Native PyTorch)
# ============================================================================


def torch_gather(
    input: torch.Tensor,  # (B, H, L, D)
    indices: torch.Tensor,  # (B, H, K, D) - expanded indices (same index for all D)
) -> torch.Tensor:
    """
    Gather operation using native PyTorch.
    input[B,H,L,D] + indices[B,H,K,D] -> output[B,H,K,D]

    Note: indices is (B, H, K, D) with same index for all D elements.
    The D dimension of indices is ignored - index at (b, h, k) is used for all d.
    """
    return torch.gather(input, dim=2, index=indices)


def torch_scatter(
    input: torch.Tensor,  # (B, H, K, D)
    indices: torch.Tensor,  # (B, H, K, D) - expanded indices (same index for all D)
    output_size: int,  # L
) -> torch.Tensor:
    """
    Scatter operation using native PyTorch.
    input[B,H,K,D] + indices[B,H,K,D] -> output[B,H,L,D]

    Note: indices is (B, H, K, D) with same index for all D elements.
    The D dimension of indices is ignored - index at (b, h, k) is used for all d.
    """
    B, H, K, D = input.shape
    output = torch.zeros(B, H, output_size, D, dtype=input.dtype, device=input.device)
    output.scatter_(dim=2, index=indices, src=input)
    return output


# ============================================================================
# Token Sparse 核心逻辑
# ============================================================================


class TokenSparseSelector:
    """
    基于累积注意力权重的 token 选择器

    核心思想：
    1. 计算 prefix token 的累积注意力权重
    2. 基于 coverage τ 选择覆盖 τ% 注意力所需的 top-k tokens
    3. 保留 window_size 个局部 tokens (attention sink)
    """

    def __init__(
        self,
        coverage: float = 0.005,  # τ: 覆盖的注意力比例
        min_tokens: int = 1024,  # 最少保留的 token 数
        window_size: int = 128,  # 保留的局部 window 大小
        kernel_size: int = 7,  # 平滑 kernel 大小
    ):
        self.coverage = coverage
        self.min_tokens = min_tokens
        self.window_size = window_size
        self.kernel_size = kernel_size

    def select_indices(
        self,
        query_states: torch.Tensor,  # (B, H, L, D)
        key_states: torch.Tensor,  # (B, H, L, D)
        attn_cache: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        选择需要保留的 token 索引（单序列版本）。

        Args:
            query_states: (B, H, L, D)
            key_states: (B, H, L, D)
            attn_cache: optional precomputed attn cache. If None, computed internally.

        Returns:
            indices: (B, H, k) 选中的 token 索引，或 None 表示不稀疏化
        """
        B, H, L, D = query_states.shape

        # 短序列不稀疏化
        if self.min_tokens + self.window_size >= L:
            return None

        # 获取累积注意力缓存（如果未提供）
        if attn_cache is None:
            attn_cache = triton_get_attn_cache(
                query_states, key_states, self.window_size
            )

        return self._select_indices_from_cache(attn_cache, query_states.device, L, B, H)

    def _select_indices_from_cache(
        self,
        attn_cache: torch.Tensor,  # (B, H, prefix_len)
        device: torch.device,
        L: int,
        B: int,
        H: int,
    ) -> torch.Tensor | None:
        """
        从预计算的 attn_cache 执行 token 选择。

        attn_cache 是 prefix attention cache，只包含 prefix 位置的值。
        输出 indices 需要覆盖所有 L 个位置（prefix + window）。
        """
        # 平滑处理
        attn_cache_pooled = torch.nn.functional.avg_pool1d(
            attn_cache,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        # Per-head 归一化
        attn_cache_normalized = attn_cache_pooled / (
            attn_cache_pooled.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Head-averaged 归一化
        attn_cache_global = attn_cache.mean(dim=1)  # (B, prefix_len)
        attn_cache_normalized_global = attn_cache_global / (
            attn_cache_global.sum(dim=-1, keepdim=True) + 1e-8
        )

        # 累积求和，找到覆盖 coverage 所需的 token 数
        sorted_attn, _ = torch.sort(
            attn_cache_normalized_global, dim=-1, descending=False
        )
        cumulative_coverage = torch.cumsum(sorted_attn.float(), dim=-1)

        num_sparse_tokens = (cumulative_coverage < self.coverage).sum(dim=-1)
        if num_sparse_tokens.dim() > 0:
            num_sparse_tokens = num_sparse_tokens[0]
        num_sparse_tokens = num_sparse_tokens.item()
        if num_sparse_tokens == 0:
            return None

        num_sparse_tokens += 1

        # 选择 top-k tokens
        max_capacity = max(self.min_tokens, L - self.window_size - num_sparse_tokens)
        _, topk_indices = torch.topk(attn_cache_normalized, k=max_capacity, dim=-1)

        # 添加 window tokens
        window_indices = (
            torch.arange(L - self.window_size, L, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, H, -1)
        )

        # 合并并排序
        indices = torch.cat([topk_indices, window_indices], dim=-1)
        indices, _ = torch.sort(indices, dim=-1)

        return indices

    def compress_qkv(
        self,
        query: torch.Tensor,  # (B, H, L, D)
        key: torch.Tensor,  # (B, H, L, D)
        value: torch.Tensor,  # (B, H, L, D)
        indices: torch.Tensor,  # (B, H, k)
    ):
        """
        在压缩空间内提取 QKV
        """
        B, H, L, D = query.shape
        _, _, k = indices.shape

        # Expand indices: (B, H, k) -> (B, H, k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        q_sparse = torch_gather(query, indices_expanded)
        k_sparse = torch_gather(key, indices_expanded)
        v_sparse = torch_gather(value, indices_expanded)

        return q_sparse, k_sparse, v_sparse

    def decompress_output(
        self,
        attn_output: torch.Tensor,  # (B, H, k, D) 压缩空间输出
        indices: torch.Tensor,  # (B, H, k)
        original_length: int,
    ) -> torch.Tensor:
        """
        将压缩空间输出解压回原始维度
        """
        B, H, k, D = attn_output.shape

        # Expand indices: (B, H, k) -> (B, H, k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D)

        return torch_scatter(attn_output, indices_expanded, original_length)


# ============================================================================
# vLLM Attention Backend Interface
# ============================================================================


class TokenSparseAttentionBackend(FlashAttentionBackend):
    """
    Token Sparse Attention Backend for vLLM

    特性：
    - 支持 Prefill 阶段稀疏注意力
    - Decode 阶段保持 PagedAttention 原生实现
    - 与 FlashAttention 完全兼容
    - 可配置 coverage、window_size、kernel_size 等参数
    """

    @staticmethod
    def get_name() -> str:
        return "TOKEN_SPARSE"

    @staticmethod
    def get_impl_cls() -> type[TokenSparseAttentionImpl]:
        return TokenSparseAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TokenSparseAttentionMetadataBuilder]:
        return TokenSparseAttentionMetadataBuilder


@dataclass
class TokenSparseAttentionMetadata(FlashAttentionMetadata):
    """Token Sparse Attention 元数据，继承自 FlashAttentionMetadata"""

    # Token Sparse 特有字段
    sparse_indices: torch.Tensor | None = None  # 稀疏选择的 token 索引
    is_prefill: bool = True  # 是否为 prefill 阶段

    # 元信息
    head_dim: int = 128
    num_kv_heads: int = 0


class TokenSparseAttentionMetadataBuilder(
    AttentionMetadataBuilder[TokenSparseAttentionMetadata],
):
    # Same cudagraph support as FlashAttention since we inherit FlashAttentionImpl
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    """Token Sparse Attention 元数据构建器"""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config

        # Token Sparse 配置 - 从环境变量读取，使用 hf_config 中的值（如果模型配置中有）
        # 否则使用默认值
        hf_config = self.model_config.hf_text_config
        self.coverage = float(
            os.environ.get(
                "VLLM_TOKEN_SPARSE_COVERAGE",
                getattr(hf_config, "token_sparse_coverage", 0.005),
            )
        )
        self.window_size = int(
            os.environ.get(
                "VLLM_TOKEN_SPARSE_WINDOW_SIZE",
                getattr(hf_config, "token_sparse_window_size", 128),
            )
        )
        self.min_tokens = int(
            os.environ.get(
                "VLLM_TOKEN_SPARSE_MIN_TOKENS",
                getattr(hf_config, "token_sparse_min_tokens", 1024),
            )
        )
        self.kernel_size = int(
            os.environ.get(
                "VLLM_TOKEN_SPARSE_KERNEL_SIZE",
                getattr(hf_config, "token_sparse_kernel_size", 7),
            )
        )

        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        logger.debug(
            "TokenSparseAttentionMetadataBuilder initialized: "
            "coverage=%s, window_size=%s, min_tokens=%s",
            self.coverage,
            self.window_size,
            self.min_tokens,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TokenSparseAttentionMetadata:
        """构建 Token Sparse Attention 元数据"""
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        slot_mapping = common_attn_metadata.slot_mapping
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        max_seq_len = common_attn_metadata.max_seq_len

        # Determine if prefill based on query length
        is_prefill = max_query_len > 1

        return TokenSparseAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            is_prefill=is_prefill,
            # FlashAttentionMetadata cascade fields (not used in token sparse)
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )


class TokenSparseAttentionImpl(FlashAttentionImpl):
    """
    Token Sparse Attention 实现，继承自 FlashAttentionImpl。

    工作流程：
    1. Prefill 阶段：Token 选择 -> 压缩 -> FlashAttention -> 解压
    2. Decode 阶段：使用父类的标准 FlashAttention
    """

    # Token Sparse 最小序列长度阈值：小于此值时回退到 FlashAttention
    # vLLM 使用 chunked prefill，短 chunk 稀疏化收益低，开销大
    SPARSE_MIN_SEQ_LEN = int(os.environ.get("VLLM_TOKEN_SPARSE_MIN_SEQ_LEN", "32768"))

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        # Token Sparse 特有参数
        token_sparse_coverage: float | None = None,
        token_sparse_window_size: int | None = None,
        token_sparse_min_tokens: int | None = None,
        token_sparse_kernel_size: int | None = None,
        # Layer 配置
        layer_name: str | None = None,
    ):
        # 调用父类初始化，处理通用的 FlashAttention 字段
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )

        # Token Sparse 选择器
        self.layer_idx = self._extract_layer_index(layer_name)

        # 从环境变量读取 sparse_layers 配置
        # 格式: "15,23,31,39,47,53,58"
        sparse_layers_env = os.environ.get(
            "VLLM_TOKEN_SPARSE_SPARSE_LAYERS",
            "30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50",
        )
        if sparse_layers_env:
            self.sparse_layers = [
                int(x.strip()) for x in sparse_layers_env.split(",") if x.strip()
            ]
        else:
            self.sparse_layers = list(range(100))  # 默认所有层

        # 是否启用稀疏化（可在层级别配置）
        # 逻辑：已知 layer_idx 时，检查是否在 sparse_layers 中
        #       未知 layer_idx 时（无法解析），默认启用（因为 sparse_layers 默认为所有层）
        if self.layer_idx is not None:
            self.use_sparse = self.layer_idx in self.sparse_layers
        else:
            # 无法确定层索引：
            self.use_sparse = False

        # Token Sparse 参数 - 从环境变量读取，使用默认值
        # 这些参数也可以通过 hf_config 传递
        if token_sparse_coverage is not None:
            coverage = token_sparse_coverage
        else:
            coverage = float(os.environ.get("VLLM_TOKEN_SPARSE_COVERAGE", "0.005"))

        if token_sparse_window_size is not None:
            window_size = token_sparse_window_size
        else:
            window_size = int(os.environ.get("VLLM_TOKEN_SPARSE_WINDOW_SIZE", "128"))

        if token_sparse_min_tokens is not None:
            min_tokens = token_sparse_min_tokens
        else:
            min_tokens = int(os.environ.get("VLLM_TOKEN_SPARSE_MIN_TOKENS", "1024"))

        if token_sparse_kernel_size is not None:
            kernel_size = token_sparse_kernel_size
        else:
            kernel_size = int(os.environ.get("VLLM_TOKEN_SPARSE_KERNEL_SIZE", "7"))

        logger.debug(
            "TokenSparseAttentionImpl initialized: "
            "layer_idx=%s, use_sparse=%s, coverage=%s, window_size=%s, "
            "min_tokens=%s, kernel_size=%s, sparse_layers=%s, "
            "min_seq_len=%s",
            self.layer_idx,
            self.use_sparse,
            coverage,
            window_size,
            min_tokens,
            kernel_size,
            self.sparse_layers,
            self.SPARSE_MIN_SEQ_LEN,
        )

        if self.use_sparse:
            self.selector = TokenSparseSelector(
                coverage=coverage,
                min_tokens=min_tokens,
                window_size=window_size,
                kernel_size=kernel_size,
            )
        else:
            self.selector = None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TokenSparseAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        # 如果不满足稀疏条件，调用父类的 forward
        # 注意：在 vLLM v1 中，prefill 可能 chunked 执行，每个 chunk 的
        # num_tokens 可能只有 1，但 max_query_len 反映的是整个 batch 中
        # 最大序列的长度。所以用 max_query_len 来判断是否是 prefill 阶段。
        apply_sparse = (
            self.use_sparse
            and self.selector is not None
            and attn_metadata is not None
            and (
                attn_metadata.max_query_len
                > self.selector.min_tokens + self.selector.window_size
            )
            and attn_metadata.max_query_len > self.SPARSE_MIN_SEQ_LEN
        )

        if not apply_sparse:
            logger.debug(
                "TokenSparse forward: layer_idx=%s, using flash attention "
                "(sparse=%s, selector=%s)",
                self.layer_idx,
                self.use_sparse,
                self.selector is not None,
            )
            return super().forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        # Token Sparse 路径：使用 packed tensor 格式
        # 按 cu_seqlens_q 拆分每个序列，逐个执行 Token Sparse，然后拼接
        num_tokens = query.shape[0]
        output_3d = self._forward_sparse_packed(
            layer=layer,
            query=query,  # (num_tokens, H, D) — packed
            key=key,  # (num_tokens, H, D) — packed
            value=value,  # (num_tokens, H, D) — packed
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        # 写回 output (output 是 3D view: (num_tokens, H, D))
        output.copy_(output_3d)

        logger.debug(
            "TokenSparse forward: layer_idx=%s, num_tokens=%s, output.shape=%s",
            self.layer_idx,
            num_tokens,
            output.shape,
        )
        return output

    def _forward_sparse_packed(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # (num_tokens, num_heads, D) — packed
        key: torch.Tensor,  # (num_tokens, num_kv_heads, D) — packed
        value: torch.Tensor,  # (num_tokens, num_kv_heads, D) — packed
        kv_cache: torch.Tensor,
        attn_metadata: TokenSparseAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        处理 packed multi-sequence 输入的 Token Sparse Attention。

        使用 triton_get_attn_cache_packed 一次计算所有序列的 attention cache，
        然后对每个序列执行 Token Sparse，最后拼接结果。
        """
        cu_seqlens_q = attn_metadata.query_start_loc  # (num_seqs + 1,)
        num_seqs = cu_seqlens_q.shape[0] - 1
        H = query.shape[1]  # num_heads_q
        H_kv = key.shape[1]  # num_kv_heads
        D = query.shape[2]
        num_tokens = query.shape[0]

        # 保存原始 KV（num_kv_heads），用于 super().forward()
        key_original = key
        value_original = value

        # GQA: repeat KV heads to match Q heads for triton_get_attn_cache_packed
        # key/value: (total_tokens, num_kv_heads, D) -> (total_tokens, num_heads, D)
        if self.num_queries_per_kv > 1:
            key = (
                key.reshape(num_tokens, H_kv, 1, D)
                .expand(num_tokens, H_kv, self.num_queries_per_kv, D)
                .reshape(num_tokens, H, D)
            )
            value = (
                value.reshape(num_tokens, H_kv, 1, D)
                .expand(num_tokens, H_kv, self.num_queries_per_kv, D)
                .reshape(num_tokens, H, D)
            )

        # Step 0: 一次性计算所有序列的 attention cache
        # attn_cache_packed: (total_tokens, H) — prefix 位置有值，window 位置为 0
        attn_cache_packed = triton_get_attn_cache_packed(
            query, key, cu_seqlens_q, self.selector.window_size
        )

        # 一次性取到 CPU，避免循环内多次 .item() sync
        cu_seqlens_list = cu_seqlens_q.cpu().tolist()

        # 第一遍：分类 fallback 和 sparse 序列
        fallback_seqs: list[
            tuple[int, int, int, int]
        ] = []  # (seq_idx, start, end, seq_len)
        sparse_seqs: list[
            tuple[int, int, int, int]
        ] = []  # (seq_idx, start, end, seq_len)

        for seq_idx in range(num_seqs):
            start = cu_seqlens_list[seq_idx]
            end = cu_seqlens_list[seq_idx + 1]
            seq_len = end - start
            if seq_len <= self.selector.min_tokens + self.selector.window_size:
                fallback_seqs.append((seq_idx, start, end, seq_len))
            else:
                sparse_seqs.append((seq_idx, start, end, seq_len))

        output_3d_list: list[torch.Tensor | None] = [None] * num_seqs

        # Fallback batching：一次性处理所有 fallback 序列
        if fallback_seqs:
            q_fb = torch.cat([query[s:e] for _, s, e, _ in fallback_seqs])
            k_fb = torch.cat([key_original[s:e] for _, s, e, _ in fallback_seqs])
            v_fb = torch.cat([value_original[s:e] for _, s, e, _ in fallback_seqs])
            fb_total = q_fb.shape[0]
            out_fb = torch.empty(
                fb_total, H, D, device=output.device, dtype=output.dtype
            )

            fb_seq_lens = [seq_len for _, _, _, seq_len in fallback_seqs]
            fb_cu_list = [0]
            for l in fb_seq_lens:
                fb_cu_list.append(fb_cu_list[-1] + l)
            fb_cu = torch.tensor(fb_cu_list, device=query.device, dtype=torch.int32)

            fb_indices = [seq_idx for seq_idx, _, _, _ in fallback_seqs]
            fb_block_table = attn_metadata.block_table[fb_indices]
            fb_slot_mapping = torch.cat(
                [attn_metadata.slot_mapping[s:e] for _, s, e, _ in fallback_seqs]
            )

            fb_metadata = TokenSparseAttentionMetadata(
                num_actual_tokens=fb_total,
                max_query_len=max(fb_seq_lens),
                query_start_loc=fb_cu,
                max_seq_len=max(fb_seq_lens),
                seq_lens=torch.tensor(
                    fb_seq_lens, device=query.device, dtype=torch.int32
                ),
                block_table=fb_block_table,
                slot_mapping=fb_slot_mapping,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                max_dcp_context_kv_len=None,
                dcp_context_kv_lens=None,
                scheduler_metadata=None,
                prefix_scheduler_metadata=None,
                max_num_splits=0,
                causal=True,
                sparse_indices=None,
                is_prefill=True,
                head_dim=attn_metadata.head_dim,
                num_kv_heads=attn_metadata.num_kv_heads,
            )

            super().forward(
                layer=layer,
                query=q_fb,
                key=k_fb,
                value=v_fb,
                kv_cache=kv_cache,
                attn_metadata=fb_metadata,
                output=out_fb,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

            offset = 0
            for seq_idx, _, _, seq_len in fallback_seqs:
                output_3d_list[seq_idx] = out_fb[offset : offset + seq_len].reshape(
                    seq_len, H, D
                )
                offset += seq_len

        # Sparse 路径：收集所有压缩 QKV，统一调用一次 FA，再逐个解压
        q_sparse_list: list[torch.Tensor] = []
        k_sparse_list: list[torch.Tensor] = []
        v_sparse_list: list[torch.Tensor] = []
        # (seq_idx, seq_len, k, offset, indices)
        sparse_compute: list[tuple[int, int, int, int, torch.Tensor]] = []
        running_offset = 0

        for seq_idx, start, end, seq_len in sparse_seqs:
            # 提取该序列的 attention cache
            seq_cache = attn_cache_packed[start:end]  # (seq_len, H)
            seq_cache_t = seq_cache.t().unsqueeze(0)  # (1, H, seq_len)

            # 提取 QKV，转换为 (1, seq_len, H, D)
            seq_q = query[start:end].unsqueeze(0)  # (1, L, H, D)
            seq_k = key[start:end].unsqueeze(0)  # (1, L, H, D)
            seq_v = value[start:end].unsqueeze(0)  # (1, L, H, D)

            # Step 1: Token 选择
            seq_q_t = seq_q.transpose(1, 2)  # (1, H, L, D)
            seq_k_t = seq_k.transpose(1, 2)
            seq_v_t = seq_v.transpose(1, 2)

            indices = self.selector.select_indices(seq_q_t, seq_k_t, seq_cache_t)

            if indices is None:
                logger.info(
                    "TokenSparse _forward_sparse_packed: layer_idx=%s, "
                    "sequence too short (L=%s), falling back to flash attention",
                    self.layer_idx,
                    seq_len,
                )
                seq_q_3d = seq_q.squeeze(0)
                seq_k_3d = key_original[start:end]
                seq_v_3d = value_original[start:end]
                temp_out = output[start:end]
                seq_metadata = TokenSparseAttentionMetadata(
                    num_actual_tokens=seq_len,
                    max_query_len=seq_len,
                    query_start_loc=torch.tensor(
                        [0, seq_len], device=query.device, dtype=torch.int32
                    ),
                    max_seq_len=seq_len,
                    seq_lens=torch.tensor(
                        [seq_len], device=query.device, dtype=torch.int32
                    ),
                    block_table=(
                        attn_metadata.block_table[seq_idx : seq_idx + 1]
                        if attn_metadata.block_table.numel() > 0
                        else attn_metadata.block_table
                    ),
                    slot_mapping=attn_metadata.slot_mapping[start:end],
                    use_cascade=False,
                    common_prefix_len=0,
                    cu_prefix_query_lens=None,
                    prefix_kv_lens=None,
                    suffix_kv_lens=None,
                    max_dcp_context_kv_len=None,
                    dcp_context_kv_lens=None,
                    scheduler_metadata=None,
                    prefix_scheduler_metadata=None,
                    max_num_splits=0,
                    causal=True,
                    sparse_indices=None,
                    is_prefill=True,
                    head_dim=attn_metadata.head_dim,
                    num_kv_heads=attn_metadata.num_kv_heads,
                )
                super().forward(
                    layer=layer,
                    query=seq_q_3d,
                    key=seq_k_3d,
                    value=seq_v_3d,
                    kv_cache=kv_cache,
                    attn_metadata=seq_metadata,
                    output=temp_out,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                )
                output_3d_list[seq_idx] = temp_out.reshape(seq_len, H, D)
                continue

            # Step 2: 压缩 QKV
            q_sparse, k_sparse, v_sparse = self.selector.compress_qkv(
                seq_q_t, seq_k_t, seq_v_t, indices
            )
            k_selected = q_sparse.shape[2]
            compression_ratio = seq_len / k_selected if k_selected > 0 else 0

            logger.debug(
                "TokenSparse: layer_idx=%s, original_len=%s, k_selected=%s, "
                "compression_ratio=%.2fx, coverage=%s",
                self.layer_idx,
                seq_len,
                k_selected,
                compression_ratio,
                self.selector.coverage,
            )

            B_sparse, _, k, _ = q_sparse.shape
            q_sparse_list.append(q_sparse.transpose(1, 2).reshape(B_sparse * k, H, D))
            k_sparse_list.append(k_sparse.transpose(1, 2).reshape(B_sparse * k, H, D))
            v_sparse_list.append(v_sparse.transpose(1, 2).reshape(B_sparse * k, H, D))

            sparse_compute.append((seq_idx, seq_len, k, running_offset, indices))
            running_offset += B_sparse * k

        # Step 3: 统一 FlashAttention
        if sparse_compute:
            q_all = torch.cat(q_sparse_list)
            k_all = torch.cat(k_sparse_list)
            v_all = torch.cat(v_sparse_list)

            cu_list = [0]
            for _, _, k, _, _ in sparse_compute:
                cu_list.append(cu_list[-1] + k)
            cu_sparse = torch.tensor(cu_list, device=query.device, dtype=torch.int32)

            max_k = max(k for _, _, k, _, _ in sparse_compute)
            attn_out_all = torch.empty(
                running_offset, H, D, device=query.device, dtype=query.dtype
            )

            flash_attn_varlen_func(
                q=q_all,
                k=k_all,
                v=v_all,
                out=attn_out_all,
                cu_seqlens_q=cu_sparse,
                cu_seqlens_k=cu_sparse,
                max_seqlen_q=max_k,
                max_seqlen_k=max_k,
                softmax_scale=self.scale,
                causal=True,
            )

            # Step 4: 逐个解压
            for seq_idx, seq_len, k, offset, indices in sparse_compute:
                attn_out = (
                    attn_out_all[offset : offset + k]
                    .reshape(1, k, H, D)
                    .transpose(1, 2)
                )
                seq_out_4d = self.selector.decompress_output(
                    attn_out, indices, seq_len
                ).transpose(1, 2)
                output_3d_list[seq_idx] = seq_out_4d.squeeze(0)

        output_3d = torch.cat(output_3d_list, dim=0)
        return output_3d

    def _extract_layer_index(self, layer_name: str) -> int | None:
        """
        Extract layer index from layer_name string.

        Examples:
        - "model.layers.0.attn" -> 0
        - "model.layers.15.self_attn.attn" -> 15
        - "model.decoder.layers.10.attn" -> 10

        Returns None if no layer index found.
        """
        if layer_name is None:
            return None
        parts = layer_name.split(".")
        for part in reversed(parts):
            try:
                return int(part)
            except ValueError:
                continue
        return None

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update the KV cache with new key/value tensors.

        This method is called after forward() to store the key and value
        tensors in the KV cache for use in future decoding steps.
        """
        if self.attn_type != AttentionType.DECODER:
            # For non-decoder attention types, skip KV cache update
            return

        if kv_cache is None or kv_cache.numel() == 0:
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Use reshape_and_cache from flash attention utils
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA: 重复 KV heads
        输入: (B, num_kv_heads, L, D)
        输出: (B, num_kv_heads * n_rep, L, D)
        """
        if n_rep == 1:
            return x
        batch, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
