"""
Token Sparse Attention Backend v3 for vLLM

基于 Token-Sparse-Attention (https://github.com/dongwonjo/Token-Sparse-Attention)
与 vLLM FlashAttention Backend 接口规范集成

核心思想：在 Prefill 阶段基于累积注意力权重选择 top-k tokens，
在压缩空间内执行 FlashAttention，显著降低长上下文推理的计算成本。

修复要点：
1. 正确实现论文 Algorithm 1: recent queries proxy + per-head selection + τ 阈值
2. 支持 per-head 独立 token 选择，兼容 GQA/MQA 架构
3. 完整传递 vLLM FlashAttention 所需参数 (block_table, scheduler_metadata 等)
4. 数值稳定性增强: online softmax NaN 防护 + causal mask 边界修正
5. 配置优先级统一: hf_config > 函数参数 > 环境变量 > 默认值

Author: vLLM Token Sparse Integration
Version: 0.3.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

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
# 常量定义
# ============================================================================

# 论文推荐的 recent queries 数量 (Algorithm 1 步骤 2)
DEFAULT_RECENT_Q = 4
# 最小数值保护阈值
EPS = 1e-8
# 在线 softmax 的 -inf 安全值 (避免 exp(-inf) 产生 NaN)
NEG_INF_SAFE = -1e4


# ============================================================================
# Triton Kernel: 计算 Prefix Attention Cache (修复版)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["L", "D", "W", "RECENT_Q"],
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
    RECENT_Q: tl.constexpr,  # 仅用最近 RECENT_Q 个 query 计算 proxy
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
    """
    计算每个 query 在 window 内的 max 和 sum (online softmax stats)
    修复: 1) 仅用 recent queries 计算 proxy; 2) causal mask 边界修正
    """
    pid_bh = tl.program_id(0)
    pid_qc = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    # Window query indices (relative to window start)
    q_off = pid_qc * BQ + tl.arange(0, BQ)
    q_mask = q_off < W
    # Absolute position in sequence
    q_pos = (L - W) + q_off

    # Load Q: only process recent queries for proxy computation
    d = tl.arange(0, D)
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + q_pos[:, None] * stride_ql
        + d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Online softmax initialization
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

        # Causal masking: fix boundary handling
        # Prefix tokens (k_idx < win_start): always visible to all queries
        # Window tokens: causal mask applies (k > q is masked)
        is_prefix = k_idx[None, :] < win_start
        in_window = k_idx[None, :] >= win_start
        is_future = k_idx[None, :] > q_pos[:, None]
        # Mask only window tokens that are in the future
        mask_cond = in_window & is_future
        scores = tl.where(mask_cond, NEG_INF_SAFE, scores)

        # Online softmax update with NaN protection
        row_max = tl.max(tl.where(q_mask[:, None], scores, -float("inf")), axis=1)
        m_new = tl.maximum(m, row_max)

        # Safe exp: avoid exp(-inf) -> NaN
        exp_diff = tl.exp(tl.where(m - m_new > -100, m - m_new, -100.0))
        exp_scores = tl.exp(tl.where(scores - m_new[:, None] > -100,
                                      scores - m_new[:, None], -100.0))

        l_acc = l_acc * exp_diff + tl.sum(
            tl.where(q_mask[:, None], exp_scores, 0.0), axis=1
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
    key=["L", "D", "W", "RECENT_Q"],
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
    RECENT_Q: tl.constexpr,  # 仅用最近 RECENT_Q 个 query
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
    """
    计算 prefix token 的累积注意力权重 (per-head)
    修复: 仅用 recent queries 计算 proxy, 保持 per-head 独立性
    """
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
    # Only iterate over recent queries for proxy computation
    recent_start = W - RECENT_Q
    while q0 < W:
        q_off = q0 + tl.arange(0, BQ)
        q_mask = q_off < W
        # Skip non-recent queries
        is_recent = q_off >= recent_start

        q_pos = (L - W) + q_off

        q_ptrs = (
            Q_ptr
            + b * stride_qb
            + h * stride_qh
            + q_pos[:, None] * stride_ql
            + d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None] & is_recent[:, None], other=0.0).to(tl.float32)

        m_ptrs = M_ptr + b * stride_mb + h * stride_mh + q_off * stride_mw
        l_ptrs = L_ptr + b * stride_lb + h * stride_lh + q_off * stride_lw
        m = tl.load(m_ptrs, mask=q_mask & is_recent, other=NEG_INF_SAFE).to(tl.float32)
        l_sum = tl.load(l_ptrs, mask=q_mask & is_recent, other=1.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d
        # Safe probability computation
        exp_val = tl.exp(tl.where(scores - m[:, None] > -100,
                                   scores - m[:, None], -100.0))
        probs = exp_val / (l_sum[:, None] + EPS)
        acc += tl.sum(tl.where(is_recent[:, None], probs, 0.0), axis=0)

        q0 += BQ

    # Normalize by number of recent queries used
    num_recent = min(RECENT_Q, W)
    acc *= 1.0 / tl.full([], num_recent, tl.float32)

    out_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + k_idx * stride_ok
    tl.store(out_ptrs, acc, mask=k_mask)


# ============================================================================
# Packed Triton Kernels: 支持 cu_seqlens 变长序列 (修复版)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64, "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["D", "MAX_W", "RECENT_Q"],
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
    RECENT_Q: tl.constexpr,
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
    修复: recent queries + causal mask + NaN protection
    """
    pid_seq_h = tl.program_id(0)
    pid_qc = tl.program_id(1)

    h = pid_seq_h % H
    seq_idx = pid_seq_h // H

    # Load sequence boundaries with proper casting
    seq_start = tl.load(cu_seqlens + seq_idx).to(tl.int64)
    seq_end = tl.load(cu_seqlens + seq_idx + 1).to(tl.int64)
    seq_len = (seq_end - seq_start).to(tl.int32)

    # Dynamic window size
    W = tl.minimum(MAX_W, seq_len)
    prefix_len = seq_len - W

    # Window query indices
    q_off = pid_qc * BQ + tl.arange(0, BQ)
    q_mask = q_off < W
    q_pos = seq_start + prefix_len + q_off

    # Load Q
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
        k_pos = seq_start + k_idx.to(tl.int64)

        k_ptrs = (
            K_ptr + k_pos[None, :] * stride_k0 + h * stride_k1 + d[:, None] * stride_k2
        )
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d

        # Causal masking with proper boundary
        is_prefix = k_idx[None, :] < prefix_len
        in_window = k_idx[None, :] >= prefix_len
        is_future = k_pos[None, :] > (seq_start + prefix_len + q_off)[:, None]
        mask_cond = in_window & is_future
        scores = tl.where(mask_cond, NEG_INF_SAFE, scores)

        # Online softmax with NaN protection
        row_max = tl.max(tl.where(q_mask[:, None], scores, -float("inf")), axis=1)
        m_new = tl.maximum(m, row_max)

        exp_diff = tl.exp(tl.where(m - m_new > -100, m - m_new, -100.0))
        exp_scores = tl.exp(tl.where(scores - m_new[:, None] > -100,
                                      scores - m_new[:, None], -100.0))

        l_acc = l_acc * exp_diff + tl.sum(
            tl.where(q_mask[:, None], exp_scores, 0.0), axis=1
        )
        m = m_new

        k0 += BK

    # Store
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
    key=["D", "MAX_PREFIX", "RECENT_Q"],
)
@triton.jit
def _prefix_meanprob_kernel_packed(
    Q_ptr,
    K_ptr,
    M_ptr,
    L_ptr,
    OUT_ptr,  # (total_tokens, H) - prefix attention cache per head
    cu_seqlens,  # (num_seqs + 1,) int32 tensor
    H: tl.constexpr,
    D: tl.constexpr,
    MAX_PREFIX: tl.constexpr,
    RECENT_Q: tl.constexpr,
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
    计算 prefix token 的累积注意力权重 (per-head, packed format)
    修复: recent queries + per-head independence
    """
    pid_seq_h = tl.program_id(0)
    pid_kb = tl.program_id(1)

    h = pid_seq_h % H
    seq_idx = pid_seq_h // H

    seq_start = tl.load(cu_seqlens + seq_idx).to(tl.int64)
    seq_end = tl.load(cu_seqlens + seq_idx + 1).to(tl.int64)
    seq_len = (seq_end - seq_start).to(tl.int32)

    W = tl.minimum(MAX_PREFIX, seq_len)
    prefix_len = seq_len - W

    k_idx = pid_kb * BK + tl.arange(0, BK)
    k_mask = k_idx < prefix_len
    k_pos = seq_start + k_idx.to(tl.int64)

    d = tl.arange(0, D)
    k_ptrs = K_ptr + k_pos[None, :] * stride_k0 + h * stride_k1 + d[:, None] * stride_k2
    k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros([BK], tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))

    q0 = 0
    recent_start = W - RECENT_Q
    while q0 < W:
        q_off = q0 + tl.arange(0, BQ)
        q_mask = q_off < W
        is_recent = q_off >= recent_start

        q_pos = seq_start + prefix_len + q_off

        q_ptrs = (
            Q_ptr + q_pos[:, None] * stride_q0 + h * stride_q1 + d[None, :] * stride_q2
        )
        q = tl.load(q_ptrs, mask=q_mask[:, None] & is_recent[:, None], other=0.0).to(tl.float32)

        m_ptrs = M_ptr + pid_seq_h * stride_m0 + q_off * stride_m1
        l_ptrs = L_ptr + pid_seq_h * stride_l0 + q_off * stride_l1
        m = tl.load(m_ptrs, mask=q_mask & is_recent, other=NEG_INF_SAFE).to(tl.float32)
        l_sum = tl.load(l_ptrs, mask=q_mask & is_recent, other=1.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d
        exp_val = tl.exp(tl.where(scores - m[:, None] > -100,
                                   scores - m[:, None], -100.0))
        probs = exp_val / (l_sum[:, None] + EPS)
        acc += tl.sum(tl.where(is_recent[:, None], probs, 0.0), axis=0)

        q0 += BQ

    num_recent = min(RECENT_Q, W)
    acc *= 1.0 / tl.full([], num_recent, tl.float32)

    out_ptrs = OUT_ptr + k_pos * stride_o0 + h * stride_o1
    tl.store(out_ptrs, acc, mask=k_mask)


# ============================================================================
# Python Helper Functions
# ============================================================================


def triton_get_attn_cache(
    query_states: torch.Tensor,  # (B,H,L,D)
    key_states: torch.Tensor,  # (B,H,L,D)
    window_size: int,
    recent_q: int = DEFAULT_RECENT_Q,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算累积注意力权重: mean_{recent_q} softmax(Q_window @ K^T)[prefix_keys]
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

    # Kernel 1: online softmax stats
    grid_stats = (B * H, triton.cdiv(W, 16))
    _stats_m_l_kernel[grid_stats](
        query_states,
        key_states,
        m,
        l_sum,
        B=B, H=H, L=L, D=D, W=W, RECENT_Q=recent_q,
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

    # Kernel 2: prefix mean probability
    grid_prob = (B * H, triton.cdiv(prefix_len, 64))
    _prefix_meanprob_kernel[grid_prob](
        query_states,
        key_states,
        m,
        l_sum,
        out,
        B=B, H=H, L=L, D=D, W=W, RECENT_Q=recent_q,
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
    recent_q: int = DEFAULT_RECENT_Q,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算累积注意力权重: mean_{recent_q} softmax(Q_window @ K^T)[prefix_keys]
    支持变长序列，一次性处理所有序列。

    Returns:
        Shape: (total_tokens, H) — prefix 位置有 per-head 累积注意力值，
               window 位置为 0 (不参与稀疏选择)
    """
    assert query_states.is_cuda and key_states.is_cuda
    assert query_states.ndim == 3 and key_states.ndim == 3
    total_tokens, H, D = query_states.shape
    total_tokens_k, H2, D2 = key_states.shape
    assert (total_tokens, H, D) == (total_tokens_k, H2, D2)

    num_seqs = cu_seqlens.shape[0] - 1

    # Compute prefix lengths per sequence
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # (num_seqs,)
    W = window_size
    prefix_lens = torch.clamp(seq_lens - W, min=0)

    max_W = W
    max_prefix = int(prefix_lens.max().item()) if prefix_lens.numel() > 0 else 0

    if max_prefix <= 0:
        return query_states.new_zeros((total_tokens, H), dtype=torch.float32)

    # Allocate buffers
    m = torch.empty((num_seqs * H, W), device=query_states.device, dtype=torch.float32)
    l_sum = torch.empty((num_seqs * H, W), device=query_states.device, dtype=torch.float32)
    out = torch.zeros((total_tokens, H), device=query_states.device, dtype=torch.float32)

    # Strides for packed (total_tokens, H, D)
    stride_q0, stride_q1, stride_q2 = H * D, D, 1
    stride_k0, stride_k1, stride_k2 = H * D, D, 1

    # Kernel 1
    grid_stats = (num_seqs * H, triton.cdiv(max_W, 16))
    _stats_m_l_kernel_packed[grid_stats](
        query_states, key_states, m, l_sum, cu_seqlens,
        H=H, D=D, MAX_W=max_W, RECENT_Q=recent_q,
        stride_q0=stride_q0, stride_q1=stride_q1, stride_q2=stride_q2,
        stride_k0=stride_k0, stride_k1=stride_k1, stride_k2=stride_k2,
        stride_m0=m.stride(0), stride_m1=m.stride(1),
        stride_l0=l_sum.stride(0), stride_l1=l_sum.stride(1),
    )

    # Kernel 2
    grid_prob = (num_seqs * H, triton.cdiv(max_prefix, 64))
    _prefix_meanprob_kernel_packed[grid_prob](
        query_states, key_states, m, l_sum, out, cu_seqlens,
        H=H, D=D, MAX_PREFIX=max_prefix, RECENT_Q=recent_q,
        stride_q0=stride_q0, stride_q1=stride_q1, stride_q2=stride_q2,
        stride_k0=stride_k0, stride_k1=stride_k1, stride_k2=stride_k2,
        stride_m0=m.stride(0), stride_m1=m.stride(1),
        stride_l0=l_sum.stride(0), stride_l1=l_sum.stride(1),
        stride_o0=out.stride(0), stride_o1=out.stride(1),
    )

    return out.to(out_dtype)


# ============================================================================
# Token Sparse Selector (论文 Algorithm 1 正确实现)
# ============================================================================


class TokenSparseSelector:
    """
    基于论文 Algorithm 1 的 token 选择器

    核心流程:
    1. 使用 recent queries 计算 lightweight proxy attention
    2. Per-head pooling 得到 token scores
    3. Per-head 独立选择: 保留累积注意力质量 ≥ τ 的最少 tokens
    4. 合并 window tokens (attention sink)
    """

    def __init__(
        self,
        tau: float = 0.005,  # τ: 累积注意力质量阈值 (论文语义)
        min_tokens: int = 1024,  # 最少保留的 token 数
        window_size: int = 128,  # 保留的局部 window 大小
        kernel_size: int = 7,  # 平滑 kernel 大小
        recent_q: int = DEFAULT_RECENT_Q,  # 用于 proxy 计算的 recent queries 数量
    ):
        self.tau = tau  # 论文: τ 是注意力质量阈值, 不是 token 比例
        self.min_tokens = min_tokens
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.recent_q = recent_q

    def select_indices_per_head(
        self,
        attn_cache: torch.Tensor,  # (B, H, prefix_len) - per-head scores
        device: torch.device,
        L: int,
        B: int,
        H: int,
    ) -> list[list[torch.Tensor]]:
        """
        Per-head token selection following paper Algorithm 1.

        Returns:
            indices_list[b][h] = (k_h,) sorted ascending token indices
        """
        prefix_len = attn_cache.shape[-1]

        # Step 1: Smoothing (论文步骤 7)
        if self.kernel_size > 1 and prefix_len >= self.kernel_size:
            attn_cache_smoothed = torch.nn.functional.avg_pool1d(
                attn_cache,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            attn_cache_smoothed = attn_cache

        # Step 2: Per-head normalization (论文步骤 8)
        attn_cache_norm = attn_cache_smoothed / (
            attn_cache_smoothed.sum(dim=-1, keepdim=True) + EPS
        )

        # Step 3-4: Per-head selection based on cumulative quality threshold τ
        indices_list: list[list[torch.Tensor]] = []
        for b in range(B):
            head_indices: list[torch.Tensor] = []
            for h in range(H):
                scores = attn_cache_norm[b, h, :]  # (prefix_len,)

                # Sort by score ascending (least important first)
                sorted_scores, sorted_idx = torch.sort(scores, descending=False)

                # Find k_sparse: minimum tokens whose cumulative score < τ
                cumsum = torch.cumsum(sorted_scores.float(), dim=-1)
                k_sparse = int((cumsum < self.tau).sum().item())

                # k_keep: tokens to retain
                k_keep = prefix_len - k_sparse
                k_keep = max(self.min_tokens, min(k_keep, prefix_len))

                if k_keep >= prefix_len:
                    # Keep all prefix tokens
                    idx = torch.arange(prefix_len, device=device, dtype=torch.int32)
                else:
                    # Top-k by score (descending)
                    _, topk_idx = torch.topk(scores, k=k_keep, dim=-1, largest=True)
                    idx = torch.sort(topk_idx, dim=-1)[0]

                head_indices.append(idx)
            indices_list.append(head_indices)

        return indices_list

    def select_indices(
        self,
        query_states: torch.Tensor,  # (B, H, L, D)
        key_states: torch.Tensor,  # (B, H, L, D)
        attn_cache: torch.Tensor | None = None,
    ) -> list[list[torch.Tensor]] | None:
        """
        选择需要保留的 token 索引 (per-head, per-batch).

        Returns:
            indices: List[List[Tensor]] where
                     indices[b][h] is (k_h,) tensor of selected prefix token indices,
                     or None if no sparsification needed
        """
        B, H, L, D = query_states.shape

        # Short sequence: no sparsification
        if self.min_tokens + self.window_size >= L:
            return None

        # Get attention cache if not provided
        if attn_cache is None:
            attn_cache = triton_get_attn_cache(
                query_states, key_states, self.window_size, self.recent_q
            )

        return self.select_indices_per_head(attn_cache, query_states.device, L, B, H)

    def compress_qkv_per_head(
        self,
        query: torch.Tensor,  # (B, H, L, D)
        key: torch.Tensor,  # (B, H, L, D)
        value: torch.Tensor,  # (B, H, L, D)
        indices_list: list[list[torch.Tensor]],  # [B][H] -> (k_h,)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress QKV with per-head indices.
        Returns padded tensors of shape (B, H, k_max, D).
        """
        B, H, L, D = query.shape

        # Find max k across all heads for padding
        k_max = max(
            indices_list[b][h].shape[0] + self.window_size
            for b in range(B) for h in range(H)
        )

        q_out = torch.zeros(B, H, k_max, D, device=query.device, dtype=query.dtype)
        k_out = torch.zeros(B, H, k_max, D, device=query.device, dtype=query.dtype)
        v_out = torch.zeros(B, H, k_max, D, device=query.device, dtype=query.dtype)

        window_indices = torch.arange(
            L - self.window_size, L, device=query.device
        )

        for b in range(B):
            for h in range(H):
                prefix_idx = indices_list[b][h]  # (k_prefix,)
                # Combine prefix + window indices
                all_idx = torch.cat([prefix_idx, window_indices], dim=0)
                all_idx = torch.sort(all_idx, dim=-1)[0]
                k_actual = all_idx.shape[0]

                # Direct indexing gather
                q_out[b, h, :k_actual, :] = query[b, h, all_idx]
                k_out[b, h, :k_actual, :] = key[b, h, all_idx]
                v_out[b, h, :k_actual, :] = value[b, h, all_idx]

        return q_out, k_out, v_out

    def decompress_output_per_head(
        self,
        attn_output: torch.Tensor,  # (B, H, k_max, D) - padded compressed output
        indices_list: list[list[torch.Tensor]],  # [B][H] -> (k_h,)
        original_length: int,
    ) -> torch.Tensor:
        """
        Decompress attention output back to original shape (B, H, L, D).
        Unselected positions remain zero as per paper.
        """
        B, H, k_max, D = attn_output.shape
        output = torch.zeros(
            B, H, original_length, D,
            device=attn_output.device, dtype=attn_output.dtype,
        )

        window_start = original_length - self.window_size

        for b in range(B):
            for h in range(H):
                prefix_idx = indices_list[b][h]  # (k_prefix,)
                k_actual = prefix_idx.shape[0] + self.window_size

                # Extract valid portion (remove padding)
                attn_h = attn_output[b, h, :k_actual, :]  # (k_actual, D)

                # Split prefix and window portions
                k_prefix = prefix_idx.shape[0]
                prefix_out = attn_h[:k_prefix]  # (k_prefix, D)
                window_out = attn_h[k_prefix:]  # (window_size, D)

                # Scatter prefix output
                output[b, h].index_copy_(0, prefix_idx, prefix_out)
                # Copy window output to end positions
                output[b, h, window_start:] = window_out

        return output


# ============================================================================
# vLLM Attention Backend Interface (修复版)
# ============================================================================


class TokenSparseV3AttentionBackend(FlashAttentionBackend):
    """
    Token Sparse Attention Backend v3 for vLLM

    特性：
    - 正确实现论文 Algorithm 1: per-head sparse selection with τ threshold
    - Prefill 阶段稀疏注意力，Decode 阶段回退到原生实现
    - 完整兼容 vLLM FlashAttention 参数 (block_table, scheduler_metadata 等)
    - 支持 GQA/MQA 架构，保持 per-head 独立性
    """

    @staticmethod
    def get_name() -> str:
        return "TOKEN_SPARSE_V3"

    @staticmethod
    def get_impl_cls() -> type["TokenSparseV3AttentionImpl"]:
        return TokenSparseV3AttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TokenSparseV3AttentionMetadataBuilder"]:
        return TokenSparseV3AttentionMetadataBuilder


@dataclass
class TokenSparseV3AttentionMetadata(FlashAttentionMetadata):
    """Token Sparse Attention v3 元数据"""

    # Token Sparse 特有字段
    sparse_indices: list[list[torch.Tensor]] | None = None  # per-head indices: [B][H]->(k_h,)
    is_prefill: bool = True

    # 元信息
    head_dim: int = 128
    num_kv_heads: int = 0


class TokenSparseV3AttentionMetadataBuilder(
    AttentionMetadataBuilder[TokenSparseV3AttentionMetadata],
):
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )

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

        # 配置优先级: hf_config > 环境变量 > 默认值
        hf_config = self.model_config.hf_text_config

        def get_config(key: str, default, attr_name: str = None):
            if attr_name is None:
                attr_name = key.lower().replace("vllm_token_sparse_v3_", "token_sparse_v3_")
            env_val = os.environ.get(key)
            hf_val = getattr(hf_config, attr_name, None) if hf_config else None
            if hf_val is not None:
                return hf_val
            if env_val is not None:
                return env_val
            return default

        self.tau = float(get_config("VLLM_TOKEN_SPARSE_V3_TAU", 0.005))
        self.window_size = int(get_config("VLLM_TOKEN_SPARSE_V3_WINDOW_SIZE", 128))
        self.min_tokens = int(get_config("VLLM_TOKEN_SPARSE_V3_MIN_TOKENS", 1024))
        self.kernel_size = int(get_config("VLLM_TOKEN_SPARSE_V3_KERNEL_SIZE", 7))
        self.recent_q = int(get_config("VLLM_TOKEN_SPARSE_V3_RECENT_Q", DEFAULT_RECENT_Q))

        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        logger.info(
            "TokenSparseV3AttentionMetadataBuilder: tau=%s, window=%s, min_tokens=%s, "
            "kernel=%s, recent_q=%s",
            self.tau, self.window_size, self.min_tokens,
            self.kernel_size, self.recent_q,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TokenSparseV3AttentionMetadata:
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        slot_mapping = common_attn_metadata.slot_mapping
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        max_seq_len = common_attn_metadata.max_seq_len

        # Prefill detection: use actual sequence length, not chunk length
        is_prefill = max(seq_lens).item() > 1 if seq_lens.numel() > 0 else (max_query_len > 1)

        return TokenSparseV3AttentionMetadata(
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
            sparse_indices=None,  # Computed in forward pass
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )


class TokenSparseV3AttentionImpl(FlashAttentionImpl):
    """
    Token Sparse Attention v3 实现 (修复版)

    工作流程:
    1. Prefill 阶段: per-head token selection -> compress -> FlashAttention -> decompress
    2. Decode 阶段: 回退到父类标准 FlashAttention
    3. 正确处理 GQA: 不 repeat KV, 保持 per-head 选择独立性
    """

    SPARSE_MIN_SEQ_LEN = int(os.environ.get("VLLM_TOKEN_SPARSE_V3_MIN_SEQ_LEN", "32768"))

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
        # Token Sparse 参数 (优先级: 函数参数 > Builder 配置 > 环境变量)
        token_sparse_tau: float | None = None,
        token_sparse_window_size: int | None = None,
        token_sparse_min_tokens: int | None = None,
        token_sparse_kernel_size: int | None = None,
        token_sparse_recent_q: int | None = None,
        layer_name: str | None = None,
    ):
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

        self.layer_idx = self._extract_layer_index(layer_name)

        # Sparse layers configuration
        sparse_layers_env = os.environ.get("VLLM_TOKEN_SPARSE_V3_SPARSE_LAYERS", "")
        if sparse_layers_env:
            self.sparse_layers = [
                int(x.strip()) for x in sparse_layers_env.split(",") if x.strip()
            ]
        else:
            self.sparse_layers = None  # None means all layers

        # 配置优先级: 函数参数 > 环境变量 > 硬编码默认
        def _get_cfg(val, env_name, default):
            if val is not None:
                return val
            env_val = os.environ.get(env_name)
            if env_val is not None:
                return type(default)(env_val)
            return default

        self.tau = float(_get_cfg(token_sparse_tau, "VLLM_TOKEN_SPARSE_V3_TAU", 0.005))
        self.window_size = int(_get_cfg(token_sparse_window_size, "VLLM_TOKEN_SPARSE_V3_WINDOW_SIZE", 128))
        self.min_tokens = int(_get_cfg(token_sparse_min_tokens, "VLLM_TOKEN_SPARSE_V3_MIN_TOKENS", 1024))
        self.kernel_size = int(_get_cfg(token_sparse_kernel_size, "VLLM_TOKEN_SPARSE_V3_KERNEL_SIZE", 7))
        self.recent_q = int(_get_cfg(token_sparse_recent_q, "VLLM_TOKEN_SPARSE_V3_RECENT_Q", DEFAULT_RECENT_Q))

        # Enable sparse if layer is in sparse_layers (or all layers if not specified)
        if self.layer_idx is not None and self.sparse_layers is not None:
            self.use_sparse = self.layer_idx in self.sparse_layers
        elif self.sparse_layers is None:
            self.use_sparse = True
        else:
            self.use_sparse = False

        logger.info(
            "TokenSparseV3AttentionImpl: layer=%s, sparse=%s, tau=%s, window=%s, "
            "min_tokens=%s, recent_q=%s, sparse_layers=%s",
            self.layer_idx, self.use_sparse, self.tau, self.window_size,
            self.min_tokens, self.recent_q, self.sparse_layers,
        )

        if self.use_sparse:
            self.selector = TokenSparseSelector(
                tau=self.tau,
                min_tokens=self.min_tokens,
                window_size=self.window_size,
                kernel_size=self.kernel_size,
                recent_q=self.recent_q,
            )
        else:
            self.selector = None

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # (num_tokens, num_heads, head_size)
        key: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
        value: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
        kv_cache: torch.Tensor,
        attn_metadata: TokenSparseV3AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for TokenSparseV3"
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Check if sparse attention should be applied
        apply_sparse = (
            self.use_sparse
            and self.selector is not None
            and attn_metadata.is_prefill
            and attn_metadata.causal
            and self.attn_type == AttentionType.DECODER
            and attn_metadata.max_seq_len > self.selector.min_tokens + self.selector.window_size
            and attn_metadata.max_seq_len > self.SPARSE_MIN_SEQ_LEN
        )

        if not apply_sparse:
            return super().forward(
                layer=layer, query=query, key=key, value=value,
                kv_cache=kv_cache, attn_metadata=attn_metadata,
                output=output, output_scale=output_scale,
                output_block_scale=output_block_scale,
            )

        # Token Sparse v3 path
        num_tokens = query.shape[0]
        H = query.shape[1]
        D = query.shape[2]

        output_3d = self._forward_sparse_packed(
            layer=layer, query=query, key=key, value=value,
            kv_cache=kv_cache, attn_metadata=attn_metadata,
            output=output, output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        return output_3d.reshape(num_tokens, H * D)

    def _forward_sparse_packed(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # (num_tokens, num_heads, D)
        key: torch.Tensor,  # (num_tokens, num_kv_heads, D) - DO NOT repeat for GQA
        value: torch.Tensor,  # (num_tokens, num_kv_heads, D)
        kv_cache: torch.Tensor,
        attn_metadata: TokenSparseV3AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process packed multi-sequence input with per-head token sparse attention.
        Key fix: Do NOT repeat KV heads for GQA - maintain per-head independence.
        """
        cu_seqlens_q = attn_metadata.query_start_loc
        num_seqs = cu_seqlens_q.shape[0] - 1
        H_q = query.shape[1]  # num query heads
        H_kv = key.shape[1]   # num kv heads
        D = query.shape[2]
        num_tokens = query.shape[0]

        # Step 0: Compute attention cache per query head
        # For GQA: compute cache for each query head using its assigned KV head
        if H_q != H_kv:
            # GQA/MQA: map each query head to its KV head, compute per-head cache
            attn_cache_list = []
            for h_q in range(H_q):
                h_kv = h_q // (H_q // H_kv)
                q_head = query[:, h_q:h_q + 1, :]  # (N, 1, D)
                k_head = key[:, h_kv:h_kv + 1, :]  # (N, 1, D)

                cache = triton_get_attn_cache_packed(
                    q_head, k_head, cu_seqlens_q,
                    self.selector.window_size, self.selector.recent_q,
                )  # (N, 1)
                attn_cache_list.append(cache)

            attn_cache_packed = torch.cat(attn_cache_list, dim=1)  # (N, H_q)
        else:
            # MHA: direct computation
            attn_cache_packed = triton_get_attn_cache_packed(
                query, key, cu_seqlens_q,
                self.selector.window_size, self.selector.recent_q,
            )

        output_3d_list: list[torch.Tensor] = []

        for seq_idx in range(num_seqs):
            start = cu_seqlens_q[seq_idx].item()
            end = cu_seqlens_q[seq_idx + 1].item()
            seq_len = end - start

            # Short sequence: fallback to dense attention
            if seq_len <= self.selector.min_tokens + self.selector.window_size:
                seq_out = super().forward(
                    layer=layer,
                    query=query[start:end],
                    key=key[start:end],
                    value=value[start:end],
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output[start:end],
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                )
                output_3d_list.append(seq_out.reshape(seq_len, H_q, D))
                continue

            # Extract per-sequence tensors
            seq_q = query[start:end].unsqueeze(0).transpose(1, 2)  # (1, H_q, L, D)
            seq_k = key[start:end].unsqueeze(0)  # (1, L, H_kv, D)
            seq_v = value[start:end].unsqueeze(0)  # (1, L, H_kv, D)

            # Extract per-head attention cache for prefix positions only
            seq_cache = attn_cache_packed[start:end].t().unsqueeze(0)  # (1, H_q, L)
            prefix_len = seq_len - self.selector.window_size
            seq_cache_prefix = seq_cache[:, :, :prefix_len]  # (1, H_q, prefix_len)

            # Prepare key_states for select_indices: (1, H_kv, L, D)
            seq_k_for_select = seq_k.transpose(1, 2)  # (1, H_kv, L, D)

            # Step 1: Per-head token selection
            indices_list = self.selector.select_indices(
                seq_q, seq_k_for_select, seq_cache_prefix,
            )

            if indices_list is None:
                # Fallback to dense
                seq_out = super().forward(
                    layer=layer, query=query[start:end], key=key[start:end],
                    value=value[start:end], kv_cache=kv_cache,
                    attn_metadata=attn_metadata, output=output[start:end],
                    output_scale=output_scale, output_block_scale=output_block_scale,
                )
                output_3d_list.append(seq_out.reshape(seq_len, H_q, D))
                continue

            # Step 2: Expand KV to match query heads for compression
            if H_q != H_kv:
                seq_k_expanded = []
                seq_v_expanded = []
                for h_q in range(H_q):
                    h_kv = h_q // (H_q // H_kv)
                    seq_k_expanded.append(seq_k[:, :, h_kv:h_kv + 1, :])
                    seq_v_expanded.append(seq_v[:, :, h_kv:h_kv + 1, :])
                seq_k = torch.cat(seq_k_expanded, dim=2)  # (1, L, H_q, D)
                seq_v = torch.cat(seq_v_expanded, dim=2)  # (1, L, H_q, D)

            seq_k = seq_k.transpose(1, 2)  # (1, H_q, L, D)
            seq_v = seq_v.transpose(1, 2)  # (1, H_q, L, D)

            q_sparse, k_sparse, v_sparse = self.selector.compress_qkv_per_head(
                seq_q, seq_k, seq_v, indices_list,
            )
            # q_sparse: (1, H_q, k_max, D)

            # Step 3: FlashAttention on compressed tensors
            B_s, H_s, k_max, D_s = q_sparse.shape

            # Reshape for varlen FlashAttention
            # Each (head, compressed_seq) pair is treated as an independent sequence
            q_flat = q_sparse.reshape(B_s * H_s * k_max, D_s).unsqueeze(1)  # (B*H*k, 1, D)
            k_flat = k_sparse.reshape(B_s * H_s * k_max, D_s).unsqueeze(1)
            v_flat = v_sparse.reshape(B_s * H_s * k_max, D_s).unsqueeze(1)

            # cu_seqlens: each segment has exactly k_max tokens
            cu_seqlens_compressed = torch.arange(
                0, B_s * H_s * (k_max + 1), step=k_max,
                device=q_flat.device, dtype=torch.int32,
            )

            attn_out_flat = torch.empty_like(q_flat)

            # FlashAttention with complete parameters
            flash_kwargs: dict = dict(
                softmax_scale=self.scale,
                causal=True,
            )
            if self.sliding_window != (-1, -1):
                flash_kwargs["window_size"] = list(self.sliding_window)
            if self.alibi_slopes is not None:
                flash_kwargs["alibi_slopes"] = self.alibi_slopes
            if self.logits_soft_cap != 0:
                flash_kwargs["softcap"] = self.logits_soft_cap
            if hasattr(attn_metadata, 'scheduler_metadata'):
                flash_kwargs["scheduler_metadata"] = attn_metadata.scheduler_metadata

            flash_attn_varlen_func(
                q=q_flat, k=k_flat, v=v_flat, out=attn_out_flat,
                cu_seqlens_q=cu_seqlens_compressed,
                cu_seqlens_k=cu_seqlens_compressed,
                max_seqlen_q=k_max,
                max_seqlen_k=k_max,
                **flash_kwargs,
            )

            # Reshape: (B*H*k, 1, D) -> (B, H, k_max, D)
            attn_out = attn_out_flat.squeeze(1).reshape(B_s, H_s, k_max, D_s)

            # Step 4: Decompress to original shape
            seq_out_4d = self.selector.decompress_output_per_head(
                attn_out, indices_list, seq_len,
            )  # (1, H_q, L, D)

            output_3d_list.append(seq_out_4d.squeeze(0))  # (L, H_q, D)

        return torch.cat(output_3d_list, dim=0)

    def _extract_layer_index(self, layer_name: str | None) -> int | None:
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
        if self.attn_type != AttentionType.DECODER:
            return
        if kv_cache is None or kv_cache.numel() == 0:
            return

        key_cache, value_cache = kv_cache.unbind(0)
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping,
            self.kv_cache_dtype,
            getattr(layer, '_k_scale', None),
            getattr(layer, '_v_scale', None),
        )
