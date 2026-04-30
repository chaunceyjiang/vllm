# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token Sparse Attention Backend v2 for vLLM V1.

Combines the best of both reference implementations:

From downloads/tsa/files (tsa/):
- Fast probe-based token scoring (no full softmax overhead)
- Triton gather/scatter kernels
- Single batched FlashAttention call for all sequences
- Clean config dataclass with dynamic token coverage (tau)

From vllm/v1/attention/backends/token_sparse.py:
- Full vLLM V1 Backend integration (inherits FlashAttentionBackend)
- Environment variable + hf_config configuration
- Layer-level sparse control (sparse_layers)
- Proper prefill/decode distinction
- Window-size attention sink (always keep recent tokens)
- GQA-aware scoring

Author: vLLM Token Sparse Integration
Version: 0.2.0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F

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
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# ============================================================================
# Triton availability check
# ============================================================================

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# ============================================================================
# Triton Kernel 1: Token Scoring
# ============================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _token_score_kernel(
        Q_ptr,
        K_ptr,
        out_ptr,
        seq_len,
        num_heads,
        head_dim,
        num_probe,
        scale,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute per-token importance as mean dot-product across probes and heads."""
        pid = tl.program_id(0)
        n_off = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_off < seq_len

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        stride_seq = num_heads * head_dim
        stride_head = head_dim

        for h in range(num_heads):
            for p in range(num_probe):
                q_base = (p * num_heads + h) * head_dim
                k_base = n_off * stride_seq + h * stride_head

                dot = tl.zeros([BLOCK_N], dtype=tl.float32)
                for d in range(0, head_dim, BLOCK_D):
                    d_off = d + tl.arange(0, BLOCK_D)
                    d_mask = d_off < head_dim

                    q_tile = tl.load(
                        Q_ptr + q_base + d_off, mask=d_mask, other=0.0
                    ).to(tl.float32)
                    k_ptrs = K_ptr + k_base[:, None] + d_off[None, :]
                    k_tile = tl.load(
                        k_ptrs,
                        mask=n_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    dot += tl.sum(k_tile * q_tile[None, :], axis=1)

                acc += dot

        acc = acc * (scale / (num_heads * num_probe))
        tl.store(out_ptr + n_off, acc, mask=n_mask)

    # -------------------------------------------------------------------------
    # Triton Kernel 2: Gather
    # -------------------------------------------------------------------------

    @triton.jit
    def _gather_kernel(
        src_ptr,
        idx_ptr,
        dst_ptr,
        n_keep,
        stride_seq,
        total_cols,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        """dst[i, :] = src[idx[i], :] for i in 0 .. n_keep-1."""
        row_pid = tl.program_id(0)
        col_pid = tl.program_id(1)

        row_off = row_pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
        col_off = col_pid * BLOCK_COL + tl.arange(0, BLOCK_COL)
        row_mask = row_off < n_keep
        col_mask = col_off < total_cols

        src_rows = tl.load(idx_ptr + row_off, mask=row_mask, other=0)
        src_ptrs = src_ptr + src_rows[:, None] * stride_seq + col_off[None, :]
        vals = tl.load(
            src_ptrs,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        dst_ptrs = dst_ptr + row_off[:, None] * stride_seq + col_off[None, :]
        tl.store(dst_ptrs, vals, mask=row_mask[:, None] & col_mask[None, :])

    # -------------------------------------------------------------------------
    # Triton Kernel 3: Scatter
    # -------------------------------------------------------------------------

    @triton.jit
    def _scatter_kernel(
        src_ptr,
        idx_ptr,
        dst_ptr,
        n_keep,
        stride_seq,
        total_cols,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        """dst[idx[i], :] = src[i, :] for i in 0 .. n_keep-1."""
        row_pid = tl.program_id(0)
        col_pid = tl.program_id(1)

        row_off = row_pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
        col_off = col_pid * BLOCK_COL + tl.arange(0, BLOCK_COL)
        row_mask = row_off < n_keep
        col_mask = col_off < total_cols

        dst_rows = tl.load(idx_ptr + row_off, mask=row_mask, other=0)
        src_ptrs = src_ptr + row_off[:, None] * stride_seq + col_off[None, :]
        vals = tl.load(
            src_ptrs,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        dst_ptrs = dst_ptr + dst_rows[:, None] * stride_seq + col_off[None, :]
        tl.store(dst_ptrs, vals, mask=row_mask[:, None] & col_mask[None, :])


# ============================================================================
# Python wrappers for Triton kernels (with PyTorch fallbacks)
# ============================================================================

def _token_score_triton(
    probe_q: torch.Tensor,  # [num_probe, num_heads, head_dim]
    key: torch.Tensor,  # [seq_len, num_heads, head_dim]
    scale: float,
) -> torch.Tensor:
    """Launch Triton token scoring kernel."""
    assert probe_q.is_contiguous() and key.is_contiguous()
    num_probe, num_heads, head_dim = probe_q.shape
    seq_len = key.shape[0]

    out = torch.empty(seq_len, dtype=torch.float32, device=key.device)
    BLOCK_N = 64
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(seq_len, BLOCK_N),)

    _token_score_kernel[grid](
        probe_q,
        key,
        out,
        seq_len,
        num_heads,
        head_dim,
        num_probe,
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return out


def _token_score_pytorch(
    probe_q: torch.Tensor,  # [num_probe, num_heads, head_dim]
    key: torch.Tensor,  # [seq_len, num_heads, head_dim]
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch token scorer fallback."""
    pq = probe_q.to(torch.float32).permute(1, 0, 2)  # [H, P, D]
    k = key.to(torch.float32).permute(1, 0, 2)  # [H, N, D]
    logits = torch.bmm(pq, k.transpose(1, 2)) * scale  # [H, P, N]
    weights = F.softmax(logits, dim=-1)
    scores = weights.mean(dim=(0, 1))  # [N]
    return scores


def _score_tokens(
    probe_q: torch.Tensor,  # [num_probe, num_heads_q, head_dim]
    key: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    scale: float,
    use_triton: bool = True,
) -> torch.Tensor:
    """Unified token scoring with GQA support.

    For GQA, averages query heads within each KV group before scoring.
    """
    num_probe, num_heads_q, head_dim = probe_q.shape
    num_kv_heads = key.shape[1]

    logger.info(
        "[TokenSparseV2] _score_tokens: probe_q=%s, key=%s, scale=%.4f, "
        "use_triton=%s, triton_available=%s",
        tuple(probe_q.shape), tuple(key.shape), scale,
        use_triton, _TRITON_AVAILABLE,
    )

    if num_heads_q > num_kv_heads:
        # GQA: average query heads per KV group
        num_queries_per_kv = num_heads_q // num_kv_heads
        logger.info(
            "[TokenSparseV2] GQA detected: num_heads_q=%d, num_kv_heads=%d, "
            "num_queries_per_kv=%d",
            num_heads_q, num_kv_heads, num_queries_per_kv,
        )
        probe_q = probe_q.reshape(
            num_probe, num_kv_heads, num_queries_per_kv, head_dim
        )
        probe_q = probe_q.mean(dim=2)  # [num_probe, num_kv_heads, head_dim]

    if (
        use_triton
        and _TRITON_AVAILABLE
        and probe_q.is_contiguous()
        and key.is_contiguous()
    ):
        logger.info("[TokenSparseV2] using Triton scoring path")
        return _token_score_triton(probe_q, key, scale)
    else:
        if use_triton and not _TRITON_AVAILABLE:
            logger.info("[TokenSparseV2] Triton not available, falling back to PyTorch scoring")
        elif use_triton and not probe_q.is_contiguous():
            logger.info("[TokenSparseV2] probe_q not contiguous, falling back to PyTorch scoring")
        elif use_triton and not key.is_contiguous():
            logger.info("[TokenSparseV2] key not contiguous, falling back to PyTorch scoring")
        else:
            logger.info("[TokenSparseV2] using PyTorch scoring path")
        return _token_score_pytorch(probe_q, key, scale)


def _gather_tokens(
    src: torch.Tensor,  # [seq_len, num_heads, head_dim]
    idx: torch.Tensor,  # [n_keep] int32
) -> torch.Tensor:
    """Gather selected token rows. Uses Triton if available."""
    seq_len, num_heads, head_dim = src.shape
    n_keep = idx.shape[0]

    if n_keep == 0:
        logger.warning("[TokenSparseV2] _gather_tokens: n_keep=0, returning empty tensor")
        return torch.empty((0, num_heads, head_dim), dtype=src.dtype, device=src.device)

    # Range check
    if idx.numel() > 0:
        min_idx = int(idx.min().item())
        max_idx = int(idx.max().item())
        if min_idx < 0 or max_idx >= seq_len:
            logger.error(
                "[TokenSparseV2] _gather_tokens: idx out of range! "
                "min=%d, max=%d, seq_len=%d",
                min_idx, max_idx, seq_len,
            )

    if not _TRITON_AVAILABLE:
        logger.info("[TokenSparseV2] _gather_tokens: Triton not available, using PyTorch indexing")
        return src[idx]

    stride = num_heads * head_dim
    total_col = stride

    dst = torch.empty((n_keep, num_heads, head_dim), dtype=src.dtype, device=src.device)
    BLOCK_ROW = 32
    BLOCK_COL = min(128, triton.next_power_of_2(total_col))
    grid = (triton.cdiv(n_keep, BLOCK_ROW), triton.cdiv(total_col, BLOCK_COL))

    logger.info(
        "[TokenSparseV2] _gather_tokens Triton: src=%s, idx=%s, dst=%s, grid=%s",
        tuple(src.shape), tuple(idx.shape), tuple(dst.shape), grid,
    )

    _gather_kernel[grid](
        src,
        idx,
        dst,
        n_keep,
        stride,
        total_col,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )
    return dst


def _scatter_tokens(
    src: torch.Tensor,  # [n_keep, num_heads, head_dim]
    idx: torch.Tensor,  # [n_keep] int32
    seq_len: int,
) -> torch.Tensor:
    """Scatter sparse output back to full sequence. Uses Triton if available."""
    n_keep, num_heads, head_dim = src.shape
    stride = num_heads * head_dim
    total_col = stride

    # Range check
    if idx.numel() > 0:
        min_idx = int(idx.min().item())
        max_idx = int(idx.max().item())
        if min_idx < 0 or max_idx >= seq_len:
            logger.error(
                "[TokenSparseV2] _scatter_tokens: idx out of range! "
                "min=%d, max=%d, seq_len=%d",
                min_idx, max_idx, seq_len,
            )

    dst = torch.zeros(
        (seq_len, num_heads, head_dim), dtype=src.dtype, device=src.device
    )
    if n_keep == 0:
        logger.info("[TokenSparseV2] _scatter_tokens: n_keep=0, returning zero tensor")
        return dst

    if not _TRITON_AVAILABLE:
        logger.info("[TokenSparseV2] _scatter_tokens: Triton not available, using PyTorch indexing")
        dst[idx] = src
        return dst

    BLOCK_ROW = 32
    BLOCK_COL = min(128, triton.next_power_of_2(total_col))
    grid = (triton.cdiv(n_keep, BLOCK_ROW), triton.cdiv(total_col, BLOCK_COL))

    logger.info(
        "[TokenSparseV2] _scatter_tokens Triton: src=%s, idx=%s, dst_shape=%s, grid=%s",
        tuple(src.shape), tuple(idx.shape), tuple(dst.shape), grid,
    )

    _scatter_kernel[grid](
        src,
        idx,
        dst,
        n_keep,
        stride,
        total_col,
        BLOCK_ROW=BLOCK_ROW,
        BLOCK_COL=BLOCK_COL,
    )
    return dst


# ============================================================================
# Dynamic Token Coverage
# ============================================================================

def _dynamic_token_coverage(
    scores: torch.Tensor,  # [seq_len] float32 importance scores
    tau: float,
    min_keep_ratio: float = 0.10,
    window_size: int = 0,
) -> torch.Tensor:
    """Select tokens to keep such that dropped mass <= tau.

    Algorithm:
        1. Sort scores ascending (least important first).
        2. Compute prefix cumulative sum.
        3. Find largest k where cum_sum[k-1] <= tau -> drop k tokens.
        4. Apply min_keep_ratio safety floor.
        5. Always keep the last `window_size` tokens (attention sink).

    Returns:
        keep_idx: int32 tensor [n_keep], sorted ascending.
    """
    N = scores.shape[0]
    device = scores.device

    # Sort ascending: sorted_scores[0] is least important
    sorted_scores, sorted_idx = torch.sort(scores, descending=False)
    cum = torch.cumsum(sorted_scores, dim=0)
    n_drop = int((cum <= tau).sum().item())

    # Safety floor
    max_drop = int(N * (1.0 - min_keep_ratio))
    n_drop = min(n_drop, max_drop)

    # Window tokens are always kept
    window_start = max(0, N - window_size)

    logger.info(
        "[TokenSparseV2] _dynamic_token_coverage: N=%d, tau=%.6f, "
        "n_drop(raw)=%d, max_drop=%d, n_drop(final)=%d, window_start=%d",
        N, tau, int((cum <= tau).sum().item()), max_drop, n_drop, window_start,
    )

    if n_drop == 0:
        return torch.arange(N, device=device, dtype=torch.int32)

    # Mark dropped tokens (but not window tokens)
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)
    drop_candidates = sorted_idx[:n_drop]
    # Don't drop window tokens
    valid_drop = drop_candidates < window_start
    dropped_window_tokens = (drop_candidates >= window_start).sum().item()
    if dropped_window_tokens > 0:
        logger.info(
            "[TokenSparseV2] _dynamic_token_coverage: protected %d window tokens "
            "from being dropped",
            dropped_window_tokens,
        )
    keep_mask[drop_candidates[valid_drop]] = False

    keep_idx = keep_mask.nonzero(as_tuple=True)[0].to(torch.int32)
    logger.info(
        "[TokenSparseV2] _dynamic_token_coverage: keep %d/%d tokens "
        "(drop_rate=%.2f%%)",
        keep_idx.shape[0], N, (N - keep_idx.shape[0]) / N * 100,
    )
    return keep_idx


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TokenSparseV2Config:
    """Configuration for Token Sparse Attention v2."""

    # Dynamic token coverage threshold.
    # Larger tau -> more aggressive sparsity. Paper default: 0.005.
    tau: float = 0.005

    # Only apply sparse attention when sequence length >= this.
    min_seq_len: int = 1024

    # Safety floor: always keep at least this fraction of tokens.
    min_keep_ratio: float = 0.10

    # Number of recent query vectors used as probes for scoring.
    num_probe_queries: int = 16

    # Always keep the last `window_size` tokens (attention sink).
    window_size: int = 128

    # Use Triton kernels (True) or pure PyTorch (False).
    use_triton: bool = True

    # Log sparsity stats per forward call.
    log_stats: bool = False

    # Layer indices to apply sparse attention. None = all layers.
    apply_to_layers: Optional[List[int]] = None

    def should_apply(self, layer_idx: int) -> bool:
        if self.apply_to_layers is None:
            return True
        return layer_idx in self.apply_to_layers


# ============================================================================
# Token Selector
# ============================================================================

class TokenSparseV2Selector:
    """Token selector using probe-based scoring + dynamic coverage."""

    def __init__(self, config: TokenSparseV2Config):
        self.config = config

    def select_indices(
        self,
        query: torch.Tensor,  # [seq_len, num_heads_q, head_dim]
        key: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
        scale: float,
    ) -> torch.Tensor | None:
        """Select token indices to keep. Returns None if no sparsity needed."""
        seq_len = query.shape[0]
        cfg = self.config

        # Short sequence: keep all
        if seq_len < cfg.min_seq_len:
            return None

        num_probe = min(cfg.num_probe_queries, seq_len)
        probe_q = query[-num_probe:].contiguous()  # [num_probe, H, D]

        scores = _score_tokens(probe_q, key, scale, use_triton=cfg.use_triton)

        # NaN/Inf guard on scores
        if torch.isnan(scores).any():
            logger.error(
                "[TokenSparseV2] scores contain NaN! probe_q=%s, key=%s, scale=%s",
                tuple(probe_q.shape), tuple(key.shape), scale,
            )
        if torch.isinf(scores).any():
            logger.error(
                "[TokenSparseV2] scores contain Inf! probe_q=%s, key=%s, scale=%s",
                tuple(probe_q.shape), tuple(key.shape), scale,
            )

        logger.info(
            "[TokenSparseV2] select_indices: seq_len=%d, num_probe=%d, "
            "scores_range=[%.4f, %.4f], scores_mean=%.4f, scores_sum=%.4f",
            seq_len, num_probe,
            float(scores.min().item()), float(scores.max().item()),
            float(scores.mean().item()), float(scores.sum().item()),
        )

        keep_idx = _dynamic_token_coverage(
            scores,
            tau=cfg.tau,
            min_keep_ratio=cfg.min_keep_ratio,
            window_size=cfg.window_size,
        )

        n_keep = keep_idx.shape[0]
        if n_keep >= seq_len:
            logger.info(
                "[TokenSparseV2] select_indices: no sparsity (keep_idx=%d == seq_len=%d)",
                n_keep, seq_len,
            )
            return None

        logger.info(
            "[TokenSparseV2] select_indices: keep %d/%d tokens "
            "(drop_rate=%.2f%%), keep_idx_range=[%d, %d]",
            n_keep, seq_len, (seq_len - n_keep) / seq_len * 100,
            int(keep_idx.min().item()) if keep_idx.numel() > 0 else -1,
            int(keep_idx.max().item()) if keep_idx.numel() > 0 else -1,
        )
        return keep_idx


# ============================================================================
# vLLM V1 Backend Integration
# ============================================================================

@dataclass
class TokenSparseV2AttentionMetadata(FlashAttentionMetadata):
    """Metadata for Token Sparse Attention v2."""

    is_prefill: bool = True


class TokenSparseV2AttentionMetadataBuilder(
    AttentionMetadataBuilder[TokenSparseV2AttentionMetadata],
):
    """Builder for Token Sparse v2 attention metadata."""

    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

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
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.num_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.head_dim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        logger.info(
            "[TokenSparseV2] Builder init: num_heads=%s, num_kv_heads=%s, "
            "head_dim=%s, block_size=%s, device=%s",
            self.num_heads, self.num_kv_heads, self.head_dim,
            self.block_size, device,
        )

        # Read config from environment variables and hf_config
        hf_config = self.model_config.hf_text_config

        def _get_env_or_hf(env_name: str, hf_attr: str, default):
            env_val = os.environ.get(env_name)
            if env_val is not None:
                logger.info("[TokenSparseV2] Config %s from env: %s", env_name, env_val)
                return type(default)(env_val)
            hf_val = getattr(hf_config, hf_attr, None) if hf_config else None
            if hf_val is not None:
                logger.info("[TokenSparseV2] Config %s from hf_config.%s: %s", env_name, hf_attr, hf_val)
                return hf_val
            logger.info("[TokenSparseV2] Config %s using default: %s", env_name, default)
            return default

        self.tau = float(
            _get_env_or_hf("VLLM_TOKEN_SPARSE_V2_TAU", "token_sparse_v2_tau", 0.005)
        )
        self.min_seq_len = int(
            _get_env_or_hf(
                "VLLM_TOKEN_SPARSE_V2_MIN_SEQ_LEN",
                "token_sparse_v2_min_seq_len",
                1024,
            )
        )
        self.min_keep_ratio = float(
            _get_env_or_hf(
                "VLLM_TOKEN_SPARSE_V2_MIN_KEEP_RATIO",
                "token_sparse_v2_min_keep_ratio",
                0.10,
            )
        )
        self.num_probe_queries = int(
            _get_env_or_hf(
                "VLLM_TOKEN_SPARSE_V2_NUM_PROBE",
                "token_sparse_v2_num_probe",
                16,
            )
        )
        self.window_size = int(
            _get_env_or_hf(
                "VLLM_TOKEN_SPARSE_V2_WINDOW_SIZE",
                "token_sparse_v2_window_size",
                128,
            )
        )
        self.use_triton = (
            os.environ.get("VLLM_TOKEN_SPARSE_V2_USE_TRITON", "1").lower() == "1"
        )
        self.log_stats = (
            os.environ.get("VLLM_TOKEN_SPARSE_V2_LOG_STATS", "0").lower() == "1"
        )

        sparse_layers_env = os.environ.get("VLLM_TOKEN_SPARSE_V2_SPARSE_LAYERS", "")
        if sparse_layers_env:
            self.sparse_layers = [
                int(x.strip()) for x in sparse_layers_env.split(",") if x.strip()
            ]
        else:
            self.sparse_layers = None  # None = all layers

        logger.info(
            "[TokenSparseV2] MetadataBuilder initialized: "
            "tau=%s, min_seq_len=%s, min_keep_ratio=%s, "
            "num_probe=%s, window=%s, triton=%s, sparse_layers=%s",
            self.tau,
            self.min_seq_len,
            self.min_keep_ratio,
            self.num_probe_queries,
            self.window_size,
            self.use_triton,
            self.sparse_layers,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TokenSparseV2AttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        is_prefill = max_query_len > 1

        logger.info(
            "[TokenSparseV2] MetadataBuilder.build: num_actual_tokens=%s, "
            "max_query_len=%s, max_seq_len=%s, num_seqs=%s, is_prefill=%s, "
            "causal=%s, common_prefix_len=%s",
            num_actual_tokens, max_query_len, max_seq_len,
            query_start_loc.shape[0] - 1, is_prefill, causal, common_prefix_len,
        )

        return TokenSparseV2AttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            causal=causal,
            is_prefill=is_prefill,
            use_cascade=common_prefix_len > 0,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )

    def update_block_table(
        self,
        metadata: TokenSparseV2AttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> TokenSparseV2AttentionMetadata:
        import copy

        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


class TokenSparseV2AttentionImpl(FlashAttentionImpl):
    """Token Sparse Attention v2 implementation.

    Workflow:
        1. Prefill: Probe scoring -> dynamic coverage selection ->
           batched gather -> single FlashAttention -> scatter back.
        2. Decode: fallback to parent FlashAttention.
    """

    # Minimum sequence length to consider sparse attention.
    # vLLM uses chunked prefill; short chunks have low sparse benefit.
    SPARSE_MIN_SEQ_LEN = int(
        os.environ.get("VLLM_TOKEN_SPARSE_V2_SPARSE_MIN_SEQ_LEN", "32768")
    )

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
        # v2 specific params (passed from builder via layer creation)
        tau: float | None = None,
        min_seq_len: int | None = None,
        min_keep_ratio: float | None = None,
        num_probe_queries: int | None = None,
        window_size: int | None = None,
        use_triton: bool | None = None,
        log_stats: bool | None = None,
        sparse_layers: list[int] | None = None,
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

        # Determine if sparse should be used for this layer
        if sparse_layers is not None and self.layer_idx is not None:
            self.use_sparse = self.layer_idx in sparse_layers
        elif sparse_layers is not None and self.layer_idx is None:
            self.use_sparse = False
        else:
            self.use_sparse = True

        # Build config
        def _get(val, env_name, default):
            if val is not None:
                return val
            env_val = os.environ.get(env_name)
            if env_val is not None:
                return type(default)(env_val)
            return default

        cfg = TokenSparseV2Config(
            tau=_get(tau, "VLLM_TOKEN_SPARSE_V2_TAU", 0.005),
            min_seq_len=_get(min_seq_len, "VLLM_TOKEN_SPARSE_V2_MIN_SEQ_LEN", 1024),
            min_keep_ratio=_get(
                min_keep_ratio, "VLLM_TOKEN_SPARSE_V2_MIN_KEEP_RATIO", 0.10
            ),
            num_probe_queries=_get(
                num_probe_queries, "VLLM_TOKEN_SPARSE_V2_NUM_PROBE", 16
            ),
            window_size=_get(
                window_size, "VLLM_TOKEN_SPARSE_V2_WINDOW_SIZE", 128
            ),
            use_triton=_get(use_triton, "VLLM_TOKEN_SPARSE_V2_USE_TRITON", True),
            log_stats=_get(log_stats, "VLLM_TOKEN_SPARSE_V2_LOG_STATS", False),
        )
        self.config = cfg

        if self.use_sparse:
            self.selector = TokenSparseV2Selector(cfg)
        else:
            self.selector = None

        logger.info(
            "[TokenSparseV2] Impl initialized: layer_idx=%s, use_sparse=%s, "
            "num_heads=%s, num_kv_heads=%s, head_size=%s, scale=%s, "
            "tau=%s, min_seq_len=%s, min_keep_ratio=%s, "
            "window_size=%s, sparse_min_seq_len=%s, triton=%s, log_stats=%s, "
            "sparse_layers=%s, attn_type=%s",
            self.layer_idx, self.use_sparse, num_heads, num_kv_heads, head_size, scale,
            cfg.tau, cfg.min_seq_len, cfg.min_keep_ratio,
            cfg.window_size, self.SPARSE_MIN_SEQ_LEN, cfg.use_triton, cfg.log_stats,
            sparse_layers, attn_type,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TokenSparseV2AttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional token sparse attention."""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for TokenSparseV2"
            )

        if attn_metadata is None:
            logger.warning("[TokenSparseV2] layer=%s attn_metadata is None, filling output with zeros", self.layer_idx)
            return output.fill_(0)

        logger.info(
            "[TokenSparseV2] forward start: layer=%s, query_shape=%s, key_shape=%s, "
            "value_shape=%s, kv_cache_shape=%s, num_actual_tokens=%s, "
            "max_query_len=%s, max_seq_len=%s, is_prefill=%s, causal=%s",
            self.layer_idx, tuple(query.shape), tuple(key.shape),
            tuple(value.shape), tuple(kv_cache.shape) if kv_cache is not None else None,
            attn_metadata.num_actual_tokens, attn_metadata.max_query_len,
            attn_metadata.max_seq_len, attn_metadata.is_prefill, attn_metadata.causal,
        )

        # P0 Safety: chunked prefill detection.
        is_chunked_prefill = False
        if attn_metadata.is_prefill:
            cu_seqlens_q = attn_metadata.query_start_loc
            num_seqs = cu_seqlens_q.shape[0] - 1
            num_reqs = attn_metadata.seq_lens.shape[0]
            actual_seqs = min(num_seqs, num_reqs)
            if actual_seqs > 0:
                query_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1])[:actual_seqs]
                seq_lens = attn_metadata.seq_lens[:actual_seqs]
                is_chunked_prefill = not torch.all(query_lens == seq_lens).item()
                if is_chunked_prefill:
                    mismatch = (query_lens != seq_lens).nonzero(as_tuple=True)[0]
                    logger.warning(
                        "[TokenSparseV2] layer=%s CHUNKED PREFILL DETECTED: "
                        "%d/%d seqs have query_len < seq_len. "
                        "First mismatch seq=%d, query_len=%d, seq_len=%d. "
                        "Falling back to dense FlashAttention.",
                        self.layer_idx, mismatch.numel(), actual_seqs.item() if isinstance(actual_seqs, torch.Tensor) else actual_seqs,
                        int(mismatch[0].item()) if mismatch.numel() > 0 else -1,
                        int(query_lens[mismatch[0]].item()) if mismatch.numel() > 0 else -1,
                        int(seq_lens[mismatch[0]].item()) if mismatch.numel() > 0 else -1,
                    )

        # Only apply sparse attention during prefill on decoder causal attention
        apply_sparse = (
            self.use_sparse
            and self.selector is not None
            and attn_metadata.is_prefill
            and attn_metadata.causal
            and self.attn_type == AttentionType.DECODER
            and attn_metadata.max_query_len > self.SPARSE_MIN_SEQ_LEN
            and not is_chunked_prefill
        )

        if not apply_sparse:
            reason = []
            if not self.use_sparse:
                reason.append("use_sparse=False")
            if self.selector is None:
                reason.append("selector=None")
            if not attn_metadata.is_prefill:
                reason.append("not_prefill")
            if not attn_metadata.causal:
                reason.append("not_causal")
            if self.attn_type != AttentionType.DECODER:
                reason.append(f"attn_type={self.attn_type}")
            if attn_metadata.max_query_len <= self.SPARSE_MIN_SEQ_LEN:
                reason.append(f"max_query_len={attn_metadata.max_query_len}<={self.SPARSE_MIN_SEQ_LEN}")
            if is_chunked_prefill:
                reason.append("chunked_prefill")
            logger.info(
                "[TokenSparseV2] layer=%s FALLBACK to dense: %s",
                self.layer_idx, ", ".join(reason) if reason else "unknown",
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

        logger.info("[TokenSparseV2] layer=%s entering sparse batched path", self.layer_idx)
        return self._forward_sparse_batched(
            layer=layer,
            query=query,
            key=key,
            value=value,
            attn_metadata=attn_metadata,
            output=output,
        )

    def _forward_sparse_batched(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        attn_metadata: TokenSparseV2AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Batched token sparse attention.

        Scores and selects tokens per-sequence, gathers all sparse Q/K/V,
        runs a single batched FlashAttention call, then scatters back.
        """
        num_actual_tokens = attn_metadata.num_actual_tokens
        cu_seqlens_q = attn_metadata.query_start_loc
        num_seqs = cu_seqlens_q.shape[0] - 1

        device = query.device
        dtype = query.dtype
        H = query.shape[1]
        H_kv = key.shape[1]
        D = query.shape[2]

        q_actual = query[:num_actual_tokens]
        k_actual = key[:num_actual_tokens]
        v_actual = value[:num_actual_tokens]

        # Per-sequence token selection and gather
        q_sparse_list: list[torch.Tensor] = []
        k_sparse_list: list[torch.Tensor] = []
        v_sparse_list: list[torch.Tensor] = []
        keep_idx_list: list[torch.Tensor] = []
        orig_starts: list[int] = []
        new_cu: list[int] = [0]

        total_dropped = 0
        total_input = 0
        num_sparse_seqs = 0

        for b in range(num_seqs):
            q_start = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b + 1].item())
            seq_len = q_end - q_start

            q_b = q_actual[q_start:q_end]  # [N, H, D]
            k_b = k_actual[q_start:q_end]  # [N, H_kv, D]
            v_b = v_actual[q_start:q_end]  # [N, H_kv, D]

            logger.info(
                "[TokenSparseV2] seq=%d: q_start=%d, q_end=%d, seq_len=%d, "
                "q_shape=%s, k_shape=%s, v_shape=%s",
                b, q_start, q_end, seq_len, tuple(q_b.shape), tuple(k_b.shape), tuple(v_b.shape),
            )

            keep_idx = self.selector.select_indices(q_b, k_b, self.scale)

            if keep_idx is None:
                # Short sequence or nothing to drop: keep all
                keep_idx = torch.arange(
                    seq_len, device=device, dtype=torch.int32
                )
                logger.info(
                    "[TokenSparseV2] seq=%d: NO SPARSITY (seq_len=%d \u003c min_seq_len=%d or nothing dropped), "
                    "keeping all %d tokens",
                    b, seq_len, self.config.min_seq_len, seq_len,
                )
            else:
                num_sparse_seqs += 1
                total_dropped += seq_len - keep_idx.shape[0]
                logger.info(
                    "[TokenSparseV2] seq=%d: sparse selected %d/%d tokens "
                    "(dropped %d, keep_idx_range=[%d, %d])",
                    b, keep_idx.shape[0], seq_len,
                    seq_len - keep_idx.shape[0],
                    int(keep_idx.min().item()) if keep_idx.numel() > 0 else -1,
                    int(keep_idx.max().item()) if keep_idx.numel() > 0 else -1,
                )
                # NaN/Inf guard on keep_idx
                if keep_idx.numel() > 0:
                    if torch.isnan(keep_idx.float()).any():
                        logger.error("[TokenSparseV2] seq=%d: keep_idx contains NaN!", b)
                    if keep_idx.min().item() < 0 or keep_idx.max().item() >= seq_len:
                        logger.error(
                            "[TokenSparseV2] seq=%d: keep_idx out of range! "
                            "min=%d, max=%d, seq_len=%d",
                            b, int(keep_idx.min().item()), int(keep_idx.max().item()), seq_len,
                        )

            total_input += seq_len
            n_keep = keep_idx.shape[0]
            orig_starts.append(q_start)

            # Gather sparse Q/K/V
            if self.config.use_triton and q_b.is_contiguous():
                logger.info("[TokenSparseV2] seq=%d: using Triton gather", b)
                q_sparse_list.append(_gather_tokens(q_b, keep_idx))
                k_sparse_list.append(_gather_tokens(k_b.contiguous(), keep_idx))
                v_sparse_list.append(_gather_tokens(v_b.contiguous(), keep_idx))
            else:
                logger.info("[TokenSparseV2] seq=%d: using PyTorch indexing gather", b)
                q_sparse_list.append(q_b[keep_idx])
                k_sparse_list.append(k_b[keep_idx])
                v_sparse_list.append(v_b[keep_idx])

            keep_idx_list.append(keep_idx)
            new_cu.append(new_cu[-1] + n_keep)

        if self.config.log_stats and total_input > 0:
            ratio = total_dropped / total_input * 100
            logger.info(
                "[TokenSparseV2] layer=%s sparsity=%.1f%% (%d/%d dropped, "
                "%d/%d seqs sparse)",
                self.layer_idx,
                ratio,
                total_dropped,
                total_input,
                num_sparse_seqs,
                num_seqs,
            )

        # If no sparsity applied across the batch, fall back to dense
        if num_sparse_seqs == 0:
            return super().forward(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        # Batched sparse FlashAttention
        q_packed = torch.cat(q_sparse_list, dim=0)  # [Σn_keep, H, D]
        k_packed = torch.cat(k_sparse_list, dim=0)  # [Σn_keep, H_kv, D]
        v_packed = torch.cat(v_sparse_list, dim=0)  # [Σn_keep, H_kv, D]

        new_cu_tensor = torch.tensor(new_cu, dtype=torch.int32, device=device)
        max_sparse = max(int(new_cu[b + 1] - new_cu[b]) for b in range(num_seqs))

        logger.info(
            "[TokenSparseV2] FlashAttention input: q_packed=%s, k_packed=%s, "
            "v_packed=%s, new_cu=%s, max_sparse=%d, num_seqs=%d, "
            "H_q=%d, H_kv=%d, D=%d",
            tuple(q_packed.shape), tuple(k_packed.shape), tuple(v_packed.shape),
            new_cu, max_sparse, num_seqs, H, H_kv, D,
        )

        # NaN/Inf guard before FA
        for name, tensor in [("q_packed", q_packed), ("k_packed", k_packed), ("v_packed", v_packed)]:
            if torch.isnan(tensor).any():
                logger.error("[TokenSparseV2] %s contains NaN before FlashAttention!", name)
            if torch.isinf(tensor).any():
                logger.error("[TokenSparseV2] %s contains Inf before FlashAttention!", name)

        # Prepare output buffer (3D view for scatter)
        output[:num_actual_tokens].zero_()
        output_3d = output[:num_actual_tokens].view(num_actual_tokens, H, D)

        # Run FlashAttention on sparse packed tensors
        flash_kwargs: dict = dict(softmax_scale=self.scale, causal=True)
        if self.sliding_window != (-1, -1):
            flash_kwargs["window_size"] = list(self.sliding_window)
        if self.alibi_slopes is not None:
            flash_kwargs["alibi_slopes"] = self.alibi_slopes
        if self.logits_soft_cap != 0:
            flash_kwargs["softcap"] = self.logits_soft_cap

        logger.info("[TokenSparseV2] FlashAttention kwargs: %s", flash_kwargs)

        out_packed = flash_attn_varlen_func(
            q=q_packed,
            k=k_packed,
            v=v_packed,
            cu_seqlens_q=new_cu_tensor,
            cu_seqlens_k=new_cu_tensor,
            max_seqlen_q=max_sparse,
            max_seqlen_k=max_sparse,
            **flash_kwargs,
        )  # [Σn_keep, H, D]

        logger.info(
            "[TokenSparseV2] FlashAttention output: out_packed=%s, dtype=%s, device=%s",
            tuple(out_packed.shape), out_packed.dtype, out_packed.device,
        )

        # NaN/Inf guard after FA
        if torch.isnan(out_packed).any():
            logger.error("[TokenSparseV2] out_packed contains NaN after FlashAttention!")
        if torch.isinf(out_packed).any():
            logger.error("[TokenSparseV2] out_packed contains Inf after FlashAttention!")

        # Scatter back to full sequence positions
        for b in range(num_seqs):
            sp_start = int(new_cu[b])
            sp_end = int(new_cu[b + 1])
            out_b = out_packed[sp_start:sp_end]  # [n_keep, H, D]
            keep_idx = keep_idx_list[b]
            orig_start = orig_starts[b]
            seq_len_b = int(cu_seqlens_q[b + 1].item()) - orig_start

            logger.info(
                "[TokenSparseV2] scatter seq=%d: sp_start=%d, sp_end=%d, "
                "out_b=%s, keep_idx=%s, orig_start=%d, seq_len_b=%d",
                b, sp_start, sp_end, tuple(out_b.shape), tuple(keep_idx.shape),
                orig_start, seq_len_b,
            )

            if self.config.use_triton and out_b.is_contiguous():
                logger.info("[TokenSparseV2] seq=%d: using Triton scatter", b)
                out_slice = _scatter_tokens(out_b, keep_idx, seq_len_b)
                output_3d[orig_start : orig_start + seq_len_b] = out_slice
            else:
                logger.info("[TokenSparseV2] seq=%d: using PyTorch indexing scatter", b)
                output_3d[orig_start + keep_idx] = out_b

        # Final sanity check on output
        if torch.isnan(output).any():
            logger.error("[TokenSparseV2] layer=%s output contains NaN after scatter!", self.layer_idx)
        if torch.isinf(output).any():
            logger.error("[TokenSparseV2] layer=%s output contains Inf after scatter!", self.layer_idx)

        logger.info(
            "[TokenSparseV2] layer=%s sparse forward DONE: output_shape=%s, "
            "output_dtype=%s, output_device=%s",
            self.layer_idx, tuple(output.shape), output.dtype, output.device,
        )
        return output

    def _extract_layer_index(self, layer_name: str | None) -> int | None:
        """Extract layer index from layer_name string."""
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
        """Update KV cache with new key/value tensors."""
        if self.attn_type != AttentionType.DECODER:
            logger.info(
                "[TokenSparseV2] layer=%s skipping KV cache update: attn_type=%s",
                self.layer_idx, self.attn_type,
            )
            return
        if kv_cache is None or kv_cache.numel() == 0:
            logger.info(
                "[TokenSparseV2] layer=%s skipping KV cache update: kv_cache is empty",
                self.layer_idx,
            )
            return

        logger.info(
            "[TokenSparseV2] layer=%s KV cache update: key=%s, value=%s, "
            "kv_cache=%s, slot_mapping=%s, kv_cache_dtype=%s",
            self.layer_idx, tuple(key.shape), tuple(value.shape),
            tuple(kv_cache.shape), tuple(slot_mapping.shape), self.kv_cache_dtype,
        )

        key_cache, value_cache = kv_cache.unbind(0)
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


class TokenSparseV2AttentionBackend(FlashAttentionBackend):
    """Token Sparse Attention v2 Backend for vLLM."""

    @staticmethod
    def get_name() -> str:
        return "TOKEN_SPARSE_V2"

    @staticmethod
    def get_impl_cls() -> type[TokenSparseV2AttentionImpl]:
        return TokenSparseV2AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TokenSparseV2AttentionMetadataBuilder]:
        return TokenSparseV2AttentionMetadataBuilder
