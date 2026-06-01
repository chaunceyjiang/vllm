# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashQLA GDN prefill backend integration."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _is_flash_qla_available,
    _resolve_gdn_prefill_backend,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FlashQLA requires SM90+ GPU",
)
def test_flash_qla_available_on_sm90():
    """FlashQLA should report available on SM90+ when installed."""
    assert _is_flash_qla_available()


def test_resolve_gdn_prefill_backend_flashqla_explicit():
    """Explicitly requesting flashqla should resolve to flashqla on SM90."""
    if not _is_flash_qla_available():
        pytest.skip("FlashQLA not available")

    vllm_config = MagicMock()
    vllm_config.additional_config = {"gdn_prefill_backend": "flashqla"}
    vllm_config.model_config.hf_config.linear_key_head_dim = 128

    requested, active = _resolve_gdn_prefill_backend(vllm_config)
    assert requested == "flashqla"
    assert active == "flashqla"


def test_resolve_gdn_prefill_backend_auto_prefers_flashqla():
    """Auto should prefer flashqla when available on SM90."""
    if not _is_flash_qla_available():
        pytest.skip("FlashQLA not available")

    vllm_config = MagicMock()
    vllm_config.additional_config = {"gdn_prefill_backend": "auto"}
    vllm_config.model_config.hf_config.linear_key_head_dim = 128

    requested, active = _resolve_gdn_prefill_backend(vllm_config)
    assert requested == "auto"
    assert active == "flashqla"


def test_resolve_gdn_prefill_backend_fallback_to_flashinfer():
    """When flashqla is requested but not available, fall back."""
    vllm_config = MagicMock()
    vllm_config.additional_config = {"gdn_prefill_backend": "flashqla"}
    vllm_config.model_config.hf_config.linear_key_head_dim = 128

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn._is_flash_qla_available",
            lambda: False,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend as resolve,
        )

        requested, active = resolve(vllm_config)
        assert requested == "flashqla"
        # Should fall back to flashinfer on SM90 or triton otherwise
        assert active in ("flashinfer", "triton")


@pytest.mark.skipif(
    not _is_flash_qla_available(),
    reason="FlashQLA not available",
)
def test_chunk_gated_delta_rule_flashqla_method_exists():
    """ChunkGatedDeltaRule should have the flashqla forward method."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    assert hasattr(ChunkGatedDeltaRule, "forward_flashqla")
