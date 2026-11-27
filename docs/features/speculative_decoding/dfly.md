# DFly

DFly is a checkpoint-native speculative decoder. It drafts multiple tokens
using target hidden states, sparse attention indices, and shared KV storage.
This guide covers DFly with HY4, HY4 Flash, and DeepSeek-V4.1. DFly is disabled by default; enable it
with `--speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'`.
The checkpoint must include DFly configuration and draft weights.

DFly is introduced in [AngelSpec: Towards Real-World High Performance Inference
with Speculative Decoding](https://arxiv.org/abs/2607.25852).

## Usage

For HY4 Flash FP8 on eight NVIDIA GPUs, with five speculative tokens:

```bash
vllm serve tencent/Hy4-flash-FP8 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'
```

## HY4 supported configurations

- NVIDIA GPUs or AMD gfx950 GPUs (MI350X, MI355X), and the GPU Model Runner V2.
- HY4 and HY4 Flash checkpoints with compatible DFly configuration and draft
  weights.
- BF16 computation with unquantized weights, online MoE-only block FP8
  quantization, native block FP8 checkpoints, or compressed-tensors checkpoints
  without a KV cache scheme or transforms. HY4 targets can use any layer scheme
  the model supports, such as NVFP4 experts. HY4 Flash targets are limited to
  W8A8 FP8 (128×128 weight blocks, dynamic group-128 activations) and weight-only
  MXFP4 (group-32). The shared embedding and LM head must be BF16.
- Draft layers in BF16 or W8A8 FP8 with 128×128 weight blocks and dynamic
  activations, independent of the target's scheme.
- Matching target and draft KV cache dtype: BF16 (`auto` or `bfloat16`)
  or `fp8_ds_mla`. Omit the draft KV override to inherit the target dtype.
  `fp8_ds_mla` is CUDA-only; ROCm requires BF16.
- Matching target and draft tensor parallelism. Pipeline, prefill context,
  and decode context parallel sizes must be one. HY4 Flash supports data
  parallelism, with each DP replica running a TP-local draft.
- Greedy or probabilistic draft sampling (`draft_sample_method`) with standard
  target verification. Probabilistic drafts sample at each request's
  temperature, which can improve acceptance when HY4 Flash samples at a
  nonzero temperature.

HY4 DFly currently rejects LoRA, microbatching, disaggregated prefill/decode,
heterogeneous vocabularies, adaptive verification, and a separate draft
attention backend.

On ROCm, DFly is validated with HY4 FP8 (`tencent/Hy4-FP8`), HY4 Flash FP8
(`tencent/Hy4-flash-FP8`) and HY4 Flash BF16 (`tencent/Hy4-flash`) on eight
MI355X GPUs with TP8. Other targets, such as HY4 BF16, NVFP4 experts, or
MXFP4, are accepted on gfx950 but have not been validated. Set
`VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1`, as described in the
[HY4 model notes](../../models/supported_models.md):

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1 \
vllm serve tencent/Hy4-flash-FP8 \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --speculative_config '{"num_speculative_tokens":5,"method":"dfly"}'
```

Measure acceptance and latency on your own workload using the
[acceptance metrics](acceptance_metrics.md) and
[benchmark CLI](../../benchmarking/cli.md). This implementation does not
guarantee a speedup for every workload.

## DeepSeek-V4.1 DFly2

DeepSeek-V4.1 uses a separate `DFlyDeepseekV41DraftModel` checkpoint. The
target and draft configurations must agree on the vocabulary, hidden dimensions,
RoPE parameters, and target KV/index source layers. For a block-size-six
checkpoint, use five speculative tokens:

```bash
vllm serve /path/to/DeepSeek-V4.1-Flash-1017 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_ds_mla \
    --attention-config '{"backend":"FLASHMLA_SPARSE_DSV41"}' \
    --speculative-config '{"method":"dfly","model":"/path/to/DS-V4.1-Flash-DFly2-B6","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Supported NVIDIA architectures are Hopper (SM90) and Blackwell datacenter
(SM100 family, including SM103). BF16 specifies computation; native MXFP4
expert weights remain quantized. The draft reads the target's `fp8_ds_mla`
cache and uses BF16 dense weights and shared embedding/LM head weights.
The target and draft TP sizes must match within each engine. DP, PP, and both
context-parallel sizes must be one. LoRA, microbatching, adaptive verification,
and a separate draft attention backend are unsupported.

### NIXL prefill/decode separation

DeepSeek-V4.1 DFly2 supports `NixlConnector` pull-mode PD. This does not enable
PD for HY4 DFly or for other connectors. Configure the same target, draft,
KV dtype, and speculative settings on the P and D engines. A deployment can
use two P engines with TP2 each and one D engine with TP4.

1. P computes the target prompt KV with `do_remote_decode=true` and returns
   transfer metadata. The router requests one P output token, which is not
   returned as an additional user-visible token.
2. The router forwards the original request and returned `kv_transfer_params`
   to D. NIXL transfers target KV. D recomputes the final prompt row locally.
3. DFly2 reads D's target KV, local hidden states, and local sparse index
   snapshots, then proposes five tokens for standard target verification.

Use a unique engine ID and side-channel port for each engine. For example,
add the following transfer configurations to the corresponding server commands:

```bash
# P1; use a different engine_id and side-channel port for P2.
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","engine_id":"dfly-p1"}'

# D
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","engine_id":"dfly-d"}'
```

See [Disaggregated Prefilling](../disagg_prefill.md) for connector deployment
requirements. P can use eager execution while D uses `FULL_DECODE_ONLY` CUDA
graphs. Verify actual NIXL transfer metrics as well as successful HTTP responses;
an HTTP-only smoke test does not prove that prompt KV was transferred.

For acceptance reporting, take idle-to-idle differences of the D engine's
`spec_decode_num_accepted_tokens_total`, `spec_decode_num_drafts_total`, and
`spec_decode_num_draft_tokens_total`. Accepted tokens divided by draft rounds
is the mean accepted draft length **without** a bonus token. Accepted tokens
divided by proposed tokens is the acceptance rate. Do not add P's counters.
