import torch
import math
from vllm.v1.attention.backends.token_sparse import triton_get_attn_cache_packed

def reference_attn_cache_packed(query, key, cu_seqlens, window_size):
    """
    Reference PyTorch implementation of triton_get_attn_cache_packed.
    """
    total_tokens, H, D = query.shape
    num_seqs = cu_seqlens.shape[0] - 1
    out = torch.zeros(total_tokens, H, device=query.device, dtype=torch.float32)

    for seq_idx in range(num_seqs):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        seq_len = end - start
        W = min(window_size, seq_len)
        prefix_len = seq_len - W
        if prefix_len <= 0:
            continue

        q_seq = query[start:end]  # (L, H, D)
        k_seq = key[start:end]    # (L, H, D)

        # Window queries: (W, H, D)
        q_win = q_seq[prefix_len:]  # (W, H, D)

        # Compute full attention scores for window queries
        # scores: (W, H, L)
        scores = torch.matmul(q_win, k_seq.transpose(0, 1).transpose(1, 2)) / math.sqrt(D)
        # q_win: (W, H, D), k_seq: (L, H, D)
        # We need (H, W, L)
        scores = torch.einsum('whd,lhd->hwl', q_win, k_seq) / math.sqrt(D)

        # Causal mask: for each query, mask future tokens
        # Create mask (H, W, L)
        q_pos = torch.arange(prefix_len, seq_len, device=query.device).unsqueeze(0).unsqueeze(-1)  # (1, W, 1)
        k_pos = torch.arange(seq_len, device=query.device).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        mask = k_pos > q_pos  # (1, W, L) -> broadcast to (H, W, L)

        # Also, window keys should only be masked if future
        # prefix keys are always allowed
        prefix_mask = torch.arange(seq_len, device=query.device) < prefix_len  # (L,)
        prefix_mask = prefix_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L)

        # Apply mask: future window keys are -inf, prefix keys are never masked
        scores = scores.masked_fill(mask & ~prefix_mask, float('-inf'))

        # Softmax
        m = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - m)
        l_sum = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / l_sum  # (H, W, L)

        # Average over window queries
        mean_probs = probs.mean(dim=1)  # (H, L)

        # Only prefix positions
        out[start:start+prefix_len] = mean_probs[:, :prefix_len].t()

    return out


def test_kernel():
    torch.manual_seed(42)
    device = "cuda"
    H = 8
    D = 64
    window_size = 16

    # Test case 1: two sequences
    seq_lens = [100, 80]
    cu_seqlens = torch.tensor([0, 100, 180], device=device, dtype=torch.int32)
    total_tokens = 180

    query = torch.randn(total_tokens, H, D, device=device, dtype=torch.float32)
    key = torch.randn(total_tokens, H, D, device=device, dtype=torch.float32)

    out_triton = triton_get_attn_cache_packed(query, key, cu_seqlens, window_size)
    out_ref = reference_attn_cache_packed(query, key, cu_seqlens, window_size)

    diff = (out_triton - out_ref).abs()
    print(f"Max diff: {diff.max().item():.6e}")
    print(f"Mean diff: {diff.mean().item():.6e}")

    # Check specific positions
    for seq_idx, seq_len in enumerate(seq_lens):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        prefix_len = seq_len - min(window_size, seq_len)
        if prefix_len > 0:
            seq_diff = diff[start:start+prefix_len]
            print(f"Seq {seq_idx} (len={seq_len}, prefix={prefix_len}) max diff: {seq_diff.max().item():.6e}")

    if diff.max() < 1e-3:
        print("PASS")
    else:
        print("FAIL")
        # Print some values for debugging
        print("\nTriton values (seq 0, head 0, first 10 prefix positions):")
        print(out_triton[:10, 0])
        print("\nRef values (seq 0, head 0, first 10 prefix positions):")
        print(out_ref[:10, 0])


if __name__ == "__main__":
    test_kernel()
