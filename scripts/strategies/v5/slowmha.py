"""SlowMHA: exact slow-path port (shared)."""
from typing import Optional

import torch


class SlowMHA(torch.nn.Module):
    """Faithful port of torch 1.x multi_head_attention_forward slow path,
    specialized to this model's usage: batched 3D self-attention, packed
    in-proj, bias_k/v None, statics None, add_zero_attn False, eval
    (dropout 0), need_weights=True, average=True.
    key_padding_mask: bool (B,S) tensor (None accepted but baked per trace).
    attn_mask: None (other cases raise loudly)."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sqrt_head = float(embed_dim // num_heads) ** 0.5
        self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                average_attn_weights: bool = True):
        # batch_first=True: I/O is (B,T,E); transpose to (T,B,E) for the core
        # (archived source, activation module lines 284-289 and 331-333).
        query = torch.transpose(query, 1, 0)
        key = torch.transpose(key, 1, 0)
        value = torch.transpose(value, 1, 0)
        tgt_len, bsz, embed_dim = query.size()
        num_heads = self.num_heads
        head_dim = self.head_dim
        q, k, v = torch.chunk(
            torch.nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias),
            3, dim=-1)
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        src_len = k.size(0)
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        if attn_mask is not None:
            raise RuntimeError("SlowMHA port expects attn_mask=None in this model")
        if key_padding_mask is None:
            scores = torch.bmm(q / self.sqrt_head, k.transpose(-2, -1))
        else:
            # bool (B,S) -> (B*H,1,S), then additive -inf mask (source 549-578)
            kpm = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                bsz, num_heads, 1, src_len)
            kpm = kpm.reshape(bsz * num_heads, 1, src_len)
            merged = torch.zeros_like(kpm, dtype=q.dtype).masked_fill(kpm, float("-inf"))
            # scaled dot-product (source _scaled_dot_product_attention)
            scores = torch.baddbmm(merged, q / self.sqrt_head, k.transpose(-2, -1))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn, v)
        out = torch.transpose(out.contiguous(), 0, 1).contiguous()
        out = out.view(tgt_len * bsz, embed_dim)
        out = torch.nn.functional.linear(out, self.out_proj.weight, self.out_proj.bias)
        out = out.view(tgt_len, bsz, embed_dim)
        w = attn.view(bsz, num_heads, tgt_len, src_len).sum(dim=1) / num_heads
        out = torch.transpose(out, 1, 0)
        return out, w


