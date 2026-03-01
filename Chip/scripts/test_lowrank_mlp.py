#!/usr/bin/env python3
"""
Test low-rank MLP truncation: how much can we compress MLP singular values?

If MLP weights have rapidly decaying singular values, we can truncate to rank k < 64.
This would shrink or eliminate the 256×256 glass blocks.

For W_fc (256×64): full rank is 64. If rank k suffices:
  - V^T: 64×64 (unchanged)
  - S: only k values
  - U: 256×k → pad to 256×256 but only k columns active
  - Or better: if k ≤ 64, the whole pipeline stays at 64×64!

For W_proj (64×256): full rank is 64. If rank k suffices:
  - V^T: 256×256 → if k ≤ 64, can use a 64×64 block on the k active components!
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[LowRank] Loading TinyStories-1M...")
model_name = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]


class TruncatedSVDLinear(nn.Module):
    """Linear layer using truncated SVD: only top-k singular values."""
    def __init__(self, W, bias=None, rank_k=None):
        super().__init__()
        m, n = W.shape
        U, S, Vt = np.linalg.svd(W, full_matrices=False)  # thin SVD
        r = len(S)
        if rank_k is not None and rank_k < r:
            U = U[:, :rank_k]    # m × k
            S = S[:rank_k]       # k
            Vt = Vt[:rank_k, :]  # k × n

        self.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        # x @ V^T.T @ diag(S) @ U^T = x @ V @ diag(S) @ U^T
        z = x @ self.Vt.T        # (batch, k)
        z = z * self.S            # (batch, k) — element-wise
        out = z @ self.U.T        # (batch, m)
        if self.bias is not None:
            out = out + self.bias
        return out


def compute_perplexity(model, tokenizer, prompts):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new=40):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(inputs["input_ids"], max_new_tokens=max_new,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# === First: analyze singular value spectra ===
print("\n[LowRank] === SINGULAR VALUE ANALYSIS ===")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

for layer_idx in [0, 3, 7]:
    mlp = model.transformer.h[layer_idx].mlp
    W_fc = mlp.c_fc.weight.data.numpy()
    W_proj = mlp.c_proj.weight.data.numpy()

    _, S_fc, _ = np.linalg.svd(W_fc, full_matrices=False)
    _, S_proj, _ = np.linalg.svd(W_proj, full_matrices=False)

    # Cumulative energy
    energy_fc = np.cumsum(S_fc**2) / np.sum(S_fc**2) * 100
    energy_proj = np.cumsum(S_proj**2) / np.sum(S_proj**2) * 100

    print(f"\n  Layer {layer_idx} W_fc (256×64):")
    print(f"    σ_max={S_fc[0]:.4f} σ_min={S_fc[-1]:.4f} ratio={S_fc[0]/S_fc[-1]:.1f}")
    for k in [4, 8, 16, 32, 48, 64]:
        if k <= len(S_fc):
            print(f"    rank {k:2d}: energy={energy_fc[k-1]:.2f}%  σ_{k}={S_fc[min(k-1, len(S_fc)-1)]:.4f}")

    print(f"  Layer {layer_idx} W_proj (64×256):")
    print(f"    σ_max={S_proj[0]:.4f} σ_min={S_proj[-1]:.4f} ratio={S_proj[0]/S_proj[-1]:.1f}")
    for k in [4, 8, 16, 32, 48, 64]:
        if k <= len(S_proj):
            print(f"    rank {k:2d}: energy={energy_proj[k-1]:.2f}%  σ_{k}={S_proj[min(k-1, len(S_proj)-1)]:.4f}")


# === Test various rank truncations ===
print("\n[LowRank] === PPL vs RANK (MLP only, attention untouched) ===")

# Baseline
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"  Baseline PPL: {baseline_ppl:.4f}")

ranks_to_test = [64, 48, 32, 24, 16, 12, 8, 4]
print(f"\n{'Rank':>6s} {'PPL':>10s} {'Delta%':>10s} {'Text':>50s}")
print("-" * 80)

for rank_k in ranks_to_test:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Replace MLP only (attention stays original)
    for layer_idx in range(model.config.num_layers):
        mlp = model.transformer.h[layer_idx].mlp

        W_fc = mlp.c_fc.weight.data.numpy()
        bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
        mlp.c_fc = TruncatedSVDLinear(W_fc, bias_fc, rank_k=rank_k)

        W_proj = mlp.c_proj.weight.data.numpy()
        bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
        mlp.c_proj = TruncatedSVDLinear(W_proj, bias_proj, rank_k=rank_k)

    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    text = generate_text(model, tokenizer, eval_prompts[0])
    text_status = "MATCH" if text == baseline_text else text[40:85]
    print(f"{rank_k:>4d}   {ppl:>10.2f} {delta:>+9.2f}%   {text_status}")


# === Also test truncating BOTH attention and MLP ===
print(f"\n[LowRank] === PPL vs RANK (attention + MLP, everything truncated) ===")
print(f"{'Rank':>6s} {'PPL':>10s} {'Delta%':>10s} {'Text':>50s}")
print("-" * 80)

for rank_k in ranks_to_test:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    for layer_idx in range(model.config.num_layers):
        # Attention
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data.numpy()
            bias = proj.bias.data.numpy() if proj.bias is not None else None
            setattr(attn, pname, TruncatedSVDLinear(W, bias, rank_k=min(rank_k, W.shape[0])))

        # MLP
        mlp = model.transformer.h[layer_idx].mlp
        W_fc = mlp.c_fc.weight.data.numpy()
        bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
        mlp.c_fc = TruncatedSVDLinear(W_fc, bias_fc, rank_k=rank_k)

        W_proj = mlp.c_proj.weight.data.numpy()
        bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
        mlp.c_proj = TruncatedSVDLinear(W_proj, bias_proj, rank_k=rank_k)

    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    text = generate_text(model, tokenizer, eval_prompts[0])
    text_status = "MATCH" if text == baseline_text else text[40:85]
    print(f"{rank_k:>4d}   {ppl:>10.2f} {delta:>+9.2f}%   {text_status}")


# === Implications for glass architecture ===
print(f"\n[LowRank] === IMPLICATIONS FOR GLASS ARCHITECTURE ===")
print(f"If rank k suffices for MLP:")
print(f"  rank 64 (full): 256×256 glass blocks needed")
print(f"  rank 32: could use 2× parallel 32×32 pipelines (all fit in 64×64)")
print(f"  rank 16: 16×16 glass blocks sufficient (proven at 100dB)")
print(f"  rank 8:  8×8 glass blocks sufficient (proven at 100dB)")
print(f"Key: if rank ≤ 64, MLP pipeline stays entirely within 64×64 glass blocks!")
print(f"     (V^T is at most 64×64, U columns beyond rank k are unused)")
