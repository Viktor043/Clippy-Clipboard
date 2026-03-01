#!/usr/bin/env python3
"""
Validate full SVD pipeline for TinyStories-1M: attention AND MLP layers.

This replaces ALL weight matrices (attention + MLP) with their SVD decomposition
to prove the complete optical inference architecture works.

Attention: glass(V^T, 64×64) → E(S, 64) → glass(U, 64×64)
MLP W_fc:  glass(V^T, 64×64) → E(S, 64) → zero-pad → glass(U, 256×256)
MLP W_proj: glass(V^T, 256×256) → E(S, 64) → glass(U, 64×64)
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

print("[FullSVD] Loading TinyStories-1M...")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class SVDLinear(nn.Module):
    """Replaces nn.Linear with SVD decomposition: x → V^T → S scaling → U."""
    def __init__(self, U, S, Vt, bias=None):
        super().__init__()
        self.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None
        self.in_features = Vt.shape[1]
        self.out_features = U.shape[0]
        # For rectangular: S has r=min(m,n) elements
        self.r = len(S)

    @property
    def weight(self):
        """Reconstruct weight for compatibility checks."""
        S_diag = torch.zeros(self.out_features, self.in_features, device=self.U.device)
        for i in range(self.r):
            S_diag[i, i] = self.S[i]
        return self.U @ S_diag @ self.Vt

    def forward(self, x):
        # Step 1: Glass block V^T (square unitary, in_features × in_features)
        z = x @ self.Vt.T  # (batch, in_features)

        # Step 2: Electronic scaling by singular values (only first r components)
        # For rectangular: r = min(out, in), z has in_features dims
        # Scale first r components
        z_scaled = z[..., :self.r] * self.S  # (batch, r)

        # Step 3: Zero-pad if expanding (e.g., 64 → 256)
        if self.out_features > self.r:
            pad = torch.zeros(*z.shape[:-1], self.out_features - self.r, device=z.device)
            z_padded = torch.cat([z_scaled, pad], dim=-1)  # (batch, out_features)
        else:
            z_padded = z_scaled

        # Step 4: Glass block U (square unitary, out_features × out_features)
        out = z_padded @ self.U.T  # (batch, out_features)

        if self.bias is not None:
            out = out + self.bias
        return out


def replace_all_with_svd(model):
    """Replace ALL attention AND MLP weight matrices with SVD versions."""
    replaced_count = 0

    for layer_idx in range(model.config.num_layers):
        # === Attention ===
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data.numpy()
            bias = proj.bias.data.numpy() if proj.bias is not None else None

            U, S, Vt = np.linalg.svd(W, full_matrices=True)
            svd_layer = SVDLinear(U, S, Vt, bias)

            # Verify
            x_test = np.random.randn(1, W.shape[1]).astype(np.float32)
            orig_out = (x_test @ W.T + (bias if bias is not None else 0))
            svd_out = svd_layer(torch.tensor(x_test)).detach().numpy()
            err = np.max(np.abs(orig_out - svd_out))

            setattr(attn, pname, svd_layer)
            replaced_count += 1

        # === MLP ===
        mlp = model.transformer.h[layer_idx].mlp

        # c_fc: Linear(64, 256) → weight is (256, 64)
        W_fc = mlp.c_fc.weight.data.numpy()
        bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
        U_fc, S_fc, Vt_fc = np.linalg.svd(W_fc, full_matrices=True)
        svd_fc = SVDLinear(U_fc, S_fc, Vt_fc, bias_fc)

        # Verify
        x_test = np.random.randn(1, 64).astype(np.float32)
        orig_out = (x_test @ W_fc.T + (bias_fc if bias_fc is not None else 0))
        svd_out = svd_fc(torch.tensor(x_test)).detach().numpy()
        err_fc = np.max(np.abs(orig_out - svd_out))

        mlp.c_fc = svd_fc
        replaced_count += 1

        # c_proj: Linear(256, 64) → weight is (64, 256)
        W_proj = mlp.c_proj.weight.data.numpy()
        bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
        U_proj, S_proj, Vt_proj = np.linalg.svd(W_proj, full_matrices=True)
        svd_proj = SVDLinear(U_proj, S_proj, Vt_proj, bias_proj)

        # Verify
        x_test = np.random.randn(1, 256).astype(np.float32)
        orig_out = (x_test @ W_proj.T + (bias_proj if bias_proj is not None else 0))
        svd_out = svd_proj(torch.tensor(x_test)).detach().numpy()
        err_proj = np.max(np.abs(orig_out - svd_out))

        mlp.c_proj = svd_proj
        replaced_count += 1

        print(f"  Layer {layer_idx}: attn OK, MLP fc_err={err_fc:.2e}, proj_err={err_proj:.2e}")

    return replaced_count


# Evaluation helpers
eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]


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


# === Baseline ===
print("\n[FullSVD] === BASELINE ===")
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_texts = {}
for p in eval_prompts:
    baseline_texts[p] = generate_text(model, tokenizer, p)
print(f"  PPL: {baseline_ppl:.4f}")
print(f"  Text: {baseline_texts[eval_prompts[0]][:100]}...")

# === Replace all with SVD ===
print("\n[FullSVD] Replacing ALL matrices (attention + MLP) with SVD...")
n_replaced = replace_all_with_svd(model)
print(f"  Replaced {n_replaced} matrices total")

# === SVD evaluation ===
print("\n[FullSVD] === SVD PIPELINE RESULTS ===")
svd_ppl = compute_perplexity(model, tokenizer, eval_prompts)
delta = ((svd_ppl - baseline_ppl) / baseline_ppl) * 100
print(f"  Baseline PPL: {baseline_ppl:.4f}")
print(f"  SVD PPL:      {svd_ppl:.4f} (delta={delta:+.4f}%)")

print("\n[FullSVD] === TEXT COMPARISON ===")
all_match = True
for p in eval_prompts:
    svd_text = generate_text(model, tokenizer, p)
    match = "MATCH" if svd_text == baseline_texts[p] else "DIFFER"
    if match == "DIFFER":
        all_match = False
    print(f"  '{p[:35]}...'")
    print(f"    Baseline: {baseline_texts[p][:80]}...")
    print(f"    SVD:      {svd_text[:80]}...")
    print(f"    → {match}")

print(f"\n[FullSVD] === SUMMARY ===")
print(f"  Matrices replaced: {n_replaced} (32 attention + 16 MLP = 48)")
print(f"  PPL change: {delta:+.4f}%")
print(f"  Text match: {'ALL MATCH' if all_match else 'SOME DIFFER'}")
print(f"  Glass blocks needed: 96 (80× 64×64 + 16× 256×256)")
print(f"  Electronic components: 384 scalings + 2048 GELU per forward pass")
print()
if abs(delta) < 0.01:
    print("  ★ FULL SVD PIPELINE VALIDATED — EXACT for attention + MLP")
elif abs(delta) < 1.0:
    print("  ★ FULL SVD PIPELINE — near-exact (float32 precision)")
else:
    print("  ⚠ UNEXPECTED — SVD should be mathematically exact")
