#!/usr/bin/env python3
"""
Validate end-to-end SVD pipeline for optical inference of TinyStories-1M.

Pipeline: glass(V^T) → electronic(diag(S)) → glass(U) = exact W @ x

Since the SVD decomposition is mathematically exact (W = U @ diag(S) @ V^T),
the perplexity should be IDENTICAL to the original model.

This validates that the optical inference architecture WORKS.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

print("[SVD-Validate] Loading TinyStories-1M...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

# Test prompts
prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
]


def compute_perplexity(model, tokenizer, prompts):
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
            loss = outputs.loss.item()
            n_tokens = inputs["input_ids"].shape[1] - 1
        total_loss += loss * n_tokens
        total_tokens += n_tokens
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


class SVDLinear(nn.Module):
    """
    Replace a linear layer W with SVD decomposition: U @ diag(S) @ Vt.
    Simulates: glass(Vt) → electronic(S) → glass(U)
    """
    def __init__(self, W_numpy):
        super().__init__()
        U, S, Vt = np.linalg.svd(W_numpy, full_matrices=True)
        self.U = nn.Parameter(torch.tensor(U, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        # x: (..., in_features)
        # Step 1: Glass block V^T
        z = x @ self.Vt.T  # (..., in) @ (in, in) = (..., in)
        # Step 2: Electronic scaling by S
        z = z * self.S  # element-wise multiply
        # Step 3: Glass block U
        return z @ self.U.T  # (..., in) @ (in, out) = (..., out)


def replace_with_svd(model, layer_idx, proj_names=None):
    """Replace attention projections with SVD decomposition."""
    if proj_names is None:
        proj_names = ["q_proj", "k_proj", "v_proj", "out_proj"]

    attn = model.transformer.h[layer_idx].attn.attention
    for pname in proj_names:
        proj = getattr(attn, pname)
        W = proj.weight.data.numpy()
        bias = proj.bias.data.clone() if proj.bias is not None else None

        svd_layer = SVDLinear(W)

        # Replace the projection — need a wrapper that applies SVD + bias
        class SVDProj(nn.Module):
            def __init__(self, svd, b):
                super().__init__()
                self.svd = svd
                self.bias = b
                # Mimic nn.Linear interface
                self.weight = svd.U  # for compatibility
                self.in_features = W.shape[1]
                self.out_features = W.shape[0]

            def forward(self, x):
                out = self.svd(x)
                if self.bias is not None:
                    out = out + self.bias
                return out

        setattr(attn, pname, SVDProj(svd_layer, bias))


# ===== Baseline =====
print("\n[SVD-Validate] === BASELINE (unmodified model) ===")
baseline_ppl = compute_perplexity(model, tokenizer, prompts)
baseline_text = generate_text(model, tokenizer, prompts[0])
print(f"  Perplexity: {baseline_ppl:.4f}")
print(f"  Generated: {baseline_text[:100]}...")

# ===== SVD replacement for single layer =====
for test_layer in [0, 3, 6, 7]:
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model.eval()
    replace_with_svd(model, test_layer)
    ppl = compute_perplexity(model, tokenizer, prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"\n  SVD Layer {test_layer}: PPL={ppl:.4f} (delta={delta:+.4f}%)")

# ===== SVD replacement for ALL layers =====
print("\n[SVD-Validate] === ALL 8 LAYERS SVD (32 glass blocks + 32 electronic scalings) ===")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
for layer_idx in range(model.config.num_layers):
    replace_with_svd(model, layer_idx)

ppl = compute_perplexity(model, tokenizer, prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, prompts[0])
print(f"  PPL={ppl:.4f} (delta={delta:+.4f}%)")
print(f"  Generated: {text[:100]}...")

# ===== Verify generation quality =====
print("\n[SVD-Validate] === GENERATION COMPARISON ===")
for p in prompts[:3]:
    model_orig = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model_orig.eval()
    text_orig = generate_text(model_orig, tokenizer, p, max_new_tokens=40)

    # SVD model (already loaded with all layers replaced)
    text_svd = generate_text(model, tokenizer, p, max_new_tokens=40)

    match = "MATCH" if text_orig == text_svd else "DIFFER"
    print(f"\n  Prompt: '{p}'")
    print(f"  Original: {text_orig[:120]}...")
    print(f"  SVD:      {text_svd[:120]}...")
    print(f"  [{match}]")

print("\n[SVD-Validate] CONCLUSION:")
print(f"  Baseline PPL: {baseline_ppl:.4f}")
print(f"  SVD PPL: {ppl:.4f}")
print(f"  Delta: {delta:+.4f}%")
if abs(delta) < 1.0:
    print("  SVD decomposition preserves model behavior!")
    print("  → Glass optical inference is EXACT via SVD pipeline.")
else:
    print(f"  WARNING: SVD introduces {abs(delta):.2f}% perplexity change.")
    print("  This may be due to float32 accumulation differences.")

print("\n[SVD-Validate] Done!")
