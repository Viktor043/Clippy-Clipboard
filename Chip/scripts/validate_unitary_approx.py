#!/usr/bin/env python3
"""
Validate the impact of unitary approximation on TinyStories-1M inference.

For each attention layer, replace W_Q/W_K/W_V/W_O with their nearest unitary (UV^T).
Measure the change in model output and perplexity.

This answers the critical question: can glass (which implements UV^T perfectly)
preserve the model's behavior?
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

print("[Validate] Loading TinyStories-1M...")
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

def compute_perplexity(model, tokenizer, prompts, max_length=50):
    """Compute average perplexity over prompts."""
    total_loss = 0.0
    total_tokens = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            n_tokens = input_ids.shape[1] - 1  # shifted by 1

        total_loss += loss * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def replace_with_unitary(model, layer_idx, proj_names=None):
    """Replace attention weight matrices with their nearest unitary approximation."""
    if proj_names is None:
        proj_names = ["q_proj", "k_proj", "v_proj", "out_proj"]

    attn = model.transformer.h[layer_idx].attn.attention
    replaced = {}

    for pname in proj_names:
        proj = getattr(attn, pname)
        W = proj.weight.data.numpy()
        U, S, Vt = np.linalg.svd(W, full_matrices=True)

        # Nearest unitary = UV^T
        W_unitary = U @ Vt
        cond = S[0] / max(S[-1], 1e-10)

        # Replace weight
        proj.weight.data = torch.tensor(W_unitary, dtype=torch.float32)

        replaced[pname] = {
            "cond": cond,
            "frobenius_error": np.linalg.norm(W - W_unitary, "fro"),
            "relative_error": np.linalg.norm(W - W_unitary, "fro") / np.linalg.norm(W, "fro"),
            "max_sv": S[0],
            "min_sv": S[-1],
        }

    return replaced


# ===== Baseline =====
print("\n[Validate] === BASELINE (unmodified model) ===")
baseline_ppl = compute_perplexity(model, tokenizer, prompts)
print(f"  Perplexity: {baseline_ppl:.2f}")

baseline_text = generate_text(model, tokenizer, prompts[0])
print(f"  Generated: {baseline_text[:100]}...")

# ===== Test each layer individually =====
print("\n[Validate] === SINGLE LAYER UNITARY REPLACEMENT ===")
for layer_idx in range(model.config.num_layers):
    # Reload model fresh
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model.eval()

    info = replace_with_unitary(model, layer_idx)
    ppl = compute_perplexity(model, tokenizer, prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

    avg_cond = np.mean([v["cond"] for v in info.values()])
    avg_rel_err = np.mean([v["relative_error"] for v in info.values()])

    print(f"  Layer {layer_idx}: PPL={ppl:.2f} (delta={delta:+.1f}%), "
          f"avg_cond={avg_cond:.0f}, avg_rel_err={avg_rel_err:.2f}")

# ===== Test best layer (6) — W_V only =====
print("\n[Validate] === LAYER 6 W_V ONLY ===")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
info = replace_with_unitary(model, 6, ["v_proj"])
ppl = compute_perplexity(model, tokenizer, prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, prompts[0])
print(f"  PPL={ppl:.2f} (delta={delta:+.1f}%), cond={info['v_proj']['cond']:.0f}")
print(f"  Generated: {text[:100]}...")

# ===== Test ALL layers replaced =====
print("\n[Validate] === ALL 8 LAYERS UNITARY (32 matrices) ===")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
for layer_idx in range(model.config.num_layers):
    replace_with_unitary(model, layer_idx)

ppl = compute_perplexity(model, tokenizer, prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, prompts[0])
print(f"  PPL={ppl:.2f} (delta={delta:+.1f}%)")
print(f"  Generated: {text[:100]}...")

# ===== Test W_V only across all layers =====
print("\n[Validate] === W_V ONLY across ALL 8 layers ===")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
for layer_idx in range(model.config.num_layers):
    replace_with_unitary(model, layer_idx, ["v_proj"])

ppl = compute_perplexity(model, tokenizer, prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, prompts[0])
print(f"  PPL={ppl:.2f} (delta={delta:+.1f}%)")
print(f"  Generated: {text[:100]}...")

print("\n[Validate] Done!")
