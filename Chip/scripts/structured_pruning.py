#!/usr/bin/env python3
"""
Structured pruning: which attention heads and layers can we remove?

For each head/layer, measure importance by:
1. Remove it and measure PPL increase
2. Heads with minimal PPL impact can be pruned

This directly reduces glass block count.
TinyStories-1M: 8 layers x 16 heads = 128 attention heads
Each head uses 4 projection matrices (Q,K,V,O) but they're combined.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "roneneldan/TinyStories-1M"
print("[Pruning] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_prompts = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "She liked to play with her",
    "One day, a boy found a",
    "The sun was shining and the birds",
    "He was very happy because",
    "The dog ran after the",
    "She found a beautiful flower in the",
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


# Baseline
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
print(f"\n[Baseline] PPL={baseline_ppl:.4f}")

# === TEST 1: Remove entire layers ===
print("\n[Pruning] === LAYER IMPORTANCE (remove one layer at a time) ===")
print(f"{'Layer':>6s} {'PPL':>10s} {'Delta%':>10s} {'Impact':>12s}")
print("-" * 42)

layer_impacts = []
for layer_idx in range(8):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Zero out the layer's contribution (set attention output + MLP to identity/zero)
    # Simplest: set output projection weights to zero
    with torch.no_grad():
        # Zero the attention output
        model.transformer.h[layer_idx].attn.attention.out_proj.weight.zero_()
        model.transformer.h[layer_idx].attn.attention.out_proj.bias.zero_()
        # Zero the MLP output
        model.transformer.h[layer_idx].mlp.c_proj.weight.zero_()
        model.transformer.h[layer_idx].mlp.c_proj.bias.zero_()

    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    impact = "LOW" if delta < 5 else "MEDIUM" if delta < 20 else "HIGH"
    layer_impacts.append((layer_idx, ppl, delta))
    print(f"  L{layer_idx:d}     {ppl:>10.2f} {delta:>+9.2f}% {impact:>12s}")

# === TEST 2: Remove pairs of layers ===
print("\n[Pruning] === BEST LAYER PAIRS TO REMOVE ===")
# Sort by least impact
layer_impacts.sort(key=lambda x: x[2])
print(f"  Least important layers: L{layer_impacts[0][0]} ({layer_impacts[0][2]:+.1f}%), "
      f"L{layer_impacts[1][0]} ({layer_impacts[1][2]:+.1f}%)")

# Test removing the two least important layers
l1, l2 = layer_impacts[0][0], layer_impacts[1][0]
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
with torch.no_grad():
    for li in [l1, l2]:
        model.transformer.h[li].attn.attention.out_proj.weight.zero_()
        model.transformer.h[li].attn.attention.out_proj.bias.zero_()
        model.transformer.h[li].mlp.c_proj.weight.zero_()
        model.transformer.h[li].mlp.c_proj.bias.zero_()
ppl = compute_perplexity(model, tokenizer, eval_prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, eval_prompts[0])
print(f"  Remove L{l1}+L{l2}: PPL={ppl:.2f} ({delta:+.1f}%)")
print(f"  Text: {text[:80]}...")

# Try removing 3 least important
l3 = layer_impacts[2][0]
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
with torch.no_grad():
    for li in [l1, l2, l3]:
        model.transformer.h[li].attn.attention.out_proj.weight.zero_()
        model.transformer.h[li].attn.attention.out_proj.bias.zero_()
        model.transformer.h[li].mlp.c_proj.weight.zero_()
        model.transformer.h[li].mlp.c_proj.bias.zero_()
ppl = compute_perplexity(model, tokenizer, eval_prompts)
delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
text = generate_text(model, tokenizer, eval_prompts[0])
print(f"  Remove L{l1}+L{l2}+L{l3}: PPL={ppl:.2f} ({delta:+.1f}%)")
print(f"  Text: {text[:80]}...")

# === TEST 3: Remove MLP only (keep attention) ===
print("\n[Pruning] === MLP-ONLY REMOVAL (keep attention, remove MLP) ===")
print(f"{'Layer':>6s} {'PPL':>10s} {'Delta%':>10s}")
print("-" * 30)

for layer_idx in range(8):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        model.transformer.h[layer_idx].mlp.c_proj.weight.zero_()
        model.transformer.h[layer_idx].mlp.c_proj.bias.zero_()
    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"  L{layer_idx:d}     {ppl:>10.2f} {delta:>+9.2f}%")

# === TEST 4: Attention-only removal (keep MLP) ===
print("\n[Pruning] === ATTENTION-ONLY REMOVAL (keep MLP, remove attention) ===")
print(f"{'Layer':>6s} {'PPL':>10s} {'Delta%':>10s}")
print("-" * 30)

for layer_idx in range(8):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        model.transformer.h[layer_idx].attn.attention.out_proj.weight.zero_()
        model.transformer.h[layer_idx].attn.attention.out_proj.bias.zero_()
    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"  L{layer_idx:d}     {ppl:>10.2f} {delta:>+9.2f}%")

# === GLASS SAVINGS SUMMARY ===
print("\n[Pruning] === GLASS SAVINGS SUMMARY ===")
print(f"  Full model: 96 glass blocks (128 with tiling), 8 layers")
print(f"  Per layer: 12 glass blocks (8 attn + 4 MLP)")
print(f"  Remove 1 layer: saves 12 blocks, ~16.4 cm glass")
print(f"  Remove 2 layers: saves 24 blocks, ~32.8 cm glass")
print(f"  Remove 3 layers: saves 36 blocks, ~49.2 cm glass")
print(f"  5-layer model: 60 blocks (80 tiling), 98.4 cm glass")
