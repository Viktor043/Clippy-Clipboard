#!/usr/bin/env python3
"""
Test how glass precision (noise in U and V^T) affects model perplexity.
This determines manufacturing tolerance requirements.

We add Gaussian noise to U and V^T matrices to simulate different
glass precision levels (PSNR in dB).
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

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
            total_loss += outputs.loss.item() * (inputs["input_ids"].shape[1] - 1)
            total_tokens += inputs["input_ids"].shape[1] - 1
    return np.exp(total_loss / total_tokens)


def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"], max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


class NoisySVDLinear(nn.Module):
    """SVD linear layer with controllable noise on U and V^T."""
    def __init__(self, W_numpy, psnr_db):
        super().__init__()
        U, S, Vt = np.linalg.svd(W_numpy, full_matrices=True)

        if psnr_db < 200:  # Add noise (200dB = effectively perfect)
            # PSNR = 10*log10(signal_power / noise_power)
            # noise_power = signal_power * 10^(-PSNR/10)
            # For unitary matrices, signal_power per element ≈ 1/N
            N = U.shape[0]
            signal_power = 1.0 / N  # average squared element of unitary matrix

            noise_power = signal_power * (10 ** (-psnr_db / 10))
            noise_std = np.sqrt(noise_power)

            rng = np.random.RandomState(42)
            U_noisy = U + rng.randn(*U.shape).astype(np.float32) * noise_std
            Vt_noisy = Vt + rng.randn(*Vt.shape).astype(np.float32) * noise_std
        else:
            U_noisy = U
            Vt_noisy = Vt

        self.U = nn.Parameter(torch.tensor(U_noisy, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt_noisy, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        z = x @ self.Vt.T
        z = z * self.S
        return z @ self.U.T


def replace_all_with_noisy_svd(model, psnr_db):
    for layer_idx in range(model.config.num_layers):
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data.numpy()
            bias = proj.bias.data.clone() if proj.bias is not None else None

            svd_layer = NoisySVDLinear(W, psnr_db)

            class SVDProj(nn.Module):
                def __init__(self, svd, b):
                    super().__init__()
                    self.svd = svd
                    self.bias = b
                    self.weight = svd.U
                    self.in_features = W.shape[1]
                    self.out_features = W.shape[0]

                def forward(self, x):
                    out = self.svd(x)
                    if self.bias is not None:
                        out = out + self.bias
                    return out

            setattr(attn, pname, SVDProj(svd_layer, bias))


print("[Precision] Loading TinyStories-1M...")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

# Baseline
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, prompts)
print(f"\nBaseline PPL: {baseline_ppl:.4f}")

# Test different glass precision levels
print("\n=== GLASS PRECISION vs PERPLEXITY ===")
print(f"{'PSNR (dB)':>10} {'PPL':>10} {'Delta%':>10} {'Status':>20}")
print("-" * 55)

for psnr_db in [100, 80, 60, 50, 40, 30, 20, 10]:
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    model.eval()
    replace_all_with_noisy_svd(model, psnr_db)
    ppl = compute_perplexity(model, tokenizer, prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100

    if delta < 1:
        status = "EXCELLENT"
    elif delta < 5:
        status = "GOOD"
    elif delta < 20:
        status = "ACCEPTABLE"
    elif delta < 100:
        status = "DEGRADED"
    else:
        status = "BROKEN"

    print(f"{psnr_db:>10} {ppl:>10.2f} {delta:>+10.2f}% {status:>20}")

    if psnr_db in [50, 30]:
        text = generate_text(model, tokenizer, prompts[0], max_new_tokens=40)
        print(f"          Generated: {text[:100]}...")

print("\n[Precision] Done!")
