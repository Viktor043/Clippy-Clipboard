#!/usr/bin/env python3
"""
Test manufacturing tolerance for the FULL SVD pipeline (attention + MLP).

Adds Gaussian noise to U and V^T matrices at various PSNR levels to simulate
glass manufacturing precision. Tests all 48 matrices simultaneously.
"""
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[PrecFull] Loading TinyStories-1M...")
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


class NoisySVDLinear(nn.Module):
    """SVD linear with noise added to U and V^T to simulate glass imprecision."""
    def __init__(self, U, S, Vt, bias=None, psnr_db=100):
        super().__init__()
        # Add noise to U and Vt (glass blocks)
        U_noisy = self._add_noise(U, psnr_db)
        Vt_noisy = self._add_noise(Vt, psnr_db)

        self.U = nn.Parameter(torch.tensor(U_noisy, dtype=torch.float32), requires_grad=False)
        self.S = nn.Parameter(torch.tensor(S, dtype=torch.float32), requires_grad=False)
        self.Vt = nn.Parameter(torch.tensor(Vt_noisy, dtype=torch.float32), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=False)
        else:
            self.bias = None
        self.in_features = Vt.shape[1]
        self.out_features = U.shape[0]
        self.r = len(S)

    @staticmethod
    def _add_noise(M, psnr_db):
        if psnr_db >= 100:
            return M
        signal_power = np.mean(M ** 2)
        noise_power = signal_power / (10 ** (psnr_db / 10))
        noise = np.random.randn(*M.shape).astype(np.float32) * np.sqrt(noise_power)
        return (M + noise).astype(np.float32)

    @property
    def weight(self):
        S_diag = torch.zeros(self.out_features, self.in_features, device=self.U.device)
        for i in range(self.r):
            S_diag[i, i] = self.S[i]
        return self.U @ S_diag @ self.Vt

    def forward(self, x):
        z = x @ self.Vt.T
        z_scaled = z[..., :self.r] * self.S
        if self.out_features > self.r:
            pad = torch.zeros(*z.shape[:-1], self.out_features - self.r, device=z.device)
            z_padded = torch.cat([z_scaled, pad], dim=-1)
        else:
            z_padded = z_scaled
        out = z_padded @ self.U.T
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_all_with_noisy_svd(model, psnr_db):
    """Replace all matrices with noisy SVD at given PSNR."""
    for layer_idx in range(model.config.num_layers):
        attn = model.transformer.h[layer_idx].attn.attention
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(attn, pname)
            W = proj.weight.data.numpy()
            bias = proj.bias.data.numpy() if proj.bias is not None else None
            U, S, Vt = np.linalg.svd(W, full_matrices=True)
            setattr(attn, pname, NoisySVDLinear(U, S, Vt, bias, psnr_db))

        mlp = model.transformer.h[layer_idx].mlp

        W_fc = mlp.c_fc.weight.data.numpy()
        bias_fc = mlp.c_fc.bias.data.numpy() if mlp.c_fc.bias is not None else None
        U_fc, S_fc, Vt_fc = np.linalg.svd(W_fc, full_matrices=True)
        mlp.c_fc = NoisySVDLinear(U_fc, S_fc, Vt_fc, bias_fc, psnr_db)

        W_proj = mlp.c_proj.weight.data.numpy()
        bias_proj = mlp.c_proj.bias.data.numpy() if mlp.c_proj.bias is not None else None
        U_proj, S_proj, Vt_proj = np.linalg.svd(W_proj, full_matrices=True)
        mlp.c_proj = NoisySVDLinear(U_proj, S_proj, Vt_proj, bias_proj, psnr_db)


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
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
baseline_ppl = compute_perplexity(model, tokenizer, eval_prompts)
baseline_text = generate_text(model, tokenizer, eval_prompts[0])
print(f"\n[Baseline] PPL={baseline_ppl:.4f}")
print(f"[Baseline] Text: {baseline_text[:80]}...")

# === Test each PSNR level ===
psnr_levels = [100, 80, 60, 50, 40, 30, 20, 10]

print(f"\n[PrecFull] === MANUFACTURING TOLERANCE (48 matrices, attn+MLP) ===")
print(f"{'PSNR':>6s} {'PPL':>10s} {'Delta':>10s} {'Text Sample':>60s}")
print("-" * 90)

for psnr_db in psnr_levels:
    np.random.seed(42)  # reproducible noise
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    replace_all_with_noisy_svd(model, psnr_db)

    ppl = compute_perplexity(model, tokenizer, eval_prompts)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    text = generate_text(model, tokenizer, eval_prompts[0])

    # Check text match
    text_status = "MATCH" if text == baseline_text else text[:50]
    print(f"{psnr_db:>4d}dB {ppl:>10.2f} {delta:>+9.2f}% {text_status}")

print(f"\n[PrecFull] Baseline PPL: {baseline_ppl:.4f}")
print(f"[PrecFull] 48 matrices = 96 glass blocks + 384 electronic scalings")
print(f"[PrecFull] Note: more matrices = more noise accumulation vs attention-only test")
