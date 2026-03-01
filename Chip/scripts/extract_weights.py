#!/usr/bin/env python3
"""
Extract weight matrices from TinyStories-1M (GPT-Neo) and prepare for glass regression.

Computes:
1. Raw attention weight matrices (W_Q, W_K, W_V, W_O) — all 64×64
2. SVD decomposition (U, S, V^T) for each
3. Nearest unitary approximation (UV^T from polar decomposition)
4. Exports as flat Float32 binary for MatrixTransformDataGenerator

Usage:
    python3 scripts/extract_weights.py

Output:
    data/weights/layer0_W_V_raw.bin       — raw 64×64 weight matrix
    data/weights/layer0_W_V_unitary.bin   — nearest unitary (UV^T)
    data/weights/layer0_W_V_info.txt      — SVD stats, condition number
"""

import os
import sys
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import torch

print("[WeightExtract] Loading TinyStories-1M model...")
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()

print(f"[WeightExtract] Model config: hidden_size={model.config.hidden_size}, "
      f"num_heads={model.config.num_attention_heads}, "
      f"num_layers={model.config.num_layers}")

# Output directory
outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "weights")
os.makedirs(outdir, exist_ok=True)

def analyze_and_save(name: str, W: np.ndarray):
    """Analyze a weight matrix and save raw + nearest unitary."""
    n, m = W.shape
    print(f"\n[WeightExtract] === {name} ({n}×{m}) ===")

    # SVD
    U, S, Vt = np.linalg.svd(W, full_matrices=True)

    # Stats
    cond = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
    print(f"  Singular values: min={S[-1]:.6f}, max={S[0]:.6f}, "
          f"cond={cond:.2f}, ratio(max/min)={cond:.2f}")
    print(f"  Singular value distribution: {S[:5].tolist()}")
    print(f"  Frobenius norm: {np.linalg.norm(W, 'fro'):.4f}")

    # Nearest unitary: UV^T (polar decomposition)
    W_unitary = U @ Vt  # This is the closest orthogonal matrix to W

    # Reconstruction error
    recon_err = np.linalg.norm(W - W_unitary @ np.diag(S), 'fro')
    unitary_approx_err = np.linalg.norm(W - W_unitary, 'fro')
    print(f"  Unitary approx error: {unitary_approx_err:.6f}")
    print(f"  SVD reconstruction error: {recon_err:.2e}")

    # Verify orthogonality of UV^T
    orth_err = np.max(np.abs(W_unitary.T @ W_unitary - np.eye(n)))
    print(f"  Orthogonality check |Q^T Q - I|_max: {orth_err:.2e}")

    # Save raw matrix (row-major float32)
    raw_path = os.path.join(outdir, f"{name}_raw.bin")
    W.astype(np.float32).tofile(raw_path)
    print(f"  Saved raw: {raw_path} ({os.path.getsize(raw_path)} bytes)")

    # Save nearest unitary (row-major float32)
    unitary_path = os.path.join(outdir, f"{name}_unitary.bin")
    W_unitary.astype(np.float32).tofile(unitary_path)
    print(f"  Saved unitary: {unitary_path} ({os.path.getsize(unitary_path)} bytes)")

    # Save info file
    info_path = os.path.join(outdir, f"{name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Name: {name}\n")
        f.write(f"Shape: {n}×{m}\n")
        f.write(f"Condition number: {cond:.4f}\n")
        f.write(f"Singular values: {S.tolist()}\n")
        f.write(f"Frobenius norm: {np.linalg.norm(W, 'fro'):.6f}\n")
        f.write(f"Unitary approx error: {unitary_approx_err:.6f}\n")
        f.write(f"Orthogonality error: {orth_err:.2e}\n")
    print(f"  Saved info: {info_path}")

    return S

# Extract attention matrices from all layers
all_singular_values = {}

for layer_idx in range(model.config.num_layers):
    attn = model.transformer.h[layer_idx].attn.attention

    # Q, K, V, O projection weights
    for proj_name, proj in [("W_Q", attn.q_proj), ("W_K", attn.k_proj),
                             ("W_V", attn.v_proj), ("W_O", attn.out_proj)]:
        W = proj.weight.detach().numpy()
        name = f"layer{layer_idx}_{proj_name}"
        S = analyze_and_save(name, W)
        all_singular_values[name] = S

# Summary: which matrices are closest to unitary?
print("\n" + "=" * 60)
print("[WeightExtract] SUMMARY: Condition numbers by matrix")
print("=" * 60)
for name, S in sorted(all_singular_values.items()):
    cond = S[0] / S[-1] if S[-1] > 1e-10 else float('inf')
    # How much energy is lost when using unitary approximation
    energy_kept = np.sum(np.ones_like(S)) / len(S)  # always 1 for unitary
    sv_spread = np.std(S) / np.mean(S)
    print(f"  {name:25s}: cond={cond:8.2f}, σ_spread={sv_spread:.4f}, "
          f"max_S={S[0]:.4f}, min_S={S[-1]:.4f}")

# Find the matrix closest to unitary (lowest condition number)
best_name = min(all_singular_values.keys(),
                key=lambda k: all_singular_values[k][0] / max(all_singular_values[k][-1], 1e-10))
best_cond = all_singular_values[best_name][0] / all_singular_values[best_name][-1]
print(f"\n[WeightExtract] Best candidate for unitary glass: {best_name} (cond={best_cond:.2f})")
print(f"[WeightExtract] All weights saved to: {outdir}")
print("[WeightExtract] Done!")
