#!/usr/bin/env python3
"""
Extract MLP weight matrices from TinyStories-1M and decompose via SVD.

MLP in GPT-Neo:
  W_fc:   256×64  (c_fc.weight)     — expands hidden dim
  W_proj:  64×256 (c_proj.weight)   — projects back

SVD for rectangular:
  W_fc(256×64)  = U(256×256) @ S_padded(256×64)  @ V^T(64×64)
  W_proj(64×256) = U(64×64)  @ S_padded(64×256) @ V^T(256×256)

Only r=min(m,n)=64 singular values are nonzero.
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("[ExtractMLP] Loading TinyStories-1M...")
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
model.eval()

out_dir = os.path.expanduser("~/Documents/photonic-twin/data/weights")
os.makedirs(out_dir, exist_ok=True)

num_layers = model.config.num_layers
print(f"[ExtractMLP] Model has {num_layers} layers")

for layer_idx in range(num_layers):
    mlp = model.transformer.h[layer_idx].mlp

    # c_fc: Linear(64, 256) → weight is (256, 64)
    W_fc = mlp.c_fc.weight.data.numpy()
    # c_proj: Linear(256, 64) → weight is (64, 256)
    W_proj = mlp.c_proj.weight.data.numpy()

    print(f"\n  Layer {layer_idx}:")
    print(f"    W_fc:   {W_fc.shape}  (Frobenius={np.linalg.norm(W_fc, 'fro'):.4f})")
    print(f"    W_proj: {W_proj.shape} (Frobenius={np.linalg.norm(W_proj, 'fro'):.4f})")

    for name, W in [("W_fc", W_fc), ("W_proj", W_proj)]:
        m, n = W.shape

        # Full SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=True)
        r = min(m, n)  # rank = 64

        cond = S[0] / max(S[-1], 1e-10)

        print(f"    {name}: U{U.shape} S[{len(S)}] Vt{Vt.shape} cond={cond:.1f} "
              f"σ_max={S[0]:.4f} σ_min={S[r-1]:.6f}")

        # Verify reconstruction
        S_padded = np.zeros((m, n), dtype=np.float32)
        for i in range(r):
            S_padded[i, i] = S[i]
        W_recon = U @ S_padded @ Vt
        recon_err = np.max(np.abs(W - W_recon))
        print(f"    {name}: reconstruction max_err={recon_err:.2e}")

        # Save components
        prefix = f"layer{layer_idx}_{name}"

        # Raw weight
        W.astype(np.float32).tofile(os.path.join(out_dir, f"{prefix}_raw.bin"))

        # SVD components (full-size for glass)
        U.astype(np.float32).tofile(os.path.join(out_dir, f"{prefix}_U.bin"))
        S.astype(np.float32).tofile(os.path.join(out_dir, f"{prefix}_S.bin"))
        Vt.astype(np.float32).tofile(os.path.join(out_dir, f"{prefix}_Vt.bin"))

print(f"\n[ExtractMLP] Saved all MLP SVD components to {out_dir}")

# Summary: total glass blocks needed
print(f"\n[ExtractMLP] === ARCHITECTURE SUMMARY ===")
print(f"  Attention (per layer): 8 glass blocks (4 matrices × 2 blocks, all 64×64)")
print(f"  MLP (per layer):       4 glass blocks (2 matrices × 2 blocks)")
print(f"    W_fc:   glass(V^T, 64×64) → E(S, 64) → pad → glass(U, 256×256)")
print(f"    W_proj: glass(V^T, 256×256) → E(S, 64) → glass(U, 64×64)")
print(f"  Per layer total: 12 glass blocks")
print(f"  Full model ({num_layers} layers): {num_layers * 12} glass blocks")
print(f"    64×64 blocks:  {num_layers * 10}")
print(f"    256×256 blocks: {num_layers * 2}")
