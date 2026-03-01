#!/usr/bin/env python3
"""
Create normalized versions of weight matrices where max singular value = 1.0.
This makes them more suitable for glass regression (glass with absorption can
handle singular value ratios but not extreme overall scaling).
"""
import os
import numpy as np

weightdir = os.path.expanduser("~/Documents/photonic-twin/data/weights")

for layer_idx in range(8):
    for proj in ["W_Q", "W_K", "W_V", "W_O"]:
        name = f"layer{layer_idx}_{proj}"
        raw_path = os.path.join(weightdir, f"{name}_raw.bin")

        if not os.path.exists(raw_path):
            continue

        W = np.fromfile(raw_path, dtype=np.float32).reshape(64, 64)
        U, S, Vt = np.linalg.svd(W, full_matrices=True)

        # Normalize so max singular value = 1.0
        W_normalized = W / S[0]
        cond = S[0] / max(S[-1], 1e-10)
        sv_ratio = S[-1] / S[0]  # min/max after normalization

        # Save normalized version
        norm_path = os.path.join(weightdir, f"{name}_normalized.bin")
        W_normalized.astype(np.float32).tofile(norm_path)

        print(f"{name}: σ_max={S[0]:.4f}, σ_min={S[-1]:.6f}, "
              f"cond={cond:.1f}, sv_min_norm={sv_ratio:.6f}")

print(f"\nNormalized weights saved to {weightdir}")
print("After normalization, max singular value = 1.0 for all matrices")
