#!/usr/bin/env python3
"""
Extract SVD components (U, S, V^T) of each attention weight matrix.
For the optical inference pipeline:
  y = W @ x = U @ diag(S) @ V^T @ x
  - V^T and U are unitary → glass blocks (proven at 100dB)
  - S is diagonal → electronic scaling (trivial)

Also validates that the decomposition is exact.
"""
import os
import numpy as np

weightdir = os.path.expanduser("~/Documents/photonic-twin/data/weights")

# Process all layer 6 attention matrices (our target layer)
for proj in ["W_Q", "W_K", "W_V", "W_O"]:
    name = f"layer6_{proj}"
    raw_path = os.path.join(weightdir, f"{name}_raw.bin")

    W = np.fromfile(raw_path, dtype=np.float32).reshape(64, 64)
    U, S, Vt = np.linalg.svd(W, full_matrices=True)

    # Verify: W ≈ U @ diag(S) @ Vt
    W_recon = U @ np.diag(S) @ Vt
    recon_err = np.max(np.abs(W - W_recon))

    # Verify U and Vt are unitary
    U_orth_err = np.max(np.abs(U.T @ U - np.eye(64)))
    Vt_orth_err = np.max(np.abs(Vt @ Vt.T - np.eye(64)))

    print(f"\n{name}:")
    print(f"  σ_max={S[0]:.6f}, σ_min={S[-1]:.6f}, cond={S[0]/max(S[-1],1e-10):.1f}")
    print(f"  Reconstruction error: {recon_err:.2e}")
    print(f"  U orthogonality error: {U_orth_err:.2e}")
    print(f"  V^T orthogonality error: {Vt_orth_err:.2e}")

    # Save components
    U.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_U.bin"))
    S.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_S.bin"))
    Vt.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_Vt.bin"))
    print(f"  Saved: {name}_U.bin, {name}_S.bin, {name}_Vt.bin")

    # End-to-end validation with random vectors
    rng = np.random.RandomState(42)
    x_test = rng.randn(1000, 64).astype(np.float32)

    # Direct: y = W @ x^T
    y_direct = (W @ x_test.T).T  # (1000, 64)

    # SVD pipeline: y = U @ diag(S) @ Vt @ x^T
    step1 = (Vt @ x_test.T).T      # V^T @ x  (glass block 1)
    step2 = step1 * S[np.newaxis, :]  # diag(S) scaling (electronic)
    step3 = (U @ step2.T).T         # U @ result (glass block 2)

    pipeline_err = np.max(np.abs(y_direct - step3))
    cosine_sims = np.array([
        np.dot(y_direct[i], step3[i]) / (np.linalg.norm(y_direct[i]) * np.linalg.norm(step3[i]) + 1e-10)
        for i in range(min(100, len(x_test)))
    ])

    print(f"  Pipeline max error: {pipeline_err:.2e}")
    print(f"  Pipeline cosine similarity: {np.mean(cosine_sims):.6f}")

# Also extract layer 0 for comparison
print("\n\n=== ALSO EXTRACTING LAYER 0 (for multi-layer testing) ===")
for proj in ["W_Q", "W_K", "W_V", "W_O"]:
    name = f"layer0_{proj}"
    raw_path = os.path.join(weightdir, f"{name}_raw.bin")
    W = np.fromfile(raw_path, dtype=np.float32).reshape(64, 64)
    U, S, Vt = np.linalg.svd(W, full_matrices=True)

    U.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_U.bin"))
    S.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_S.bin"))
    Vt.astype(np.float32).tofile(os.path.join(weightdir, f"{name}_Vt.bin"))
    print(f"  {name}: cond={S[0]/max(S[-1],1e-10):.1f}, saved U, S, V^T")

print("\n[SVD] Done! All SVD components extracted.")
print("Pipeline: glass(V^T) → electronic(S) → glass(U) = exact W @ x")
