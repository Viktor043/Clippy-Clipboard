"""
Gradient check: compare adjoint-method gradients to numerical finite differences.
This verifies the backward pass is computing correct gradients.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from photonic_twin import (
    build_propagator, encode_input, decode_output, get_detector_masks,
    forward, backward, output_loss_gradient, compute_loss,
    WAVELENGTH, N_BASE, DN_MAX, VOXEL_PITCH, DZ
)

np.random.seed(42)

# Small grid for fast checking
nx, ny, nz = 16, 16, 4
n_inputs = 2
n_outputs = 2
k0 = 2.0 * np.pi / WAVELENGTH

propagator = build_propagator(nx, ny, VOXEL_PITCH, VOXEL_PITCH, DZ, WAVELENGTH, N_BASE)
masks = get_detector_masks(nx, ny, n_outputs)

# Random voxel grid
voxel_grid = np.random.uniform(0, DN_MAX, size=(nx, ny, nz))

# Random input
input_vec = np.array([0.7, 0.3])
input_field = encode_input(input_vec, nx, ny, n_inputs)
target_class = 0

# --- Analytic gradient (adjoint method) ---
output_field, fields_cache = forward(input_field, voxel_grid, propagator, k0, DZ)
output_intensities = decode_output(output_field, nx, ny, n_outputs, masks=masks)

loss, ce = compute_loss(output_intensities, target_class, n_outputs, voxel_grid, 0.0)

loss_grad_field = output_loss_gradient(
    output_field, nx, ny, n_outputs, output_intensities, target_class, masks=masks
)
analytic_grad = backward(loss_grad_field, voxel_grid, fields_cache, propagator, k0, DZ)

print(f"Forward pass — output intensities: {output_intensities}")
print(f"Loss: {loss:.6f}, CE: {ce:.6f}")
print(f"Analytic grad stats: min={analytic_grad.min():.2e}, max={analytic_grad.max():.2e}, "
      f"mean={analytic_grad.mean():.2e}, std={analytic_grad.std():.2e}")
print(f"Analytic grad nonzero fraction: {np.mean(np.abs(analytic_grad) > 1e-15):.1%}")
print()

# --- Numerical gradient (finite differences) ---
eps = 1e-6
n_checks = 50  # Check random voxels

print(f"Checking {n_checks} random voxels with eps={eps}:")
print(f"{'idx':>15s}  {'analytic':>12s}  {'numerical':>12s}  {'ratio':>10s}  {'match':>6s}")
print("-" * 65)

matches = 0
for check in range(n_checks):
    ix = np.random.randint(0, nx)
    iy = np.random.randint(0, ny)
    iz = np.random.randint(0, nz)

    # Forward at grid + eps
    grid_plus = voxel_grid.copy()
    grid_plus[ix, iy, iz] += eps
    out_plus, _ = forward(input_field, grid_plus, propagator, k0, DZ)
    I_plus = decode_output(out_plus, nx, ny, n_outputs, masks=masks)
    loss_plus, _ = compute_loss(I_plus, target_class, n_outputs, grid_plus, 0.0)

    # Forward at grid - eps
    grid_minus = voxel_grid.copy()
    grid_minus[ix, iy, iz] -= eps
    out_minus, _ = forward(input_field, grid_minus, propagator, k0, DZ)
    I_minus = decode_output(out_minus, nx, ny, n_outputs, masks=masks)
    loss_minus, _ = compute_loss(I_minus, target_class, n_outputs, grid_minus, 0.0)

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    ana_grad = analytic_grad[ix, iy, iz]

    if abs(num_grad) > 1e-10:
        ratio = ana_grad / num_grad
        match = 0.5 < abs(ratio) < 2.0
    elif abs(ana_grad) < 1e-10:
        ratio = 1.0
        match = True
    else:
        ratio = float('inf')
        match = False

    if match:
        matches += 1

    if check < 20 or not match:  # Print first 20 + all mismatches
        print(f"({ix:2d},{iy:2d},{iz:1d})  {ana_grad:12.4e}  {num_grad:12.4e}  {ratio:10.4f}  {'OK' if match else 'FAIL'}")

print(f"\nGradient check: {matches}/{n_checks} voxels match ({matches/n_checks:.0%})")
if matches / n_checks > 0.8:
    print("PASS — Adjoint gradients are correct")
else:
    print("FAIL — Adjoint gradients have errors")

# Also check gradient magnitudes relative to parameter scale
print(f"\n--- Scale analysis ---")
print(f"k0 = {k0:.2e} rad/m")
print(f"k0 * DN_MAX * DZ = {k0 * DN_MAX * DZ:.4f} rad (max phase per slice)")
print(f"Total max phase over {nz} slices: {k0 * DN_MAX * DZ * nz:.2f} rad")
print(f"Grid value range: [{voxel_grid.min():.6f}, {voxel_grid.max():.6f}]")
print(f"Gradient magnitude: {np.abs(analytic_grad).mean():.2e}")
print(f"LR * grad magnitude: {0.002 * np.abs(analytic_grad).mean():.2e}")
print(f"Gradient / DN_MAX: {np.abs(analytic_grad).mean() / DN_MAX:.2e}")
