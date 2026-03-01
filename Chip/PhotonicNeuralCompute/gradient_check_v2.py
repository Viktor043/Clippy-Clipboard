"""
Detailed gradient check: test each component of the adjoint method separately.
1. Propagation adjoint
2. Phase screen adjoint
3. Single-slice gradient
4. Multi-slice gradient
"""
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from photonic_twin import (
    build_propagator, WAVELENGTH, N_BASE, DN_MAX, VOXEL_PITCH, DZ
)

np.random.seed(42)
nx, ny = 8, 8
k0 = 2 * np.pi / WAVELENGTH

H = build_propagator(nx, ny, VOXEL_PITCH, VOXEL_PITCH, DZ, WAVELENGTH, N_BASE)
H_conj = np.conj(H)

def inner(a, b):
    """Complex inner product <a, b> = sum(conj(a) * b)"""
    return np.sum(np.conj(a) * b)

# ============================================================
# Test 1: Propagation adjoint <Ax, y> = <x, A^H y>
# ============================================================
print("=" * 60)
print("TEST 1: Propagation adjoint correctness")
print("=" * 60)

x = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
y = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

# Forward: A(x) = ifft2(fft2(x) * H)
Ax = ifft2(fft2(x) * H)

# Adjoint: A^H(y) = ifft2(fft2(y) * conj(H))
AHy = ifft2(fft2(y) * H_conj)

lhs = inner(Ax, y)    # <Ax, y>
rhs = inner(x, AHy)   # <x, A^H y>

print(f"<Ax, y>   = {lhs}")
print(f"<x, A^Hy> = {rhs}")
print(f"Ratio: {lhs/rhs:.10f}")
print(f"PASS" if abs(lhs/rhs - 1) < 1e-10 else "FAIL")

# ============================================================
# Test 2: Phase screen adjoint
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Phase screen adjoint correctness")
print("=" * 60)

phase = np.random.randn(nx, ny) * 0.5
a = np.exp(1j * phase)

x = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
y = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

# Forward: B(x) = x * a
Bx = x * a

# Adjoint: B^H(y) = y * conj(a)
BHy = y * np.conj(a)

lhs = inner(Bx, y)
rhs = inner(x, BHy)

print(f"<Bx, y>   = {lhs}")
print(f"<x, B^Hy> = {rhs}")
print(f"Ratio: {lhs/rhs:.10f}")
print(f"PASS" if abs(lhs/rhs - 1) < 1e-10 else "FAIL")

# ============================================================
# Test 3: Single voxel gradient via single-slice BPM
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Single-slice voxel gradient")
print("=" * 60)

dn_grid = np.random.uniform(0, DN_MAX, (nx, ny))
input_field = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)

def single_slice_forward(field, dn):
    """Phase screen + propagation for one slice."""
    phase = k0 * dn * DZ
    g = field * np.exp(1j * phase)
    h = ifft2(fft2(g) * H)
    return h

def total_intensity_loss(field):
    """Simple scalar loss: total intensity (easy gradient)."""
    return np.sum(np.abs(field)**2).real

# Forward
output = single_slice_forward(input_field, dn_grid)
loss = total_intensity_loss(output)

# Analytic gradient via adjoint
# dL/dh* = h  (since L = sum |h|^2, dL/d|h|^2 = 1, dL/dh* = 1 * h = h)
adj_h = output.copy()

# Reverse propagation
adj_g = ifft2(fft2(adj_h) * H_conj)

# Phase screen values
phase = k0 * dn_grid * DZ
g = input_field * np.exp(1j * phase)

# Gradient: dL/d(dn) = 2 * Re(conj(adj_g) * i*k0*dz * g)
analytic_grad = 2.0 * np.real(np.conj(adj_g) * 1j * k0 * DZ * g)

# Numerical gradient (check a few voxels)
eps = 1e-7
print(f"{'voxel':>10s}  {'analytic':>12s}  {'numerical':>12s}  {'ratio':>10s}")
print("-" * 50)

n_match = 0
for _ in range(30):
    ix, iy = np.random.randint(0, nx), np.random.randint(0, ny)

    dn_plus = dn_grid.copy(); dn_plus[ix, iy] += eps
    dn_minus = dn_grid.copy(); dn_minus[ix, iy] -= eps

    loss_plus = total_intensity_loss(single_slice_forward(input_field, dn_plus))
    loss_minus = total_intensity_loss(single_slice_forward(input_field, dn_minus))

    num_grad = (loss_plus - loss_minus) / (2 * eps)
    ana_grad = analytic_grad[ix, iy]

    if abs(num_grad) > 1e-12:
        ratio = ana_grad / num_grad
    else:
        ratio = 1.0 if abs(ana_grad) < 1e-12 else float('inf')

    match = 0.99 < abs(ratio) < 1.01
    if match:
        n_match += 1

    print(f"({ix:1d},{iy:1d})  {ana_grad:12.6e}  {num_grad:12.6e}  {ratio:10.6f}  {'OK' if match else 'FAIL'}")

print(f"\nSingle-slice: {n_match}/30 match")
print("PASS" if n_match >= 28 else "FAIL")

# ============================================================
# Test 4: Multi-slice gradient (2 slices)
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: Multi-slice (2 slices) voxel gradient")
print("=" * 60)

nz = 2
dn_3d = np.random.uniform(0, DN_MAX, (nx, ny, nz))

def multi_slice_forward(field, dn_3d):
    """Full BPM forward pass."""
    nz = dn_3d.shape[2]
    cache = [field.copy()]
    for z in range(nz):
        phase = k0 * dn_3d[:, :, z] * DZ
        field = field * np.exp(1j * phase)
        field = ifft2(fft2(field) * H)
        cache.append(field.copy())
    return field, cache

def multi_loss(field):
    return np.sum(np.abs(field)**2).real

# Forward
output, cache = multi_slice_forward(input_field, dn_3d)
loss = multi_loss(output)

# Backward (adjoint)
adj = output.copy()  # dL/dh* = h for L = sum|h|^2

grad_3d = np.zeros_like(dn_3d)
for z in range(nz - 1, -1, -1):
    # Reverse propagation
    adj = ifft2(fft2(adj) * H_conj)

    # Gradient
    phase = k0 * dn_3d[:, :, z] * DZ
    g = cache[z] * np.exp(1j * phase)
    grad_3d[:, :, z] = 2.0 * np.real(np.conj(adj) * 1j * k0 * DZ * g)

    # Reverse phase screen
    adj = adj * np.exp(-1j * phase)

# Numerical gradient
print(f"{'voxel':>12s}  {'analytic':>12s}  {'numerical':>12s}  {'ratio':>10s}")
print("-" * 55)

n_match = 0
for _ in range(40):
    ix, iy, iz = np.random.randint(0, nx), np.random.randint(0, ny), np.random.randint(0, nz)

    dn_plus = dn_3d.copy(); dn_plus[ix, iy, iz] += eps
    dn_minus = dn_3d.copy(); dn_minus[ix, iy, iz] -= eps

    out_p, _ = multi_slice_forward(input_field, dn_plus)
    out_m, _ = multi_slice_forward(input_field, dn_minus)

    num_grad = (multi_loss(out_p) - multi_loss(out_m)) / (2 * eps)
    ana_grad = grad_3d[ix, iy, iz]

    if abs(num_grad) > 1e-12:
        ratio = ana_grad / num_grad
    else:
        ratio = 1.0 if abs(ana_grad) < 1e-12 else float('inf')

    match = 0.99 < abs(ratio) < 1.01
    if match:
        n_match += 1

    print(f"({ix:1d},{iy:1d},{iz:1d})  {ana_grad:12.6e}  {num_grad:12.6e}  {ratio:10.6f}  {'OK' if match else 'FAIL'}")

print(f"\nMulti-slice: {n_match}/40 match")
print("PASS" if n_match >= 38 else "FAIL")

# ============================================================
# Test 5: Full pipeline with detector readout + CE loss
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Full pipeline (detectors + CE loss)")
print("=" * 60)

from photonic_twin import (
    encode_input, decode_output, get_detector_masks,
    forward, backward, output_loss_gradient, compute_loss
)

n_inputs, n_outputs = 2, 2
voxel_grid = np.random.uniform(0, DN_MAX, (nx, ny, nz))
masks = get_detector_masks(nx, ny, n_outputs)
input_vec = np.array([0.7, 0.3])
input_field = encode_input(input_vec, nx, ny, n_inputs)
target_class = 0

# Forward
out_field, cache = forward(input_field, voxel_grid, H, k0, DZ)
I = decode_output(out_field, nx, ny, n_outputs, masks=masks)

def full_loss(grid):
    of, _ = forward(input_field, grid, H, k0, DZ)
    intensities = decode_output(of, nx, ny, n_outputs, masks=masks)
    l, _ = compute_loss(intensities, target_class, n_outputs, grid, 0.0)
    return l

loss = full_loss(voxel_grid)

# Analytic gradient
grad_field = output_loss_gradient(out_field, nx, ny, n_outputs, I, target_class, masks=masks)
grad_grid = backward(grad_field, voxel_grid, cache, H, k0, DZ)

# Numerical
print(f"{'voxel':>12s}  {'analytic':>12s}  {'numerical':>12s}  {'ratio':>10s}")
print("-" * 55)

n_match = 0
for _ in range(40):
    ix, iy, iz = np.random.randint(0, nx), np.random.randint(0, ny), np.random.randint(0, nz)

    g_plus = voxel_grid.copy(); g_plus[ix, iy, iz] += eps
    g_minus = voxel_grid.copy(); g_minus[ix, iy, iz] -= eps

    num_grad = (full_loss(g_plus) - full_loss(g_minus)) / (2 * eps)
    ana_grad = grad_grid[ix, iy, iz]

    if abs(num_grad) > 1e-10:
        ratio = ana_grad / num_grad
    else:
        ratio = 1.0 if abs(ana_grad) < 1e-10 else float('inf')

    match = 0.95 < abs(ratio) < 1.05
    if match:
        n_match += 1

    print(f"({ix:1d},{iy:1d},{iz:1d})  {ana_grad:12.6e}  {num_grad:12.6e}  {ratio:10.6f}  {'OK' if match else 'FAIL'}")

print(f"\nFull pipeline: {n_match}/40 match")
print("PASS" if n_match >= 35 else "FAIL")
