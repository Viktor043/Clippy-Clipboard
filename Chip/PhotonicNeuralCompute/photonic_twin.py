"""
Photonic Neural Compute — BPM Digital Twin Simulator
=====================================================
SNRGY Studios — February 2026

Beam Propagation Method (BPM) simulator for optimizing 3D refractive index
patterns in glass for diffractive neural network inference.

Physics:
  - Light propagates through z-slices of a voxel grid
  - Each slice: phase screen (element-wise) + free-space diffraction (FFT)
  - Adjoint method gives gradients for all voxels in one backward pass
  - Adam optimizer with Gumbel-softmax ternary annealing

Usage:
  python3 photonic_twin.py                    # default 96x96x20 grid
  python3 photonic_twin.py --nx 256 --ny 256 --nz 40 --steps 5000
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import argparse
import time
import json
from pathlib import Path


# =============================================================================
# Physical constants and defaults
# =============================================================================

WAVELENGTH = 1.55e-6       # 1550 nm (telecom C-band)
N_BASE = 1.45              # fused silica base refractive index
DN_MAX = 0.05              # max refractive index change from laser writing
VOXEL_PITCH = 2e-6         # 2 um voxel size (x, y)
DZ = 10e-6                 # 10 um slice thickness (z)

# Ternary levels: {0: no change, 1: +dn/2, 2: +dn_max}
# Maps to delta_n: {0.0, DN_MAX/2, DN_MAX}
TERNARY_DN = np.array([0.0, DN_MAX / 2.0, DN_MAX])


# =============================================================================
# Propagation kernel (precomputed once)
# =============================================================================

def build_propagator(nx, ny, dx, dy, dz, wavelength, n_base):
    """
    Free-space angular spectrum propagation kernel for one z-step.

    H(kx, ky) = exp(i * dz * sqrt(k^2 - kx^2 - ky^2))

    where k = 2*pi*n_base / wavelength.
    Evanescent modes (kx^2 + ky^2 > k^2) are zeroed out.
    """
    k = 2.0 * np.pi * n_base / wavelength

    # Spatial frequency grids
    kx = 2.0 * np.pi * fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Propagation phase
    kz_sq = k**2 - KX**2 - KY**2
    propagating = kz_sq > 0

    kz = np.zeros_like(kz_sq)
    kz[propagating] = np.sqrt(kz_sq[propagating])

    # Angular spectrum propagator
    H = np.zeros((nx, ny), dtype=np.complex128)
    H[propagating] = np.exp(1j * kz[propagating] * dz)

    return H


# =============================================================================
# I/O Encoding
# =============================================================================

def _io_positions(n_channels, nx, ny):
    """
    Compute 2D positions for input/output channels on the grid.

    Arranges channels in a grid pattern with good separation.
    Returns list of (cx, cy) center positions.
    """
    # Arrange in a roughly square grid layout
    n_cols = int(np.ceil(np.sqrt(n_channels)))
    n_rows = int(np.ceil(n_channels / n_cols))

    margin = 0.15  # 15% margin from edges
    positions = []
    for idx in range(n_channels):
        row = idx // n_cols
        col = idx % n_cols
        cx = int(nx * (margin + (1 - 2 * margin) * col / max(n_cols - 1, 1)))
        cy = int(ny * (margin + (1 - 2 * margin) * row / max(n_rows - 1, 1)))
        positions.append((cx, cy))

    return positions


def encode_input(input_vector, nx, ny, n_inputs):
    """
    Encode a discrete input vector as spatial modes on the input face.

    Each input channel gets a Gaussian spot at a specific spatial location.
    The amplitude of each spot is the input value. Channels are arranged
    in a 2D grid pattern for maximum separation.

    Args:
        input_vector: 1D array of input values (length n_inputs)
        nx, ny: field grid dimensions
        n_inputs: number of input channels

    Returns:
        Complex field array (nx, ny)
    """
    field = np.zeros((nx, ny), dtype=np.complex128)

    # Spot size scales with grid — ensures good spatial extent
    spot_sigma = max(nx / (3.0 * np.sqrt(n_inputs)), 3.0)

    positions = _io_positions(n_inputs, nx, ny)
    Y_grid, X_grid = np.mgrid[0:nx, 0:ny]

    for i, val in enumerate(input_vector):
        cx, cy = positions[i]
        r_sq = (X_grid - cx).astype(np.float64)**2 + (Y_grid - cy).astype(np.float64)**2
        spot = np.exp(-r_sq / (2.0 * spot_sigma**2))
        field += val * spot

    # Normalize total power to 1
    power = np.sum(np.abs(field)**2)
    if power > 0:
        field /= np.sqrt(power)

    return field


def get_detector_masks(nx, ny, n_outputs):
    """
    Precompute detector masks for output readout (cached per config).

    Returns list of boolean masks, one per output channel.
    """
    positions = _io_positions(n_outputs, nx, ny)
    detector_sigma = max(nx / (3.0 * np.sqrt(n_outputs)), 3.0)
    detector_radius = int(detector_sigma * 1.5)

    Y_grid, X_grid = np.mgrid[0:nx, 0:ny]
    masks = []
    for cx, cy in positions:
        r_sq = (X_grid - cx)**2 + (Y_grid - cy)**2
        masks.append(r_sq < detector_radius**2)

    return masks


def decode_output(field, nx, ny, n_outputs, masks=None):
    """
    Read output intensities from detector regions on the output face.

    Each output class has a detector region. The integrated intensity
    at each detector is the output logit for that class.

    Args:
        field: Complex field array (nx, ny) at output face
        nx, ny: field grid dimensions
        n_outputs: number of output channels
        masks: Precomputed detector masks (optional, for performance)

    Returns:
        1D array of output intensities (length n_outputs)
    """
    intensity = np.abs(field)**2
    outputs = np.zeros(n_outputs)

    if masks is None:
        masks = get_detector_masks(nx, ny, n_outputs)

    for i in range(n_outputs):
        outputs[i] = np.sum(intensity[masks[i]])

    return outputs


# =============================================================================
# Forward pass (BPM)
# =============================================================================

def forward(field, voxel_grid, propagator, k0, dz):
    """
    BPM forward propagation through the voxel grid.

    For each z-slice:
      1. Phase screen: field *= exp(i * k0 * delta_n[z] * dz)
      2. Free-space propagation: field = IFFT2(FFT2(field) * H)

    Args:
        field: Input complex field (nx, ny)
        voxel_grid: Refractive index changes (nx, ny, nz), continuous values
        propagator: Precomputed propagation kernel H (nx, ny)
        k0: Free-space wavenumber (2*pi/wavelength)
        dz: Slice thickness

    Returns:
        output_field: Output complex field (nx, ny)
        fields_cache: List of field states at each slice (for adjoint method)
    """
    nz = voxel_grid.shape[2]
    fields_cache = [field.copy()]  # Store field before each slice

    for z in range(nz):
        # Phase screen
        phase = k0 * voxel_grid[:, :, z] * dz
        field = field * np.exp(1j * phase)

        # Free-space propagation
        field = ifft2(fft2(field) * propagator)

        fields_cache.append(field.copy())

    return field, fields_cache


# =============================================================================
# Backward pass (adjoint method)
# =============================================================================

def backward(loss_grad_field, voxel_grid, fields_cache, propagator, k0, dz):
    """
    Adjoint-method backward pass for BPM gradient computation.

    Propagates the loss gradient backward through the same physics,
    computing gradients for every voxel simultaneously.

    Args:
        loss_grad_field: Gradient of loss w.r.t. output field (nx, ny)
        voxel_grid: Current refractive index changes (nx, ny, nz)
        fields_cache: Cached forward fields (from forward pass)
        propagator: Propagation kernel H (nx, ny)
        k0: Free-space wavenumber
        dz: Slice thickness

    Returns:
        grad_grid: Gradient of loss w.r.t. each voxel (nx, ny, nz)
    """
    nz = voxel_grid.shape[2]
    nx, ny = voxel_grid.shape[0], voxel_grid.shape[1]
    grad_grid = np.zeros_like(voxel_grid)

    adjoint = loss_grad_field.copy()

    # Conjugate propagator for backward propagation
    H_conj = np.conj(propagator)

    for z in range(nz - 1, -1, -1):
        # Reverse free-space propagation
        # ∂L/∂g* = IFFT2(FFT2(∂L/∂h*) * conj(H))
        adjoint = ifft2(fft2(adjoint) * H_conj)

        # Compute gradient for this z-slice
        # For real parameter dn: dL/d(dn) = 2 * Re(conj(∂L/∂g*) * ∂g/∂dn)
        # where ∂g/∂dn = i*k0*dz * g = i*k0*dz * f*exp(i*phi)
        # Note: conj(∂L/∂g*) = ∂L/∂g (the non-conjugate Wirtinger derivative)
        field_at_z = fields_cache[z]  # Field entering this slice (before phase screen)
        phase = k0 * voxel_grid[:, :, z] * dz
        field_after_phase = field_at_z * np.exp(1j * phase)

        grad_grid[:, :, z] = 2.0 * np.real(
            np.conj(adjoint) * (1j * k0 * dz) * field_after_phase
        )

        # Reverse phase screen: ∂L/∂f* = exp(-i*phi) * ∂L/∂g*
        adjoint = adjoint * np.exp(-1j * phase)

    return grad_grid


# =============================================================================
# Loss function
# =============================================================================

def compute_loss(output_intensities, target_class, n_outputs, voxel_grid, sparsity_lambda):
    """
    Classification loss: maximize correct detector, minimize others.

    Uses a softmax cross-entropy-like loss on raw detector intensities:
        loss = -log(softmax(intensities)[target_class])
             = -I_target + log(sum(exp(I_j)))

    Plus L1 sparsity penalty.

    Args:
        output_intensities: Raw detector intensities (n_outputs,)
        target_class: Integer target class index
        n_outputs: Number of output classes
        voxel_grid: Current voxel values (for sparsity penalty)
        sparsity_lambda: Sparsity weight

    Returns:
        loss: Scalar loss value
        ce: Cross-entropy component (for logging)
    """
    # Scale intensities for numerical stability
    I = output_intensities
    I_scaled = I - np.max(I)  # shift for numerical stability
    log_sum_exp = np.log(np.sum(np.exp(I_scaled)) + 1e-10) + np.max(I)
    ce = -I[target_class] + log_sum_exp

    sparsity = sparsity_lambda * np.mean(np.abs(voxel_grid))
    return ce + sparsity, ce


def output_loss_gradient(field, nx, ny, n_outputs, output_intensities, target_class, masks=None):
    """
    Compute gradient of cross-entropy loss w.r.t. the output field.

    For softmax CE loss:
        dL/dI_i = softmax(I)_i - delta(i, target)

    Then chain through detector readout:
        dL/d(field) = sum_i dL/dI_i * d(I_i)/d(field)
        where d(I_i)/d(field) = 2 * conj(field) * mask_i

    Args:
        field: Output complex field (nx, ny)
        nx, ny: Grid dimensions
        n_outputs: Number of output channels
        output_intensities: Raw detector intensities (n_outputs,)
        target_class: Integer target class index
        masks: Precomputed detector masks

    Returns:
        grad_field: Gradient w.r.t. output field (nx, ny complex)
    """
    if masks is None:
        masks = get_detector_masks(nx, ny, n_outputs)

    # Softmax of detector intensities
    I = output_intensities
    I_shifted = I - np.max(I)
    exp_I = np.exp(I_shifted)
    softmax_I = exp_I / (np.sum(exp_I) + 1e-10)

    # Propagate ∂L/∂h* (Wirtinger conjugate derivative)
    # For I = |h|^2: ∂I/∂h* = h, so ∂L/∂h* = (dL/dI) * h
    grad_field = np.zeros((nx, ny), dtype=np.complex128)

    for i in range(n_outputs):
        # dL/dI_i = softmax(I)_i - delta(i, target)
        dL_dI = softmax_I[i] - (1.0 if i == target_class else 0.0)

        # ∂L/∂h*[x,y] = (dL/dI_i) * h[x,y] * mask_i[x,y]
        grad_field[masks[i]] += dL_dI * field[masks[i]]

    return grad_field


# =============================================================================
# Adam optimizer
# =============================================================================

class AdamOptimizer:
    """Adam optimizer for voxel grid updates."""

    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape)  # First moment
        self.v = np.zeros(shape)  # Second moment
        self.t = 0

    def step(self, grid, grad, sparsity_lambda=0.0):
        """
        Perform one Adam update step.

        Args:
            grid: Current voxel grid values
            grad: Gradient from adjoint method
            sparsity_lambda: L1 sparsity weight (adds sign(grid) to gradient)

        Returns:
            Updated grid
        """
        self.t += 1

        # Add L1 sparsity gradient
        total_grad = grad + sparsity_lambda * np.sign(grid)

        # Adam moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * total_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * total_grad**2

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update
        grid = grid - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        # Clamp to valid range [0, DN_MAX]
        grid = np.clip(grid, 0.0, DN_MAX)

        return grid


# =============================================================================
# Ternary quantization (Gumbel-softmax annealing)
# =============================================================================

def ternary_anneal(grid, iteration, total_iterations, anneal_start=0.3, anneal_end=0.9):
    """
    Gradually push continuous voxel values toward ternary levels.

    Uses a soft quantization that strengthens over training:
    - Early training: values are free (continuous for smooth gradients)
    - Late training: values are pushed toward {0, DN_MAX/2, DN_MAX}

    Args:
        grid: Current continuous voxel grid (values in [0, DN_MAX])
        iteration: Current iteration number
        total_iterations: Total number of iterations
        anneal_start: Fraction of training when annealing begins
        anneal_end: Fraction of training when annealing is at full strength

    Returns:
        Annealed grid (still continuous, but closer to ternary)
    """
    progress = iteration / total_iterations

    if progress < anneal_start:
        return grid  # No annealing yet

    # Annealing strength: 0 at anneal_start, 1 at anneal_end
    strength = min(1.0, (progress - anneal_start) / (anneal_end - anneal_start))

    # Find nearest ternary level for each voxel
    levels = TERNARY_DN  # [0, DN_MAX/2, DN_MAX]

    # Distance to each level
    distances = np.stack([np.abs(grid - level) for level in levels], axis=-1)
    nearest_idx = np.argmin(distances, axis=-1)
    nearest_val = levels[nearest_idx]

    # Interpolate toward nearest ternary level
    grid = grid + strength * (nearest_val - grid)

    return grid


def hard_quantize(grid):
    """
    Hard quantization to exact ternary values.
    Applied after optimization is complete.

    Returns:
        quantized_grid: Values are exactly {0, DN_MAX/2, DN_MAX}
        ternary_indices: Integer indices {0, 1, 2} for each voxel
    """
    levels = TERNARY_DN
    distances = np.stack([np.abs(grid - level) for level in levels], axis=-1)
    ternary_indices = np.argmin(distances, axis=-1)
    quantized_grid = levels[ternary_indices]
    return quantized_grid, ternary_indices


# =============================================================================
# Reference model (simple MLP for training data)
# =============================================================================

class ReferenceModel:
    """
    Small reference MLP that the photonic glass block learns to replicate.

    This generates input-output pairs for the optimization:
    the glass block is trained to classify inputs the same way as this model.
    """

    def __init__(self, n_inputs, n_outputs, hidden_size=32, seed=42):
        rng = np.random.RandomState(seed)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Simple 2-layer MLP with ReLU
        self.W1 = rng.randn(n_inputs, hidden_size) * np.sqrt(2.0 / n_inputs)
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.randn(hidden_size, n_outputs) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(n_outputs)

    def predict(self, x):
        """Forward pass through reference MLP. Returns class index."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = h @ self.W2 + self.b2
        return int(np.argmax(out))

    def generate_batch(self, batch_size, rng=None):
        """Generate a batch of (input_vector, target_class) pairs."""
        if rng is None:
            rng = np.random.RandomState()

        inputs = []
        targets = []
        for _ in range(batch_size):
            x = rng.randn(self.n_inputs)
            x = np.abs(x)  # Intensities are non-negative
            x = x / (np.sum(x) + 1e-8)  # Normalize
            y = self.predict(x)
            inputs.append(x)
            targets.append(y)

        return inputs, targets


# =============================================================================
# Main optimization loop
# =============================================================================

def optimize(
    nx=96, ny=96, nz=20,
    n_inputs=4, n_outputs=4,
    n_steps=2000,
    lr=5e-4,
    sparsity_lambda=1e-4,
    batch_size=8,
    log_every=100,
    seed=42
):
    """
    Run the full BPM digital twin optimization.

    Optimizes a 3D voxel grid so that light propagating through it
    replicates the behavior of a reference neural network.

    Args:
        nx, ny: Transverse grid dimensions (pixels)
        nz: Number of z-slices (depth)
        n_inputs: Number of input channels
        n_outputs: Number of output classes
        n_steps: Number of optimization steps
        lr: Learning rate for Adam
        sparsity_lambda: L1 sparsity weight
        batch_size: Training samples per step
        log_every: Print progress every N steps
        seed: Random seed

    Returns:
        results: Dictionary with optimized grid, metrics, config
    """
    rng = np.random.RandomState(seed)

    print("=" * 60)
    print("PHOTONIC NEURAL COMPUTE — BPM Digital Twin Simulator")
    print("=" * 60)
    print(f"Grid:        {nx} x {ny} x {nz} voxels")
    print(f"Voxel pitch: {VOXEL_PITCH*1e6:.1f} um (xy), {DZ*1e6:.1f} um (z)")
    print(f"Physical:    {nx*VOXEL_PITCH*1e3:.1f} x {ny*VOXEL_PITCH*1e3:.1f} x {nz*DZ*1e3:.1f} mm")
    print(f"Wavelength:  {WAVELENGTH*1e9:.0f} nm")
    print(f"I/O:         {n_inputs} inputs -> {n_outputs} outputs")
    print(f"Optimizer:   Adam (lr={lr})")
    print(f"Steps:       {n_steps}")
    print(f"Sparsity:    lambda={sparsity_lambda}")
    print(f"Batch size:  {batch_size}")
    print("=" * 60)

    # --- Setup ---
    k0 = 2.0 * np.pi / WAVELENGTH

    # Precompute propagation kernel
    print("Building propagation kernel...", end=" ", flush=True)
    propagator = build_propagator(nx, ny, VOXEL_PITCH, VOXEL_PITCH, DZ, WAVELENGTH, N_BASE)
    print("done.")

    # Initialize voxel grid — uniform around DN_MAX/2 with noise
    # (centered initialization gives optimizer room to push in both directions)
    print("Initializing voxel grid...", end=" ", flush=True)
    voxel_grid = DN_MAX * 0.5 + rng.uniform(-DN_MAX * 0.2, DN_MAX * 0.2, size=(nx, ny, nz))
    voxel_grid = np.clip(voxel_grid, 0.0, DN_MAX)
    print(f"done. ({voxel_grid.nbytes / 1e6:.1f} MB)")

    # Reference model
    print("Creating reference model...", end=" ", flush=True)
    ref_model = ReferenceModel(n_inputs, n_outputs, hidden_size=32, seed=seed)
    print("done.")

    # Adam optimizer
    optimizer = AdamOptimizer(voxel_grid.shape, lr=lr)

    # Precompute detector masks (reused every step)
    print("Building detector masks...", end=" ", flush=True)
    detector_masks = get_detector_masks(nx, ny, n_outputs)
    print("done.")

    # --- Training loop ---
    print("\nStarting optimization...")
    print("-" * 60)

    loss_history = []
    ce_history = []
    accuracy_history = []
    t_start = time.time()

    for step in range(n_steps):
        step_loss = 0.0
        step_ce = 0.0
        step_correct = 0
        accumulated_grad = np.zeros_like(voxel_grid)

        # Mini-batch
        inputs, targets = ref_model.generate_batch(batch_size, rng=rng)

        for b in range(batch_size):
            # Encode input
            input_field = encode_input(inputs[b], nx, ny, n_inputs)

            # Forward pass
            output_field, fields_cache = forward(
                input_field, voxel_grid, propagator, k0, DZ
            )

            # Decode output — raw detector intensities
            output_intensities = decode_output(
                output_field, nx, ny, n_outputs, masks=detector_masks
            )

            # Loss (cross-entropy on intensities)
            target_class = targets[b]
            loss, ce = compute_loss(
                output_intensities, target_class, n_outputs,
                voxel_grid, sparsity_lambda
            )
            step_loss += loss
            step_ce += ce

            # Check classification accuracy
            if np.argmax(output_intensities) == target_class:
                step_correct += 1

            # Backward pass
            loss_grad = output_loss_gradient(
                output_field, nx, ny, n_outputs,
                output_intensities, target_class, masks=detector_masks
            )
            grad = backward(loss_grad, voxel_grid, fields_cache, propagator, k0, DZ)
            accumulated_grad += grad

        # Average gradients
        accumulated_grad /= batch_size
        avg_loss = step_loss / batch_size
        avg_ce = step_ce / batch_size
        accuracy = step_correct / batch_size

        loss_history.append(avg_loss)
        ce_history.append(avg_ce)
        accuracy_history.append(accuracy)

        # Adam update
        voxel_grid = optimizer.step(voxel_grid, accumulated_grad, sparsity_lambda)

        # Ternary annealing
        voxel_grid = ternary_anneal(voxel_grid, step, n_steps)

        # Logging
        if step % log_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t_start
            ms_per_step = (elapsed / (step + 1)) * 1000

            # Ternary statistics
            q_grid, t_idx = hard_quantize(voxel_grid)
            n_zero = np.sum(t_idx == 0)
            n_half = np.sum(t_idx == 1)
            n_full = np.sum(t_idx == 2)
            total = t_idx.size

            print(f"Step {step:5d}/{n_steps} | "
                  f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                  f"Acc: {accuracy:.1%} | "
                  f"Ternary: [{n_zero/total:.0%}/{n_half/total:.0%}/{n_full/total:.0%}] | "
                  f"{ms_per_step:.0f} ms/step")

    # --- Final quantization ---
    elapsed_total = time.time() - t_start
    print("-" * 60)
    print(f"Optimization complete in {elapsed_total:.1f}s")

    print("\nApplying hard ternary quantization...")
    quantized_grid, ternary_indices = hard_quantize(voxel_grid)

    # Evaluate quantized model
    print("Evaluating quantized model...")
    n_eval = 200
    eval_inputs, eval_targets = ref_model.generate_batch(n_eval, rng=np.random.RandomState(999))

    correct_continuous = 0
    correct_quantized = 0

    for b in range(n_eval):
        input_field = encode_input(eval_inputs[b], nx, ny, n_inputs)

        # Continuous
        out_field_c, _ = forward(input_field, voxel_grid, propagator, k0, DZ)
        out_c = decode_output(out_field_c, nx, ny, n_outputs, masks=detector_masks)
        if np.argmax(out_c) == eval_targets[b]:
            correct_continuous += 1

        # Quantized
        out_field_q, _ = forward(input_field, quantized_grid, propagator, k0, DZ)
        out_q = decode_output(out_field_q, nx, ny, n_outputs, masks=detector_masks)
        if np.argmax(out_q) == eval_targets[b]:
            correct_quantized += 1

    acc_cont = correct_continuous / n_eval
    acc_quant = correct_quantized / n_eval

    # Ternary stats
    n_zero = np.sum(ternary_indices == 0)
    n_half = np.sum(ternary_indices == 1)
    n_full = np.sum(ternary_indices == 2)
    total_voxels = ternary_indices.size

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Continuous accuracy:  {acc_cont:.1%}")
    print(f"Quantized accuracy:   {acc_quant:.1%}")
    print(f"Random baseline:      {1.0/n_outputs:.1%}")
    print(f"Quantization loss:    {acc_cont - acc_quant:+.1%}")
    print(f"\nTernary distribution:")
    print(f"  Zero  (dn=0):       {n_zero:,d} ({n_zero/total_voxels:.1%}) — no laser writing")
    print(f"  Half  (dn={DN_MAX/2}):  {n_half:,d} ({n_half/total_voxels:.1%})")
    print(f"  Full  (dn={DN_MAX}):  {n_full:,d} ({n_full/total_voxels:.1%})")
    print(f"  Sparsity:           {n_zero/total_voxels:.1%} voxels untouched")
    print(f"\nPerformance:")
    print(f"  Total time:         {elapsed_total:.1f}s")
    print(f"  Per step:           {elapsed_total/n_steps*1000:.1f} ms")
    print(f"  Grid memory:        {voxel_grid.nbytes/1e6:.1f} MB (continuous)")
    print(f"  Ternary memory:     {total_voxels * 2 / 8 / 1e6:.2f} MB (2-bit packed)")
    print(f"{'=' * 60}")

    # --- Save results ---
    results = {
        "config": {
            "nx": nx, "ny": ny, "nz": nz,
            "n_inputs": n_inputs, "n_outputs": n_outputs,
            "wavelength": WAVELENGTH, "n_base": N_BASE,
            "dn_max": DN_MAX, "voxel_pitch": VOXEL_PITCH, "dz": DZ,
            "n_steps": n_steps, "lr": lr,
            "sparsity_lambda": sparsity_lambda,
            "batch_size": batch_size
        },
        "metrics": {
            "final_loss": float(loss_history[-1]),
            "final_ce": float(ce_history[-1]),
            "accuracy_continuous": float(acc_cont),
            "accuracy_quantized": float(acc_quant),
            "random_baseline": float(1.0 / n_outputs),
            "total_time_s": float(elapsed_total),
            "ms_per_step": float(elapsed_total / n_steps * 1000)
        },
        "ternary_stats": {
            "n_zero": int(n_zero),
            "n_half": int(n_half),
            "n_full": int(n_full),
            "total_voxels": int(total_voxels),
            "sparsity": float(n_zero / total_voxels)
        },
        "loss_history": [float(x) for x in loss_history[::max(1, len(loss_history)//100)]],
        "accuracy_history": [float(x) for x in accuracy_history[::max(1, len(accuracy_history)//100)]]
    }

    return results, voxel_grid, quantized_grid, ternary_indices


# =============================================================================
# Save / export
# =============================================================================

def save_results(results, voxel_grid, ternary_indices, output_dir):
    """Save optimization results and engraving file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save continuous grid (for further optimization)
    np.save(output_dir / "voxel_grid_continuous.npy", voxel_grid)

    # Save ternary indices (the "engraving file")
    np.save(output_dir / "engraving_file.npy", ternary_indices.astype(np.uint8))

    # Save packed 2-bit version (compact engraving file)
    packed = pack_ternary(ternary_indices)
    np.save(output_dir / "engraving_file_packed.npy", packed)

    print(f"\nResults saved to {output_dir}/")
    print(f"  results.json              — metrics and config")
    print(f"  voxel_grid_continuous.npy — continuous grid ({voxel_grid.nbytes/1e6:.1f} MB)")
    print(f"  engraving_file.npy        — ternary indices ({ternary_indices.nbytes/1e6:.1f} MB)")
    print(f"  engraving_file_packed.npy — 2-bit packed ({packed.nbytes/1e6:.2f} MB)")


def pack_ternary(indices):
    """Pack ternary indices (0,1,2) into 2-bit representation (4 voxels per byte)."""
    flat = indices.flatten().astype(np.uint8)
    # Pad to multiple of 4
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])

    # Pack 4 values per byte
    packed = (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]
    return packed


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Photonic Neural Compute — BPM Digital Twin Simulator"
    )
    parser.add_argument("--nx", type=int, default=96, help="Grid width (default: 96)")
    parser.add_argument("--ny", type=int, default=96, help="Grid height (default: 96)")
    parser.add_argument("--nz", type=int, default=20, help="Grid depth (default: 20)")
    parser.add_argument("--inputs", type=int, default=4, help="Input channels (default: 4)")
    parser.add_argument("--outputs", type=int, default=4, help="Output classes (default: 4)")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps (default: 2000)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--sparsity", type=float, default=1e-4, help="L1 sparsity weight (default: 1e-4)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N steps (default: 100)")

    args = parser.parse_args()

    results, voxel_grid, quantized_grid, ternary_indices = optimize(
        nx=args.nx, ny=args.ny, nz=args.nz,
        n_inputs=args.inputs, n_outputs=args.outputs,
        n_steps=args.steps, lr=args.lr,
        sparsity_lambda=args.sparsity,
        batch_size=args.batch,
        log_every=args.log_every,
        seed=args.seed
    )

    # Save results
    output_dir = args.output or f"./results_{args.nx}x{args.ny}x{args.nz}"
    save_results(results, voxel_grid, ternary_indices, output_dir)

    return results


if __name__ == "__main__":
    main()
