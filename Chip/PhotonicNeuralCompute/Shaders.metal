// Photonic Neural Compute — Metal Compute Shaders
// SNRGY Studios — February 2026
//
// GPU kernels for BPM simulation with 4-bit (16-level) voxel quantization.
//
//   1. phase_screen_forward          — continuous delta_n phase screen (training)
//   2. phase_screen_forward_quantized — packed 4-bit LUT phase screen (inference)
//   3. elementwise_multiply          — FFT-domain propagation
//   4. elementwise_multiply_conj     — FFT-domain backward propagation
//   5. cache_field                   — snapshot field for adjoint pass
//   6. compute_gradient_and_reverse_phase — Wirtinger adjoint gradient
//   7. gumbel_softmax_forward        — logits -> continuous delta_n
//   8. gumbel_softmax_backward       — grad_deltaN -> grad_logits
//   9. adam_update                   — Adam optimizer on logits with L1 sparsity
//  10. zero_buffer / zero_complex_buffer — utility clears

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Function Constants (set from Swift at pipeline creation)
// ============================================================

constant uint  NX       [[function_constant(0)]];
constant uint  NY       [[function_constant(1)]];
constant uint  NZ       [[function_constant(2)]];
constant float K0       [[function_constant(3)]];
constant float DZ_CONST [[function_constant(4)]];
constant float DN_MAX   [[function_constant(5)]];
constant uint  N_LEVELS [[function_constant(6)]];

// ============================================================
// Complex number helpers — float2 (x=real, y=imag)
// ============================================================

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y,
                  a.x * b.y + a.y * b.x);
}

inline float2 cconj(float2 a) {
    return float2(a.x, -a.y);
}

inline float cabs2(float2 a) {
    return a.x * a.x + a.y * a.y;
}

// cexp(phase) = cos(phase) + i*sin(phase)
inline float2 cexp(float phase) {
    return float2(cos(phase), sin(phase));
}

// ============================================================
// 1. phase_screen_forward
//    Training kernel: field[x,y] *= exp(i * K0 * deltaN[x,y,z] * DZ_CONST)
//    Uses continuous delta_n values (Gumbel-Softmax output).
// ============================================================

kernel void phase_screen_forward(
    device float2       *field       [[buffer(0)]],  // Complex field [NX * NY]
    device const float  *delta_n     [[buffer(1)]],  // Continuous delta_n [NX * NY * NZ]
    constant uint       &z_slice     [[buffer(2)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;

    uint field_idx = gid.x * NY + gid.y;
    uint grid_idx  = gid.x * NY * NZ + gid.y * NZ + z_slice;

    float dn    = delta_n[grid_idx];
    float phase = K0 * dn * DZ_CONST;
    float2 pf   = cexp(phase);

    field[field_idx] = cmul(field[field_idx], pf);
}

// ============================================================
// 2. phase_screen_forward_quantized
//    Inference kernel: reads packed 4-bit voxels and uses a
//    precomputed 16-entry exp(i*phase) lookup table.
//    Two voxels per byte: low nibble = even index, high nibble = odd.
//    NO trigonometric math — pure table lookup.
// ============================================================

kernel void phase_screen_forward_quantized(
    device float2             *field      [[buffer(0)]],  // Complex field [NX * NY]
    device const uchar        *packed     [[buffer(1)]],  // Packed 4-bit voxels [ceil(NX*NY*NZ / 2)]
    constant float2           *phase_lut  [[buffer(2)]],  // Precomputed exp(i*phase) for each level [16]
    constant uint             &z_slice    [[buffer(3)]],
    uint2 gid                             [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;

    uint field_idx  = gid.x * NY + gid.y;
    uint voxel_idx  = gid.x * NY * NZ + gid.y * NZ + z_slice;

    // Unpack 4-bit level from byte
    uint byte_idx   = voxel_idx >> 1;       // voxel_idx / 2
    uchar raw       = packed[byte_idx];
    uint level;
    if ((voxel_idx & 1u) == 0u) {
        level = uint(raw & 0x0F);           // low nibble (even voxel)
    } else {
        level = uint((raw >> 4) & 0x0F);    // high nibble (odd voxel)
    }

    // Pure LUT — no trig
    float2 pf = phase_lut[level];

    field[field_idx] = cmul(field[field_idx], pf);
}

// ============================================================
// 3. elementwise_multiply
//    FFT-domain forward propagation: field_fft[k] *= H[k]
// ============================================================

kernel void elementwise_multiply(
    device float2       *field_fft  [[buffer(0)]],  // FFT of field [NX * NY]
    device const float2 *H          [[buffer(1)]],  // Propagation transfer function [NX * NY]
    uint2 gid                       [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;
    uint idx = gid.x * NY + gid.y;
    field_fft[idx] = cmul(field_fft[idx], H[idx]);
}

// ============================================================
// 4. elementwise_multiply_conj
//    FFT-domain backward propagation: field_fft[k] *= conj(H[k])
// ============================================================

kernel void elementwise_multiply_conj(
    device float2       *field_fft  [[buffer(0)]],
    device const float2 *H          [[buffer(1)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;
    uint idx = gid.x * NY + gid.y;
    field_fft[idx] = cmul(field_fft[idx], cconj(H[idx]));
}

// ============================================================
// 5. cache_field
//    Copy current field slice into cache for adjoint backward pass.
//    cache layout: [NZ][NX][NY] — cache[z * NX * NY + field_idx]
// ============================================================

kernel void cache_field(
    device const float2 *field      [[buffer(0)]],
    device float2       *cache      [[buffer(1)]],  // [NZ * NX * NY]
    constant uint       &z_slice    [[buffer(2)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;
    uint field_idx = gid.x * NY + gid.y;
    uint cache_idx = z_slice * NX * NY + field_idx;
    cache[cache_idx] = field[field_idx];
}

// ============================================================
// 6. compute_gradient_and_reverse_phase
//    Adjoint method: accumulates voxel gradient and reverses
//    the phase screen on the adjoint field.
//
//    Wirtinger gradient (CRITICAL — must use conj(adjoint)):
//      grad[x,y,z] += 2 * Re( conj(adjoint[x,y]) * i*K0*DZ * field_after_phase[x,y] )
//
//    Reverse phase on adjoint:
//      adjoint[x,y] *= exp(-i * K0 * deltaN * DZ_CONST)
// ============================================================

kernel void compute_gradient_and_reverse_phase(
    device float2       *adjoint    [[buffer(0)]],  // Adjoint field [NX * NY] — modified in-place
    device const float2 *cache      [[buffer(1)]],  // Cached field snapshots [NZ * NX * NY]
    device const float  *delta_n    [[buffer(2)]],  // Continuous delta_n [NX * NY * NZ]
    device float        *grad       [[buffer(3)]],  // Output gradient [NX * NY * NZ]
    constant uint       &z_slice    [[buffer(4)]],
    uint2 gid                       [[thread_position_in_grid]])
{
    if (gid.x >= NX || gid.y >= NY) return;

    uint field_idx = gid.x * NY + gid.y;
    uint grid_idx  = gid.x * NY * NZ + gid.y * NZ + z_slice;
    uint cache_idx = z_slice * NX * NY + field_idx;

    float dn    = delta_n[grid_idx];
    float phase = K0 * dn * DZ_CONST;

    // Reconstruct field_after_phase = cached_field_before_phase * exp(i * phase)
    float2 f_before = cache[cache_idx];
    float2 pf       = cexp(phase);
    float2 f_after  = cmul(f_before, pf);

    // i * K0 * DZ_CONST * f_after
    // Multiply by i: (a + bi) * i = (-b + ai)
    float2 ik0dz_f = float2(-K0 * DZ_CONST * f_after.y,
                             K0 * DZ_CONST * f_after.x);

    // Wirtinger gradient: 2 * Re( conj(adjoint) * i*K0*DZ*f_after )
    float2 adj      = adjoint[field_idx];
    float2 conj_adj = cconj(adj);
    float2 product  = cmul(conj_adj, ik0dz_f);
    grad[grid_idx] += 2.0f * product.x;  // += for batch accumulation

    // Reverse phase screen on adjoint: adjoint *= exp(-i * phase)
    float2 neg_pf = cexp(-phase);
    adjoint[field_idx] = cmul(adj, neg_pf);
}

// ============================================================
// 7. gumbel_softmax_forward
//    Convert per-voxel logits (16 per voxel) to continuous delta_n.
//
//    For each voxel v:
//      softmax_j = exp(logit_j / temperature) / sum_k exp(logit_k / temperature)
//      delta_n[v] = sum_j softmax_j * dn_levels[j]
//
//    Buffers:
//      logits:    [totalVoxels * N_LEVELS]  (16 logits per voxel)
//      delta_n:   [totalVoxels]
//      dn_levels: [N_LEVELS]  (the 16 discrete delta_n values)
// ============================================================

kernel void gumbel_softmax_forward(
    device const float  *logits     [[buffer(0)]],  // [totalVoxels * N_LEVELS]
    device float        *delta_n    [[buffer(1)]],  // [totalVoxels]
    constant float      *dn_levels  [[buffer(2)]],  // [N_LEVELS] discrete delta_n values
    constant float      &temperature [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total_voxels = NX * NY * NZ;
    if (gid >= total_voxels) return;

    uint base = gid * N_LEVELS;

    // Numerically stable softmax: find max logit first
    float max_logit = logits[base];
    for (uint j = 1; j < N_LEVELS; j++) {
        max_logit = max(max_logit, logits[base + j]);
    }

    // Compute exp((logit_j - max) / temperature) and sum
    float sum_exp = 0.0f;
    float softmax_vals[16];  // N_LEVELS <= 16 always
    for (uint j = 0; j < N_LEVELS; j++) {
        float e = exp((logits[base + j] - max_logit) / temperature);
        softmax_vals[j] = e;
        sum_exp += e;
    }

    // Weighted sum: delta_n = sum_j softmax_j * dn_levels[j]
    float dn = 0.0f;
    float inv_sum = 1.0f / sum_exp;
    for (uint j = 0; j < N_LEVELS; j++) {
        dn += (softmax_vals[j] * inv_sum) * dn_levels[j];
    }

    delta_n[gid] = dn;
}

// ============================================================
// 8. gumbel_softmax_backward
//    Backprop gradient from grad_deltaN through Gumbel-Softmax.
//
//    For each voxel v, for each level j:
//      d(delta_n)/d(logit_j) = softmax_j * (dn_levels[j] - delta_n) / temperature
//      grad_logit_j = grad_deltaN * d(delta_n)/d(logit_j)
//
//    This follows directly from the Jacobian of softmax-weighted sum.
// ============================================================

kernel void gumbel_softmax_backward(
    device const float  *logits       [[buffer(0)]],  // [totalVoxels * N_LEVELS]
    device const float  *delta_n      [[buffer(1)]],  // [totalVoxels] (forward output)
    device const float  *grad_delta_n [[buffer(2)]],  // [totalVoxels] gradient w.r.t. delta_n
    device float        *grad_logits  [[buffer(3)]],  // [totalVoxels * N_LEVELS] output
    constant float      *dn_levels    [[buffer(4)]],  // [N_LEVELS]
    constant float      &temperature  [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]])
{
    uint total_voxels = NX * NY * NZ;
    if (gid >= total_voxels) return;

    uint base = gid * N_LEVELS;

    // Recompute softmax (same as forward — needed for gradient)
    float max_logit = logits[base];
    for (uint j = 1; j < N_LEVELS; j++) {
        max_logit = max(max_logit, logits[base + j]);
    }

    float sum_exp = 0.0f;
    float softmax_vals[16];
    for (uint j = 0; j < N_LEVELS; j++) {
        float e = exp((logits[base + j] - max_logit) / temperature);
        softmax_vals[j] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;

    float dn       = delta_n[gid];
    float grad_dn  = grad_delta_n[gid];
    float inv_temp = 1.0f / temperature;

    // grad_logit_j = grad_deltaN * softmax_j * (dn_levels[j] - delta_n) / temperature
    for (uint j = 0; j < N_LEVELS; j++) {
        float sj = softmax_vals[j] * inv_sum;
        grad_logits[base + j] = grad_dn * sj * (dn_levels[j] - dn) * inv_temp;
    }
}

// ============================================================
// 9. adam_update
//    Adam optimizer step on the logits buffer.
//    Operates on totalVoxels * N_LEVELS elements.
//
//    Includes an L1 sparsity term applied to the delta_n values
//    (not the logits directly). The sparsity gradient is distributed
//    to logits via the chain rule through Gumbel-Softmax, but here
//    we approximate by adding lambda * sign(delta_n) to each logit's
//    gradient proportional to its softmax weight.
//
//    For simplicity, sparsity is added to the raw gradient buffer
//    before the Adam moment update. The caller is responsible for
//    injecting sparsity gradients into grad_logits before calling
//    this kernel, or setting sparsity_lambda > 0 to use the built-in
//    L1 penalty on the current logit values.
// ============================================================

kernel void adam_update(
    device float        *logits    [[buffer(0)]],   // Parameters [totalVoxels * N_LEVELS]
    device float        *grad      [[buffer(1)]],   // Gradients [totalVoxels * N_LEVELS]
    device float        *m         [[buffer(2)]],   // First moment [totalVoxels * N_LEVELS]
    device float        *v         [[buffer(3)]],   // Second moment [totalVoxels * N_LEVELS]
    constant float      &lr        [[buffer(4)]],
    constant float      &beta1     [[buffer(5)]],
    constant float      &beta2     [[buffer(6)]],
    constant float      &eps       [[buffer(7)]],
    constant float      &m_correction [[buffer(8)]],  // 1 / (1 - beta1^t)
    constant float      &v_correction [[buffer(9)]],  // 1 / (1 - beta2^t)
    constant float      &sparsity_lambda [[buffer(10)]],
    device const float  *delta_n   [[buffer(11)]],  // Current delta_n values [totalVoxels]
    uint gid                       [[thread_position_in_grid]])
{
    uint total_elements = NX * NY * NZ * N_LEVELS;
    if (gid >= total_elements) return;

    float g = grad[gid];

    // L1 sparsity gradient on delta_n, distributed to this logit element.
    // Each logit element maps to a voxel: voxel_id = gid / N_LEVELS
    if (sparsity_lambda > 0.0f) {
        uint voxel_id = gid / N_LEVELS;
        float dn_val = delta_n[voxel_id];
        // sign(delta_n) — subgradient of L1 norm
        float sign_dn = 0.0f;
        if (dn_val > 0.0f) sign_dn = 1.0f;
        else if (dn_val < 0.0f) sign_dn = -1.0f;
        g += sparsity_lambda * sign_dn;
    }

    // Adam moment updates
    float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
    float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = m_new;
    v[gid] = v_new;

    // Bias-corrected estimates
    float m_hat = m_new * m_correction;
    float v_hat = v_new * v_correction;

    // Parameter update
    float update = lr * m_hat / (sqrt(v_hat) + eps);
    logits[gid] -= update;

    // Zero gradient for next iteration
    grad[gid] = 0.0f;
}

// ============================================================
// 10–12. GPU FFT — Stockham autosort (radix-2, out-of-place)
//
// 2D FFT = 1D row FFT + 1D column FFT
// Each 1D FFT uses log2(N) Stockham butterfly passes.
// Two buffers ping-pong (field ↔ temp) so no bit-reversal needed.
//
// For NX=NY=64: 6 passes per dimension, 12 total per 2D FFT.
// All passes are encoded into one command encoder = zero CPU/GPU sync.
// ============================================================

// 10a. Stockham FFT pass along rows (y-dimension)
//   Processes all NX rows simultaneously.
//   Total threads: NX * NY / 2
kernel void fft_stockham_rows(
    device const float2 *src     [[buffer(0)]],
    device float2       *dst     [[buffer(1)]],
    constant uint       &pass    [[buffer(2)]],   // Pass index s (0..log2(NY)-1)
    constant float      &sign    [[buffer(3)]],   // -1.0 = forward, +1.0 = inverse
    uint gid [[thread_position_in_grid]])
{
    uint half_ny = NY / 2;
    uint row = gid / half_ny;
    uint k = gid % half_ny;
    if (row >= NX) return;

    uint group_size = 1u << pass;
    uint group = k >> pass;
    uint pair_idx = k & (group_size - 1u);

    // Source indices (natural order from previous pass)
    uint row_offset = row * NY;
    uint src_even = row_offset + k;
    uint src_odd  = row_offset + k + half_ny;

    // Destination indices (reordered by Stockham)
    uint dst0 = row_offset + (group << (pass + 1u)) + pair_idx;
    uint dst1 = dst0 + group_size;

    // Twiddle: exp(sign * i * 2π * pair_idx / (2 * group_size))
    float angle = sign * 2.0f * M_PI_F * float(pair_idx) / float(group_size << 1u);
    float2 tw = float2(cos(angle), sin(angle));

    float2 a = src[src_even];
    float2 b = cmul(src[src_odd], tw);

    dst[dst0] = a + b;
    dst[dst1] = a - b;
}

// 10b. Stockham FFT pass along columns (x-dimension)
//   Processes all NY columns simultaneously.
//   Total threads: NX / 2 * NY
kernel void fft_stockham_cols(
    device const float2 *src     [[buffer(0)]],
    device float2       *dst     [[buffer(1)]],
    constant uint       &pass    [[buffer(2)]],
    constant float      &sign    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_nx = NX / 2;
    uint col = gid / half_nx;
    uint k = gid % half_nx;
    if (col >= NY) return;

    uint group_size = 1u << pass;
    uint group = k >> pass;
    uint pair_idx = k & (group_size - 1u);

    // Source (column-stride access)
    uint src_even = k * NY + col;
    uint src_odd  = (k + half_nx) * NY + col;

    // Destination
    uint row0 = (group << (pass + 1u)) + pair_idx;
    uint row1 = row0 + group_size;
    uint dst0 = row0 * NY + col;
    uint dst1 = row1 * NY + col;

    float angle = sign * 2.0f * M_PI_F * float(pair_idx) / float(group_size << 1u);
    float2 tw = float2(cos(angle), sin(angle));

    float2 a = src[src_even];
    float2 b = cmul(src[src_odd], tw);

    dst[dst0] = a + b;
    dst[dst1] = a - b;
}

// 10c. Normalize after inverse FFT: field /= (NX * NY)
kernel void fft_normalize(
    device float2 *field [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= NX * NY) return;
    float scale = 1.0f / float(NX * NY);
    field[gid] *= scale;
}

// ============================================================
// 11. Utility kernels: zero_buffer, zero_complex_buffer
// ============================================================

kernel void zero_buffer(
    device float *buf  [[buffer(0)]],
    uint gid           [[thread_position_in_grid]])
{
    buf[gid] = 0.0f;
}

kernel void zero_complex_buffer(
    device float2 *buf [[buffer(0)]],
    uint gid           [[thread_position_in_grid]])
{
    buf[gid] = float2(0.0f, 0.0f);
}
