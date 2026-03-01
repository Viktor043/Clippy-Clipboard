// BPM.metal — Beam Propagation Method kernels (concatenated after Common.metal)

// =============================================================================
// Phase screen: field *= exp(i · k0 · Δn · dz) * exp(-k0 · κ_eff · dz)
// With saturable absorption: κ_eff = κ / (1 + |E|² / I_sat)
// 3D dispatch: (NX, NY, BATCH)
// =============================================================================

struct BPMParams {
    float k0;
    float dz;
    float I_sat;   // saturation intensity: 0 = linear, >0 = saturable absorption
};

kernel void phase_screen_forward_batched(
    device       float2 *field      [[buffer(0)]],  // [B * NX * NY]
    device const float  *deltaN     [[buffer(1)]],  // [NX * NY * NZ] (shared, not batched)
    constant     uint   &z_slice    [[buffer(2)]],
    constant     BPMParams &params  [[buffer(3)]],
    device const float  *deltaKappa [[buffer(4)]],  // [NX * NY * NZ] absorption
    uint3 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    uint b = gid.z;
    if (x >= NX || y >= NY || b >= BATCH) return;

    uint field_idx = b * NX * NY + x * NY + y;
    uint voxel_idx = z_slice * NX * NY + x * NY + y;

    float phase = params.k0 * deltaN[voxel_idx] * params.dz;
    float kappa = deltaKappa[voxel_idx];

    // Saturable absorption: effective_kappa = kappa / (1 + I / I_sat)
    float2 f = field[field_idx];
    float effective_kappa = kappa;
    if (params.I_sat > 0.0f) {
        float intensity = cabs2(f);
        effective_kappa = kappa / (1.0f + intensity / params.I_sat);
    }
    float atten = exp(-params.k0 * effective_kappa * params.dz);

    float2 phasor = cexp(phase);
    phasor *= atten;
    field[field_idx] = cmul(f, phasor);
}

// =============================================================================
// Elementwise multiply: field *= H (propagator in Fourier space)
// H is NOT batched — same propagator for all samples.
// Threads: BATCH * NX * NY (1D dispatch)
// =============================================================================

kernel void elementwise_multiply_batched(
    device       float2 *field      [[buffer(0)]],  // [B * NX * NY]
    device const float2 *propagator [[buffer(1)]],   // [NX * NY]
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    uint pixel = gid % (NX * NY);
    float2 h = propagator[pixel];
    float2 f = field[gid];
    field[gid] = cmul(f, h);
}

/// Multiply by conjugate of propagator (for backward pass)
kernel void elementwise_multiply_conj_batched(
    device       float2 *field      [[buffer(0)]],
    device const float2 *propagator [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    uint pixel = gid % (NX * NY);
    float2 h = cconj(propagator[pixel]);
    float2 f = field[gid];
    field[gid] = cmul(f, h);
}

// =============================================================================
// Cache field: save field snapshot for adjoint reconstruction
// Threads: BATCH * NX * NY (1D dispatch)
// =============================================================================

kernel void cache_field_batched(
    device const float2 *field [[buffer(0)]],    // [B * NX * NY]
    device       float2 *cache [[buffer(1)]],    // [B * (NZ+1) * NX * NY]
    constant     uint   &z_idx [[buffer(2)]],    // which z slot to cache into
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    uint b = gid / (NX * NY);
    uint pixel = gid % (NX * NY);
    uint cache_idx = b * (NZ + 1) * NX * NY + z_idx * NX * NY + pixel;
    cache[cache_idx] = field[gid];
}

// =============================================================================
// Gradient computation + reverse phase screen (backward pass)
//
// Computes Wirtinger gradient dL/d(deltaN) and dL/d(deltaKappa) for one z-slice
// and reverses the phase screen on the adjoint field.
// Supports saturable absorption (I_sat > 0).
//
// Each batch sample writes to its own gradient region to avoid atomics.
// Threads: BATCH * NX * NY (1D dispatch)
// =============================================================================

kernel void gradient_reverse_phase_batched(
    device       float2 *adjoint            [[buffer(0)]],  // [B * NX * NY] (modified in-place)
    device const float2 *field_cache        [[buffer(1)]],  // [B * (NZ+1) * NX * NY]
    device const float  *deltaN             [[buffer(2)]],  // [NX * NY * NZ]
    device       float  *per_batch_grad     [[buffer(3)]],  // [B * totalVoxels] dn gradient
    constant     uint   &z_slice            [[buffer(4)]],
    constant     BPMParams &params          [[buffer(5)]],
    device const float  *deltaKappa         [[buffer(6)]],  // [NX * NY * NZ] absorption
    device       float  *per_batch_grad_kappa [[buffer(7)]], // [B * totalVoxels] kappa gradient
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    uint b = gid / (NX * NY);
    uint pixel = gid % (NX * NY);

    uint voxel_idx = z_slice * NX * NY + pixel;
    float dn = deltaN[voxel_idx];
    float kappa = deltaKappa[voxel_idx];
    float phase = params.k0 * dn * params.dz;

    // Read cached field at this z (field entering this slice, before phase screen)
    uint cache_idx = b * (NZ + 1) * NX * NY + z_slice * NX * NY + pixel;
    float2 field_at_z = field_cache[cache_idx];

    // Compute effective absorption (saturable if I_sat > 0)
    float intensity = cabs2(field_at_z);
    float effective_kappa = kappa;
    float denom = 1.0f;
    if (params.I_sat > 0.0f) {
        denom = 1.0f + intensity / params.I_sat;
        effective_kappa = kappa / denom;
    }
    float atten = exp(-params.k0 * effective_kappa * params.dz);

    // Field after full phase screen (phase + saturable absorption)
    float2 phasor = cexp(phase);
    phasor *= atten;
    float2 field_after = cmul(field_at_z, phasor);

    // Adjoint field at this point
    float2 adj = adjoint[gid];

    // Gradient computation: conj(adj) * field_after
    float2 conj_adj = cconj(adj);
    float2 product = cmul(conj_adj, field_after);

    // deltaN gradient: dL/d(dn) = 2*k0*dz * (-product.y) — unchanged by saturation
    float gradient_dn = 2.0f * params.k0 * params.dz * (-product.y);

    // deltaKappa gradient with saturation correction:
    // d(effective_kappa)/d(kappa) = 1/denom
    float gradient_kappa = -2.0f * params.k0 * params.dz * product.x / denom;

    // Write to per-batch gradient buffers
    uint grad_idx = b * NX * NY * NZ + voxel_idx;
    per_batch_grad[grad_idx] = gradient_dn;
    per_batch_grad_kappa[grad_idx] = gradient_kappa;

    // Reverse phase screen on adjoint field
    // For saturable absorption: df/dE = phasor * (1 + I * k0*κ*dz / (I_sat * denom²))
    // Since correction is real, adj_in = conj(phasor) * correction * adj_out
    float2 conj_phasor = cconj(phasor);
    if (params.I_sat > 0.0f) {
        float correction = 1.0f + intensity * params.k0 * kappa * params.dz
                          / (params.I_sat * denom * denom);
        conj_phasor *= correction;
    }
    adjoint[gid] = cmul(adj, conj_phasor);
}

// =============================================================================
// Recurrent pulse kernels: threshold + feedback mixing
// =============================================================================

// Soft threshold: zero out intensity below cutoff, differentiable via sigmoid
// f_out = f_in * sigmoid(steepness * (|f_in|² - threshold²))
// Threads: BATCH * NX * NY (1D dispatch)

kernel void soft_threshold_batched(
    device       float2 *field       [[buffer(0)]],  // [B * NX * NY] modified in-place
    device       float2 *cache_out   [[buffer(1)]],  // [B * NX * NY] save pre-threshold for backward
    constant     float  &threshold   [[buffer(2)]],  // learnable threshold
    constant     float  &steepness   [[buffer(3)]],  // sigmoid steepness (e.g. 20)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    float2 f = field[gid];
    cache_out[gid] = f;  // save for backward

    float I = cabs2(f);
    float gate = 1.0f / (1.0f + exp(-steepness * (I - threshold * threshold)));
    field[gid] = f * gate;
}

// Backward through soft threshold
// Given adj_out (adjoint of thresholded field), compute adj_in
// df/dE = gate + E * gate*(1-gate)*steepness*2*E*  (Wirtinger)
//       = gate * (1 + 2*steepness*|E|²*(1-gate))
// Threads: BATCH * NX * NY (1D dispatch)

kernel void soft_threshold_backward_batched(
    device       float2 *adjoint     [[buffer(0)]],  // [B * NX * NY] modified in-place
    device const float2 *cached_field [[buffer(1)]], // [B * NX * NY] pre-threshold field
    constant     float  &threshold   [[buffer(2)]],
    constant     float  &steepness   [[buffer(3)]],
    device       float  *grad_threshold [[buffer(4)]], // [1] atomic accumulator for threshold grad
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    float2 f = cached_field[gid];
    float I = cabs2(f);
    float gate = 1.0f / (1.0f + exp(-steepness * (I - threshold * threshold)));
    float gate_deriv = gate * (1.0f - gate);

    // Adjoint propagation through threshold
    float correction = gate + 2.0f * steepness * I * gate_deriv;
    float2 adj = adjoint[gid];
    adjoint[gid] = adj * correction;

    // Threshold gradient disabled: 131K atomic CAS on one float causes GPU hang.
    // TODO: replace with threadgroup reduction or per-batch buffer + reduce kernel.
    // For now, threshold stays at its initial value.
    // float2 conj_adj = cconj(adj);
    // float2 prod = cmul(conj_adj, f);
    // float dL_dthresh = 2.0f * prod.x * (-gate_deriv * steepness * 2.0f * threshold);
    // atomic_add_f32(grad_threshold, dL_dthresh);
}

// Feedback mixing: field = ratio * thresholded_output + (1 - ratio) * fresh_input
// Threads: BATCH * NX * NY (1D dispatch)

kernel void feedback_mix_batched(
    device       float2 *field       [[buffer(0)]],  // [B * NX * NY] thresholded output, overwritten
    device const float2 *input_field [[buffer(1)]],  // [B * NX * NY] fresh input encoding
    constant     float  &ratio       [[buffer(2)]],  // learnable feedback ratio
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    float2 feedback = field[gid];
    float2 fresh = input_field[gid];
    field[gid] = ratio * feedback + (1.0f - ratio) * fresh;
}

// Backward through feedback mixing
// adj_feedback = ratio * adj_mixed
// adj_input = (1 - ratio) * adj_mixed
// grad_ratio += sum(Re(conj(adj_mixed) * (feedback - input)))
// Threads: BATCH * NX * NY (1D dispatch)

kernel void feedback_mix_backward_batched(
    device       float2 *adj_mixed       [[buffer(0)]],  // [B * NX * NY] input adjoint, becomes adj_feedback
    device       float2 *adj_input       [[buffer(1)]],  // [B * NX * NY] output: adjoint for fresh input
    device const float2 *feedback_field  [[buffer(2)]],  // [B * NX * NY] thresholded output (cached)
    device const float2 *input_field     [[buffer(3)]],  // [B * NX * NY] fresh input (cached)
    constant     float  &ratio           [[buffer(4)]],
    device       float  *grad_ratio      [[buffer(5)]],  // [1] atomic accumulator
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;

    float2 adj = adj_mixed[gid];
    float2 fb = feedback_field[gid];
    float2 inp = input_field[gid];

    // Ratio gradient disabled: 131K atomic CAS on one float causes GPU hang.
    // TODO: replace with threadgroup reduction or per-batch buffer + reduce kernel.
    // float2 diff = fb - inp;
    // float2 conj_adj = cconj(adj);
    // float dL_dratio = 2.0f * cmul(conj_adj, diff).x;
    // atomic_add_f32(grad_ratio, dL_dratio);

    // Propagate adjoint
    adj_mixed[gid] = adj * ratio;          // adj_feedback
    adj_input[gid] = adj * (1.0f - ratio); // adj_fresh_input (accumulated across pulses)
}
