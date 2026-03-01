// Training.metal — Gumbel-softmax + Adam optimizer (concatenated after Common.metal)

// =============================================================================
// Gumbel-softmax forward: logits → deltaN
//
// For each voxel, compute softmax over NLEVELS logits (with Gumbel noise),
// then weighted sum of dn_levels to produce continuous deltaN.
// Threads: TOTAL_VOXELS (1D dispatch, NOT batched — grid is shared)
// =============================================================================

kernel void gumbel_softmax_forward(
    device const float *logits   [[buffer(0)]],   // [totalVoxels * nLevels]
    device       float *deltaN   [[buffer(1)]],   // [totalVoxels]
    device const float *dn_levels [[buffer(2)]],  // [nLevels]
    constant SimParams &params   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= TOTAL_VOXELS) return;

    float inv_temp = 1.0f / max(params.temperature, 1e-6f);
    uint base = gid * NLEVELS;

    // First pass: compute perturbed logits and find max for numerical stability
    float perturbed[16];  // NLEVELS max = 16
    float max_val = -1e30f;
    for (uint j = 0; j < NLEVELS; j++) {
        float g = 0.0f;
        if (false) {  // Gumbel noise disabled: pure softmax for continuous optimization
            float u = philox_uniform(gid * NLEVELS + j, params.step, params.seed);
            u = clamp(u, 1e-6f, 1.0f - 1e-6f);
            g = -log(-log(u));
        }
        perturbed[j] = logits[base + j] + g;
        max_val = max(max_val, perturbed[j]);
    }

    // Second pass: compute exp (shifted by max of perturbed values for stability)
    float sum_exp = 0.0f;
    float exps[16];
    for (uint j = 0; j < NLEVELS; j++) {
        float val = (perturbed[j] - max_val) * inv_temp;
        exps[j] = exp(val);
        sum_exp += exps[j];
    }

    // Normalize and compute weighted sum
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    float weighted_sum = 0.0f;
    for (uint j = 0; j < NLEVELS; j++) {
        float prob = exps[j] * inv_sum;
        weighted_sum += prob * dn_levels[j];
    }

    deltaN[gid] = weighted_sum;
}

// =============================================================================
// Gumbel-softmax backward: gradDeltaN → gradLogits
//
// Chain rule: dL/d_logit_j = (1/T) * p_j * (level_j - deltaN) * dL/d_deltaN
// Threads: TOTAL_VOXELS (1D dispatch)
// =============================================================================

kernel void gumbel_softmax_backward(
    device const float *logits     [[buffer(0)]],   // [totalVoxels * nLevels]
    device const float *deltaN     [[buffer(1)]],   // [totalVoxels]
    device const float *gradDeltaN [[buffer(2)]],   // [totalVoxels]
    device       float *gradLogits [[buffer(3)]],   // [totalVoxels * nLevels]
    device const float *dn_levels  [[buffer(4)]],   // [nLevels]
    constant SimParams &params     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= TOTAL_VOXELS) return;

    float inv_temp = 1.0f / max(params.temperature, 1e-6f);
    uint base = gid * NLEVELS;
    float dn = deltaN[gid];
    float grad_dn = gradDeltaN[gid];

    // Recompute softmax with SAME Gumbel noise as forward (consistent gradient)
    float max_val = -1e30f;
    float perturbed[16];
    for (uint j = 0; j < NLEVELS; j++) {
        float g = 0.0f;
        if (false) {  // Gumbel noise disabled: pure softmax for continuous optimization
            float u = philox_uniform(gid * NLEVELS + j, params.step, params.seed);
            u = clamp(u, 1e-6f, 1.0f - 1e-6f);
            g = -log(-log(u));
        }
        perturbed[j] = logits[base + j] + g;
        max_val = max(max_val, perturbed[j]);
    }

    float sum_exp = 0.0f;
    float probs[16];
    for (uint j = 0; j < NLEVELS; j++) {
        probs[j] = exp((perturbed[j] - max_val) * inv_temp);
        sum_exp += probs[j];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);

    for (uint j = 0; j < NLEVELS; j++) {
        float p = probs[j] * inv_sum;
        // Omit inv_temp factor: exact gradient is (1/T)*p*(level-dn)*grad_dn,
        // but 1/T causes 20x gradient explosion at low T, overwhelming Adam.
        // Without 1/T this is a stable rescaled STE that still points correctly.
        gradLogits[base + j] = p * (dn_levels[j] - dn) * grad_dn;
    }
}

// =============================================================================
// Adam update: update logits from gradLogits
// Threads: TOTAL_VOXELS * NLEVELS (1D dispatch)
// =============================================================================

kernel void adam_update(
    device       float *logits     [[buffer(0)]],
    device const float *gradLogits [[buffer(1)]],
    device       float *momentM    [[buffer(2)]],
    device       float *momentV    [[buffer(3)]],
    constant SimParams &params     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= TOTAL_VOXELS * NLEVELS) return;

    float g = gradLogits[gid];

    // Per-element gradient clipping (prevents outlier gradients from destabilizing training)
    if (params.grad_clip_max > 0.0f) {
        g = clamp(g, -params.grad_clip_max, params.grad_clip_max);
    }

    float m = momentM[gid];
    float v = momentV[gid];

    // Update moments
    m = params.beta1 * m + (1.0f - params.beta1) * g;
    v = params.beta2 * v + (1.0f - params.beta2) * g * g;

    // Bias correction
    float step_f = float(params.step + 1);
    float m_hat = m / (1.0f - pow(params.beta1, step_f));
    float v_hat = v / (1.0f - pow(params.beta2, step_f));

    // Update
    logits[gid] -= params.lr * m_hat / (sqrt(v_hat) + params.epsilon);

    // Store updated moments
    momentM[gid] = m;
    momentV[gid] = v;
}

// =============================================================================
// Reduce batch gradients: sum per-sample grads → accumulated gradient
// Threads: TOTAL_VOXELS (1D dispatch)
// =============================================================================

kernel void reduce_batch_gradients(
    device const float *per_batch_grad [[buffer(0)]],  // [B * totalVoxels]
    device       float *gradDeltaN     [[buffer(1)]],  // [totalVoxels]
    constant     uint  &batch_size     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= TOTAL_VOXELS) return;

    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += per_batch_grad[b * TOTAL_VOXELS + gid];
    }
    gradDeltaN[gid] += sum;  // accumulate: supports multi-pulse recurrent (gradients zeroed at step start)
}

// =============================================================================
// Scale buffer: multiply all elements by a scalar
// Threads: buffer length (1D dispatch)
// =============================================================================

kernel void scale_buffer(
    device float    *data   [[buffer(0)]],
    constant float  &scale  [[buffer(1)]],
    constant uint   &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    data[gid] *= scale;
}

// =============================================================================
// Fill buffer: set all elements to a constant value
// Unlike scale_buffer, this handles NaN correctly (NaN * 0 = NaN, but fill = 0 works)
// Threads: buffer length (1D dispatch)
// =============================================================================

kernel void fill_buffer(
    device float    *data   [[buffer(0)]],
    constant float  &value  [[buffer(1)]],
    constant uint   &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    data[gid] = value;
}

// =============================================================================
// EMA weight update: ema[i] = decay * ema[i] + (1 - decay) * current[i]
// Maintains exponentially-weighted average of weights for Polyak averaging.
// Threads: buffer length (1D dispatch)
// =============================================================================

kernel void ema_update(
    device const float *current [[buffer(0)]],   // current weights (logits)
    device       float *ema     [[buffer(1)]],   // EMA shadow weights
    constant float     &decay   [[buffer(2)]],   // EMA decay (e.g. 0.999)
    constant uint      &count   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    ema[gid] = decay * ema[gid] + (1.0f - decay) * current[gid];
}

// =============================================================================
// Direct Adam update: optimize deltaN directly (no Gumbel-softmax intermediary)
// Operates on TOTAL_VOXELS (not TOTAL_VOXELS * NLEVELS)
// Includes clamping to [-dn_max, +dn_max] after update
// Threads: TOTAL_VOXELS (1D dispatch)
// =============================================================================

kernel void adam_update_direct(
    device       float *deltaN     [[buffer(0)]],
    device const float *gradDeltaN [[buffer(1)]],
    device       float *momentM    [[buffer(2)]],
    device       float *momentV    [[buffer(3)]],
    constant SimParams &params     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= TOTAL_VOXELS) return;

    float g = gradDeltaN[gid];

    // Per-element gradient clipping
    if (params.grad_clip_max > 0.0f) {
        g = clamp(g, -params.grad_clip_max, params.grad_clip_max);
    }

    float m = momentM[gid];
    float v = momentV[gid];

    // Update moments
    m = params.beta1 * m + (1.0f - params.beta1) * g;
    v = params.beta2 * v + (1.0f - params.beta2) * g * g;

    // Bias correction
    float step_f = float(params.step + 1);
    float m_hat = m / (1.0f - pow(params.beta1, step_f));
    float v_hat = v / (1.0f - pow(params.beta2, step_f));

    // Update
    float dn = deltaN[gid] - params.lr * m_hat / (sqrt(v_hat) + params.epsilon);

    // Phase mode (dn_max ≈ π): wrap to [-π, π] instead of clamping.
    // Clamping creates walls where Adam momentum builds up and overshoots.
    // Wrapping is smooth since exp(i*φ) is 2π-periodic.
    if (params.dn_max > 1.0f) {
        dn = fmod(dn + params.dn_max, 2.0f * params.dn_max);
        if (dn < 0.0f) dn += 2.0f * params.dn_max;
        dn -= params.dn_max;
    } else {
        dn = clamp(dn, -params.dn_max, params.dn_max);
    }
    deltaN[gid] = dn;

    // Store updated moments
    momentM[gid] = m;
    momentV[gid] = v;
}
