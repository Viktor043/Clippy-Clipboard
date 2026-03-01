// IO.metal — Input encoding + Loss/Gradient (concatenated after Common.metal)

// =============================================================================
// Encode input: generate Gaussian spot patterns on the input face
//
// 3D dispatch: (NX, NY, BATCH)
// Each thread computes one pixel's complex field value for one sample.
// =============================================================================

struct IOParams {
    uint n_inputs;         // total input channels
    float spot_sigma;      // Gaussian spot width
    uint use_phase_encoding;  // 0 = amplitude, 1 = phase encoding (value → phase angle)
};

kernel void encode_input_batched(
    device       float2 *field        [[buffer(0)]],   // [B * NX * NY] output
    device const float  *input_values [[buffer(1)]],   // [B * n_inputs]
    device const float2 *spot_centers [[buffer(2)]],   // [n_inputs] (cx, cy)
    constant IOParams   &io_params    [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    uint b = gid.z;
    if (x >= NX || y >= NY || b >= BATCH) return;

    float sigma = io_params.spot_sigma;
    float two_sigma_sq = 2.0f * sigma * sigma;
    float cutoff_sq = 16.0f * two_sigma_sq;  // 4 sigma cutoff
    uint phase_enc = io_params.use_phase_encoding;

    float real_sum = 0.0f;
    float imag_sum = 0.0f;

    for (uint i = 0; i < io_params.n_inputs; i++) {
        float cx = spot_centers[i].x;
        float cy = spot_centers[i].y;
        float dx = float(x) - cx;
        float dy = float(y) - cy;
        float r_sq = dx * dx + dy * dy;

        if (r_sq < cutoff_sq) {
            float value = input_values[b * io_params.n_inputs + i];
            float envelope = exp(-r_sq / two_sigma_sq);

            if (phase_enc) {
                // Phase encoding (Ozcan 2024): input value → phase angle
                // Gaussian envelope carries constant amplitude, value encodes as phase
                // value in [0,1] → phase in [0, π] (half rotation)
                float phase = value * 3.14159265f;
                real_sum += envelope * cos(phase);
                imag_sum += envelope * sin(phase);
            } else {
                // Amplitude encoding (default): input value modulates amplitude
                real_sum += value * envelope;
            }
        }
    }

    uint idx = b * NX * NY + x * NY + y;
    field[idx] = float2(real_sum, imag_sum);
}

// =============================================================================
// Encode image input: stamp a full 2D image onto the optical field
//
// For UCLA-style D2NN: the MNIST image IS the input field amplitude.
// Images are resized to NX×NY on CPU, then this kernel copies them to the field.
// 3D dispatch: (NX, NY, BATCH)
// =============================================================================

kernel void encode_image_input_batched(
    device       float2 *field        [[buffer(0)]],   // [B * NX * NY] output
    device const float  *image_data   [[buffer(1)]],   // [B * NX * NY] pixel values (0..1)
    constant     uint   &use_phase    [[buffer(2)]],   // 0 = amplitude, 1 = phase encoding
    uint3 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;
    uint b = gid.z;
    if (x >= NX || y >= NY || b >= BATCH) return;

    uint idx = b * NX * NY + x * NY + y;
    float value = image_data[idx];

    if (use_phase) {
        // Phase encoding: constant amplitude, value → phase in [0, π]
        float phase = value * 3.14159265f;
        field[idx] = float2(cos(phase), sin(phase));
    } else {
        // Amplitude encoding: value IS the field amplitude (real part)
        field[idx] = float2(value, 0.0f);
    }
}

// =============================================================================
// Fused loss + output gradient computation
//
// Reads the output field, computes detector intensities, softmax CE loss,
// and writes the adjoint seed — all in one kernel.
//
// Threads: BATCH (1D dispatch, one thread per sample)
// =============================================================================

struct DetectorParams {
    uint n_outputs;        // number of output classes (nOutputsA)
    uint n_detectors;      // number of physical detector regions (n_outputs or 2*n_outputs)
    uint use_differential; // 0 = standard, 1 = differential (2 detectors per class: pos-neg)
    float norm_scale;      // score amplification before softmax (live-tunable)
};

kernel void compute_loss_and_gradient_batched(
    device const float2 *field          [[buffer(0)]],    // [B * NX * NY]
    device       float2 *adjoint        [[buffer(1)]],    // [B * NX * NY] output: adjoint seed
    device       float  *losses         [[buffer(2)]],    // [B]
    device       uint   *accuracy_count [[buffer(3)]],    // [1] atomic counter
    device const uint   *target_classes [[buffer(4)]],    // [B]
    device const uint   *mask_pixels    [[buffer(5)]],    // flattened pixel indices
    device const uint   *mask_offsets   [[buffer(6)]],    // [n_outputs + 1] prefix sums
    constant DetectorParams &det_params [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH) return;

    uint b = gid;
    uint target = target_classes[b];
    uint n_out = det_params.n_outputs;       // number of classes (e.g. 4)
    uint n_det = det_params.n_detectors;     // physical detectors (4 or 8)
    uint differential = det_params.use_differential;
    uint field_base = b * NX * NY;

    // Step 1: Integrate ALL physical detector intensities
    float intensities[20];  // max 10 classes × 2 (differential)
    for (uint d = 0; d < n_det; d++) {
        float sum_intensity = 0.0f;
        uint start = mask_offsets[d];
        uint end = mask_offsets[d + 1];
        for (uint p = start; p < end; p++) {
            uint pixel = mask_pixels[p];
            float2 f = field[field_base + pixel];
            sum_intensity += f.x * f.x + f.y * f.y;
        }
        intensities[d] = sum_intensity;
    }

    // Step 1b: Compute class scores
    // Standard: scores[c] = intensities[c]
    // Differential: scores[c] = intensities[2*c] - intensities[2*c+1]
    float scores[10];  // max 10 classes
    if (differential) {
        for (uint c = 0; c < n_out; c++) {
            scores[c] = intensities[2 * c] - intensities[2 * c + 1];
        }
    } else {
        for (uint c = 0; c < n_out; c++) {
            scores[c] = intensities[c];
        }
    }

    // Step 2: Score normalization + Softmax CE with label smoothing
    // Use a FIXED scale factor to amplify small score differences for effective
    // softmax discrimination. Unlike the previous data-dependent normalization
    // (S_MAX / abs_scale), a fixed scale is fully differentiable — the backward
    // pass gradient is simply scaled by the same constant.
    //
    float norm_scale = det_params.norm_scale;
    float norm_scores[10];
    for (uint c = 0; c < n_out; c++) {
        norm_scores[c] = scores[c] * norm_scale;
    }

    float alpha = 0.1f;
    float max_S = norm_scores[0];
    for (uint c = 1; c < n_out; c++) {
        max_S = max(max_S, norm_scores[c]);
    }

    float sum_exp = 0.0f;
    float softmax_vals[10];
    for (uint c = 0; c < n_out; c++) {
        softmax_vals[c] = exp(norm_scores[c] - max_S);
        sum_exp += softmax_vals[c];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-10f);
    for (uint c = 0; c < n_out; c++) {
        softmax_vals[c] *= inv_sum;
    }

    // Label-smoothed CE
    float loss = 0.0f;
    for (uint c = 0; c < n_out; c++) {
        float smooth_t = (c == target) ? (1.0f - alpha + alpha / float(n_out)) : (alpha / float(n_out));
        loss -= smooth_t * log(softmax_vals[c] + 1e-10f);
    }
    losses[b] = loss;

    // Accuracy: argmax of scores (not raw intensities)
    uint predicted = 0;
    float max_val = scores[0];
    for (uint c = 1; c < n_out; c++) {
        if (scores[c] > max_val) {
            max_val = scores[c];
            predicted = c;
        }
    }
    if (predicted == target) {
        atomic_fetch_add_explicit((device atomic_uint *)accuracy_count, 1u, memory_order_relaxed);
    }

    // Step 3: Compute adjoint seed dL/dh* for each output pixel
    // Zero this sample's adjoint region
    for (uint p = 0; p < NX * NY; p++) {
        adjoint[field_base + p] = float2(0.0f, 0.0f);
    }

    // dL/d_norm_scores[c] = softmax[c] - smooth_target[c]
    // Chain rule: dL/d_scores[c] = dL/d_norm_scores[c] * norm_scale (exact, fully differentiable)
    for (uint c = 0; c < n_out; c++) {
        float smooth_t = (c == target) ? (1.0f - alpha + alpha / float(n_out)) : (alpha / float(n_out));
        float dL_dscore = (softmax_vals[c] - smooth_t) * norm_scale;

        if (differential) {
            // Positive detector (index 2*c): dL/dI = +dL/dscore
            {
                uint d = 2 * c;
                uint start = mask_offsets[d];
                uint end = mask_offsets[d + 1];
                for (uint p = start; p < end; p++) {
                    uint pixel = mask_pixels[p];
                    float2 h = field[field_base + pixel];
                    adjoint[field_base + pixel] += float2(dL_dscore * h.x, dL_dscore * h.y);
                }
            }
            // Negative detector (index 2*c+1): dL/dI = -dL/dscore
            {
                uint d = 2 * c + 1;
                uint start = mask_offsets[d];
                uint end = mask_offsets[d + 1];
                for (uint p = start; p < end; p++) {
                    uint pixel = mask_pixels[p];
                    float2 h = field[field_base + pixel];
                    adjoint[field_base + pixel] += float2(-dL_dscore * h.x, -dL_dscore * h.y);
                }
            }
        } else {
            // Standard: one detector per class
            uint start = mask_offsets[c];
            uint end = mask_offsets[c + 1];
            for (uint p = start; p < end; p++) {
                uint pixel = mask_pixels[p];
                float2 h = field[field_base + pixel];
                adjoint[field_base + pixel] += float2(dL_dscore * h.x, dL_dscore * h.y);
            }
        }
    }
}

// =============================================================================
// MSE loss for regression mode (e.g., optical FFT glass block)
//
// Computes per-sample MSE = mean(|E_out - E_target|²) over all pixels.
// Seeds the adjoint with dL/dE = 2*(E_out - E_target) / (NX*NY).
//
// Threads: BATCH (1D dispatch, one thread per sample)
// =============================================================================

kernel void compute_mse_loss_batched(
    device const float2 *field          [[buffer(0)]],    // [B * NX * NY] output field
    device       float2 *adjoint        [[buffer(1)]],    // [B * NX * NY] adjoint seed (output)
    device       float  *losses         [[buffer(2)]],    // [B] per-sample MSE
    device       uint   *accuracy_count [[buffer(3)]],    // unused in regression, zeroed
    device const float2 *target_field   [[buffer(4)]],    // [B * NX * NY] target (e.g. exact FFT)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH) return;

    uint b = gid;
    uint base = b * NX * NY;
    float inv_n = 1.0f / float(NX * NY);

    float mse_sum = 0.0f;

    for (uint p = 0; p < NX * NY; p++) {
        float2 out_val = field[base + p];
        float2 tgt_val = target_field[base + p];
        float2 diff = out_val - tgt_val;

        // Accumulate |diff|²
        mse_sum += diff.x * diff.x + diff.y * diff.y;

        // Adjoint seed: d(SSE)/dE = 2*(E - E_target)
        // NOT divided by N — avoids vanishing gradients on large grids.
        // The loss is still reported as MSE (divided by N) for readability.
        adjoint[base + p] = 2.0f * diff;
    }

    losses[b] = mse_sum * inv_n;
}
