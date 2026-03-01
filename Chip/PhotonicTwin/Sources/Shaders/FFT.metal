// FFT.metal — Batched Stockham FFT (concatenated after Common.metal)

// =============================================================================
// Batched Stockham FFT — radix-2, out-of-place, ping-pong
//
// Each kernel handles one butterfly pass. The Swift side dispatches log2(N)
// passes per dimension, alternating src/dst buffers (ping-pong).
// Batch dimension: each sample's field is at offset batch_idx * NX * NY.
// =============================================================================

/// One Stockham butterfly pass along rows (X dimension).
/// Threads: BATCH * NX * NY / 2 (1D dispatch)
kernel void fft_stockham_rows_batched(
    device const float2 *src   [[buffer(0)]],
    device       float2 *dst   [[buffer(1)]],
    constant     uint   &pass  [[buffer(2)]],  // pass index (0..log2(NX)-1)
    constant     float  &sign  [[buffer(3)]],  // +1.0 for forward, -1.0 for inverse
    uint gid [[thread_position_in_grid]])
{
    uint half_nx = NX / 2;
    uint total_per_batch = half_nx * NY;

    uint batch_idx = gid / total_per_batch;
    uint local_id = gid - batch_idx * total_per_batch;
    if (batch_idx >= BATCH) return;

    uint row = local_id / half_nx;
    uint tid = local_id - row * half_nx;

    uint stride = 1u << pass;
    uint half_stride = stride;
    uint group = tid / half_stride;
    uint pos = tid - group * half_stride;

    uint idx0 = group * (2 * half_stride) + pos;
    uint idx1 = idx0 + half_stride;

    // Twiddle factor: W_N^(k) = exp(-sign * 2πi * k / N_pass)
    uint n_pass = 1u << (pass + 1);
    float angle = sign * 2.0f * M_PI_F * float(pos) / float(n_pass);
    float2 w = float2(cos(angle), sin(angle));

    uint base = batch_idx * NX * NY + row * NX;
    // Stockham autosort: READ from contiguous halves, WRITE to interleaved
    float2 a = src[base + tid];
    float2 b = src[base + tid + half_nx];

    // Butterfly
    float2 wb = float2(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
    dst[base + idx0] = a + wb;
    dst[base + idx1] = a - wb;
}

/// One Stockham butterfly pass along columns (Y dimension).
/// Threads: BATCH * NX * NY / 2 (1D dispatch)
kernel void fft_stockham_cols_batched(
    device const float2 *src   [[buffer(0)]],
    device       float2 *dst   [[buffer(1)]],
    constant     uint   &pass  [[buffer(2)]],
    constant     float  &sign  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint half_ny = NY / 2;
    uint total_per_batch = NX * half_ny;

    uint batch_idx = gid / total_per_batch;
    uint local_id = gid - batch_idx * total_per_batch;
    if (batch_idx >= BATCH) return;

    uint col = local_id / half_ny;
    uint tid = local_id - col * half_ny;

    uint stride = 1u << pass;
    uint half_stride = stride;
    uint group = tid / half_stride;
    uint pos = tid - group * half_stride;

    uint idx0 = group * (2 * half_stride) + pos;
    uint idx1 = idx0 + half_stride;

    uint n_pass = 1u << (pass + 1);
    float angle = sign * 2.0f * M_PI_F * float(pos) / float(n_pass);
    float2 w = float2(cos(angle), sin(angle));

    uint base = batch_idx * NX * NY;
    // Stockham autosort: READ from contiguous halves, WRITE to interleaved
    float2 a = src[base + tid * NX + col];
    float2 b = src[base + (tid + half_ny) * NX + col];

    float2 wb = float2(w.x * b.x - w.y * b.y, w.x * b.y + w.y * b.x);
    dst[base + idx0 * NX + col] = a + wb;
    dst[base + idx1 * NX + col] = a - wb;
}

/// Normalize after inverse FFT: field *= 1/(NX*NY)
/// Threads: BATCH * NX * NY
kernel void fft_normalize_batched(
    device float2 *field [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;
    float scale = 1.0f / float(NX * NY);
    field[gid] *= scale;
}

/// Copy buffer (used when log2(N) is odd to land result in correct buffer)
/// Threads: BATCH * NX * NY
kernel void copy_complex_buffer(
    device const float2 *src [[buffer(0)]],
    device       float2 *dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= BATCH * NX * NY) return;
    dst[gid] = src[gid];
}
