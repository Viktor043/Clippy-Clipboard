#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Function constants — baked at pipeline compilation time
// =============================================================================

constant uint NX      [[function_constant(0)]];
constant uint NY      [[function_constant(1)]];
constant uint NZ      [[function_constant(2)]];
constant uint BATCH   [[function_constant(3)]];
constant uint NLEVELS [[function_constant(4)]];

// Derived constants (computed from function constants)
constant uint FIELD_SIZE   = NX * NY;
constant uint TOTAL_VOXELS = NX * NY * NZ;

// =============================================================================
// Physics parameters — passed as buffer
// =============================================================================

struct SimParams {
    float k0;           // 2π / wavelength
    float dz;           // slice thickness
    float dn_max;       // max refractive index change
    float n_base;       // background refractive index
    float temperature;  // Gumbel-softmax temperature
    float lr;           // learning rate
    float beta1;        // Adam β₁
    float beta2;        // Adam β₂
    float epsilon;      // Adam ε
    float sparsity_lambda;
    uint  step;         // current training step
    uint  seed;         // RNG seed
    float inv_batch;    // 1.0 / batchSize
    float grad_clip_max; // per-element gradient clipping (0 = disabled)
};

// =============================================================================
// Complex arithmetic (float2: .x = real, .y = imag)
// =============================================================================

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

inline float2 cexp(float phase) {
    float s, c;
    s = sincos(phase, c);
    return float2(c, s);
}

// =============================================================================
// Philox counter-based RNG (deterministic, parallel, no shared state)
// =============================================================================

inline uint mulhi32(uint a, uint b) {
    return uint((ulong(a) * ulong(b)) >> 32);
}

inline uint2 philox_round(uint2 ctr, uint2 key) {
    uint lo = ctr.x * 0xD2511F53u;
    uint hi = mulhi32(ctr.x, 0xD2511F53u);
    return uint2(hi ^ key.x ^ ctr.y, lo);
}

inline float philox_uniform(uint id, uint step, uint seed) {
    uint2 ctr = uint2(id, step);
    uint2 key = uint2(seed, 0x9E3779B9u);
    for (int i = 0; i < 10; i++) {
        ctr = philox_round(ctr, key);
        key.x += 0x9E3779B9u;
    }
    return float(ctr.x) * (1.0f / 4294967296.0f);
}

inline float gumbel_sample(uint id, uint step, uint seed) {
    float u = philox_uniform(id, step, seed);
    u = clamp(u, 1e-6f, 1.0f - 1e-6f);
    return -log(-log(u));
}

// =============================================================================
// Atomic float add (CAS loop — portable across all Metal GPUs)
// =============================================================================

inline void atomic_add_f32(device float *addr, float val) {
    device atomic_uint *uaddr = (device atomic_uint *)addr;
    uint expected = atomic_load_explicit(uaddr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        uint desired = as_type<uint>(current + val);
        if (atomic_compare_exchange_weak_explicit(uaddr, &expected, desired,
            memory_order_relaxed, memory_order_relaxed)) break;
    }
}
