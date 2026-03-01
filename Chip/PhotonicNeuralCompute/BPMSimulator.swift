// Photonic Neural Compute — BPM Simulator
// SNRGY Studios — February 2026
//
// Handles BPM forward/backward pass dispatch with GPU-native FFT.
//
// Performance architecture:
//   - ALL z-slice dispatches encoded into ONE command buffer (zero CPU↔GPU sync)
//   - Stockham FFT runs entirely on GPU (no vDSP, no data transfer)
//   - Ping-pong between fieldBuffer and fftTemp for FFT passes
//   - For 64x64x40: ~1200 dispatches per forward pass, ONE commit

import Foundation
import Metal
import Accelerate

class BPMSimulator {
    let config: SimConfig
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let grid: VoxelGrid

    // Metal pipelines
    var phaseScreenPipeline: MTLComputePipelineState!
    var elemMulPipeline: MTLComputePipelineState!
    var elemMulConjPipeline: MTLComputePipelineState!
    var cacheFieldPipeline: MTLComputePipelineState!
    var gradientPipeline: MTLComputePipelineState!
    var gumbelFwdPipeline: MTLComputePipelineState!
    var gumbelBwdPipeline: MTLComputePipelineState!
    var adamPipeline: MTLComputePipelineState!
    var zeroPipeline: MTLComputePipelineState!
    var zeroComplexPipeline: MTLComputePipelineState!

    // GPU FFT pipelines
    var fftRowsPipeline: MTLComputePipelineState!
    var fftColsPipeline: MTLComputePipelineState!
    var fftNormalizePipeline: MTLComputePipelineState!

    // FFT parameters
    let log2nx: Int
    let log2ny: Int

    init(device: MTLDevice, commandQueue: MTLCommandQueue, grid: VoxelGrid, config: SimConfig) {
        self.config = config
        self.device = device
        self.commandQueue = commandQueue
        self.grid = grid

        // Verify power-of-2 dimensions for FFT
        log2nx = Int(log2(Double(config.nx)))
        log2ny = Int(log2(Double(config.ny)))
        precondition(1 << log2nx == config.nx, "NX must be a power of 2 for GPU FFT")
        precondition(1 << log2ny == config.ny, "NY must be a power of 2 for GPU FFT")
    }

    // MARK: - Shader Compilation

    func compilePipelines(library: MTLLibrary, constants: MTLFunctionConstantValues) throws {
        func make(_ name: String) throws -> MTLComputePipelineState {
            let fn = try library.makeFunction(name: name, constantValues: constants)
            return try device.makeComputePipelineState(function: fn)
        }

        phaseScreenPipeline  = try make("phase_screen_forward")
        elemMulPipeline      = try make("elementwise_multiply")
        elemMulConjPipeline  = try make("elementwise_multiply_conj")
        cacheFieldPipeline   = try make("cache_field")
        gradientPipeline     = try make("compute_gradient_and_reverse_phase")
        gumbelFwdPipeline    = try make("gumbel_softmax_forward")
        gumbelBwdPipeline    = try make("gumbel_softmax_backward")
        adamPipeline         = try make("adam_update")
        zeroPipeline         = try make("zero_buffer")
        zeroComplexPipeline  = try make("zero_complex_buffer")

        // GPU FFT pipelines
        fftRowsPipeline      = try make("fft_stockham_rows")
        fftColsPipeline      = try make("fft_stockham_cols")
        fftNormalizePipeline = try make("fft_normalize")
    }

    // MARK: - Forward Pass (ALL z-slices in ONE command buffer)

    /// Run full BPM forward propagation with batched GPU encoding.
    /// Caches fields at each slice boundary for the adjoint backward pass.
    func forwardPass() {
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        // Cache input field (before slice 0)
        encodeCacheField(encoder: encoder, zSlice: 0, source: grid.fieldBuffer)

        for z in 0..<config.nz {
            // 1. Phase screen: field *= exp(i * k0 * dn[z] * dz)
            encodePhaseScreen(encoder: encoder, zSlice: z)

            // 2. Propagation: field = IFFT2(FFT2(field) * H)
            encodeFFT2D(encoder: encoder, buffer: grid.fieldBuffer, inverse: false)
            encodeElementwiseMultiply(encoder: encoder, buffer: grid.fieldBuffer, conjugate: false)
            encodeFFT2D(encoder: encoder, buffer: grid.fieldBuffer, inverse: true)

            // Cache field after this slice
            encodeCacheField(encoder: encoder, zSlice: z + 1, source: grid.fieldBuffer)
        }

        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Backward Pass (ALL z-slices in ONE command buffer)

    /// Run adjoint backward pass with batched GPU encoding.
    /// Assumes grid.adjointBuffer contains ∂L/∂h* (loss gradient w.r.t. output field).
    /// Gradients are ACCUMULATED into grid.gradDeltaN.
    func backwardPass() {
        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!

        for z in stride(from: config.nz - 1, through: 0, by: -1) {
            // 1. Reverse propagation: adjoint = IFFT2(FFT2(adjoint) * conj(H))
            encodeFFT2D(encoder: encoder, buffer: grid.adjointBuffer, inverse: false)
            encodeElementwiseMultiply(encoder: encoder, buffer: grid.adjointBuffer, conjugate: true)
            encodeFFT2D(encoder: encoder, buffer: grid.adjointBuffer, inverse: true)

            // 2. Compute gradient and reverse phase screen
            encodeGradientAndReversePhase(encoder: encoder, zSlice: z)
        }

        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - GPU FFT (Stockham, all on GPU)

    /// Encode a full 2D FFT (or inverse FFT) into the command encoder.
    /// Uses Stockham autosort with ping-pong between the target buffer and grid.fftTemp.
    ///
    /// For log2(N) passes, the Stockham algorithm alternates src/dst:
    ///   Pass 0: buffer → temp
    ///   Pass 1: temp → buffer
    ///   Pass 2: buffer → temp
    ///   ...
    ///   Pass log2(N)-1: if log2(N) is even, ends in buffer. If odd, ends in temp → need copy.
    ///
    /// For NX=NY=64: log2(64) = 6 passes per dimension, result ends in buffer. ✓
    private func encodeFFT2D(encoder: MTLComputeCommandEncoder, buffer: MTLBuffer, inverse: Bool) {
        var sign: Float = inverse ? 1.0 : -1.0  // Forward: exp(-i...), Inverse: exp(+i...)
        let halfField = config.fieldSize / 2
        let tgs1D = MTLSize(width: min(256, halfField), height: 1, depth: 1)
        let gs1D = MTLSize(width: halfField, height: 1, depth: 1)
        let temp = grid.fftTemp

        // --- Row FFT (along y-dimension): log2(NY) passes ---
        for s in 0..<log2ny {
            var pass = UInt32(s)
            let src = (s % 2 == 0) ? buffer : temp
            let dst = (s % 2 == 0) ? temp : buffer

            encoder.setComputePipelineState(fftRowsPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(dst, offset: 0, index: 1)
            encoder.setBytes(&pass, length: 4, index: 2)
            encoder.setBytes(&sign, length: 4, index: 3)
            encoder.dispatchThreads(gs1D, threadsPerThreadgroup: tgs1D)
        }
        // After log2ny passes: if log2ny is even → result in buffer, if odd → result in temp
        if log2ny % 2 != 0 {
            // Copy temp → buffer
            encoder.setComputePipelineState(cacheFieldPipeline)
            // Reuse cache_field as a copy: src=temp, dst=buffer at z_slice=0 (just for the copy)
            // Actually, let me just add a memcpy-style dispatch
            encodeCopy(encoder: encoder, src: temp, dst: buffer)
        }

        // --- Column FFT (along x-dimension): log2(NX) passes ---
        for s in 0..<log2nx {
            var pass = UInt32(s)
            let src = (s % 2 == 0) ? buffer : temp
            let dst = (s % 2 == 0) ? temp : buffer

            encoder.setComputePipelineState(fftColsPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(dst, offset: 0, index: 1)
            encoder.setBytes(&pass, length: 4, index: 2)
            encoder.setBytes(&sign, length: 4, index: 3)
            encoder.dispatchThreads(gs1D, threadsPerThreadgroup: tgs1D)
        }
        if log2nx % 2 != 0 {
            encodeCopy(encoder: encoder, src: temp, dst: buffer)
        }

        // For inverse FFT: normalize by 1/(NX*NY)
        if inverse {
            encoder.setComputePipelineState(fftNormalizePipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            let fieldTgs = MTLSize(width: min(256, config.fieldSize), height: 1, depth: 1)
            encoder.dispatchThreads(
                MTLSize(width: config.fieldSize, height: 1, depth: 1),
                threadsPerThreadgroup: fieldTgs
            )
        }
    }

    /// Copy one complex buffer to another (using zero_complex_buffer pattern — just a flat copy)
    private func encodeCopy(encoder: MTLComputeCommandEncoder, src: MTLBuffer, dst: MTLBuffer) {
        // Simple elementwise copy: dst[i] = src[i] * 1
        // Reuse elementwise_multiply with propagator=1+0i? No, let's just do it directly.
        // Use the fft_normalize kernel's pattern but copy instead:
        // Actually the simplest: use elementwise_multiply where H = (1, 0).
        // But we'd need a buffer of ones. Instead, use a blit after ending the encoder... no.
        //
        // Simplest approach: just use the row FFT kernel with pass=0, sign=0 → twiddle=1
        // That would be: dst[k] = src[k] + 1*src[k+N/2] and dst[k+N/2] = src[k] - src[k+N/2]
        // which is NOT a copy.
        //
        // For now, use memcpy on CPU (only happens when log2(N) is odd, e.g. N=32, 128)
        // For NX=NY=64: log2(64) = 6 (EVEN), so this code is never reached. ✓
        // For safety, add an assertion.
        preconditionFailure("FFT ping-pong copy needed — dimension is not a power of 2^(2k). Use NX, NY = 4, 16, 64, 256, 1024.")
    }

    // MARK: - Gumbel-Softmax (GPU)

    /// Convert logits to continuous delta_n values via Gumbel-softmax on GPU.
    func computeDeltaN(temperature: Float) {
        var temp = temperature

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(gumbelFwdPipeline)
        encoder.setBuffer(grid.logits, offset: 0, index: 0)
        encoder.setBuffer(grid.deltaN, offset: 0, index: 1)
        encoder.setBuffer(grid.dnLevelsBuffer, offset: 0, index: 2)
        encoder.setBytes(&temp, length: 4, index: 3)

        let total = config.totalVoxels
        let tgs = MTLSize(width: min(256, total), height: 1, depth: 1)
        encoder.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                threadsPerThreadgroup: tgs)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    /// Backpropagate gradients from delta_n to logits via Gumbel-softmax Jacobian on GPU.
    /// Buffer order must match shader: logits(0), delta_n(1), grad_delta_n(2), grad_logits(3), dn_levels(4), temp(5)
    func backpropGumbelSoftmax(temperature: Float) {
        var temp = temperature

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(gumbelBwdPipeline)
        encoder.setBuffer(grid.logits, offset: 0, index: 0)
        encoder.setBuffer(grid.deltaN, offset: 0, index: 1)
        encoder.setBuffer(grid.gradDeltaN, offset: 0, index: 2)
        encoder.setBuffer(grid.gradLogits, offset: 0, index: 3)
        encoder.setBuffer(grid.dnLevelsBuffer, offset: 0, index: 4)
        encoder.setBytes(&temp, length: 4, index: 5)

        let total = config.totalVoxels
        let tgs = MTLSize(width: min(256, total), height: 1, depth: 1)
        encoder.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                threadsPerThreadgroup: tgs)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Adam Optimizer (GPU)

    func adamUpdate(step: Int) {
        var lr = config.lr
        var beta1 = config.adam_beta1
        var beta2 = config.adam_beta2
        var eps = config.adam_eps
        let t = Float(step + 1)
        var mCorr = 1.0 / (1.0 - pow(beta1, t))
        var vCorr = 1.0 / (1.0 - pow(beta2, t))
        var sparsity = config.sparsityLambda

        let total = config.totalVoxels * config.nLevels

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let encoder = cmdBuf.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(adamPipeline)
        encoder.setBuffer(grid.logits, offset: 0, index: 0)
        encoder.setBuffer(grid.gradLogits, offset: 0, index: 1)
        encoder.setBuffer(grid.momentM, offset: 0, index: 2)
        encoder.setBuffer(grid.momentV, offset: 0, index: 3)
        encoder.setBytes(&lr, length: 4, index: 4)
        encoder.setBytes(&beta1, length: 4, index: 5)
        encoder.setBytes(&beta2, length: 4, index: 6)
        encoder.setBytes(&eps, length: 4, index: 7)
        encoder.setBytes(&mCorr, length: 4, index: 8)
        encoder.setBytes(&vCorr, length: 4, index: 9)
        encoder.setBytes(&sparsity, length: 4, index: 10)
        encoder.setBuffer(grid.deltaN, offset: 0, index: 11)  // For L1 sparsity term

        let tgs = MTLSize(width: min(256, total), height: 1, depth: 1)
        encoder.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                threadsPerThreadgroup: tgs)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Encode Helpers (into existing command encoder)

    private var threadgroupSize2D: MTLSize {
        MTLSize(width: 16, height: 16, depth: 1)
    }

    private var gridSize2D: MTLSize {
        MTLSize(
            width: (config.nx + 15) / 16 * 16,
            height: (config.ny + 15) / 16 * 16,
            depth: 1
        )
    }

    private func encodePhaseScreen(encoder: MTLComputeCommandEncoder, zSlice: Int) {
        encoder.setComputePipelineState(phaseScreenPipeline)
        encoder.setBuffer(grid.fieldBuffer, offset: 0, index: 0)
        encoder.setBuffer(grid.deltaN, offset: 0, index: 1)
        var z = UInt32(zSlice)
        encoder.setBytes(&z, length: 4, index: 2)
        encoder.dispatchThreads(gridSize2D, threadsPerThreadgroup: threadgroupSize2D)
    }

    private func encodeElementwiseMultiply(
        encoder: MTLComputeCommandEncoder, buffer: MTLBuffer, conjugate: Bool
    ) {
        encoder.setComputePipelineState(conjugate ? elemMulConjPipeline : elemMulPipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(grid.propagator, offset: 0, index: 1)
        encoder.dispatchThreads(gridSize2D, threadsPerThreadgroup: threadgroupSize2D)
    }

    private func encodeCacheField(
        encoder: MTLComputeCommandEncoder, zSlice: Int, source: MTLBuffer
    ) {
        encoder.setComputePipelineState(cacheFieldPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        encoder.setBuffer(grid.fieldCache, offset: 0, index: 1)
        var z = UInt32(zSlice)
        encoder.setBytes(&z, length: 4, index: 2)
        encoder.dispatchThreads(gridSize2D, threadsPerThreadgroup: threadgroupSize2D)
    }

    private func encodeGradientAndReversePhase(
        encoder: MTLComputeCommandEncoder, zSlice: Int
    ) {
        encoder.setComputePipelineState(gradientPipeline)
        encoder.setBuffer(grid.adjointBuffer, offset: 0, index: 0)
        encoder.setBuffer(grid.fieldCache, offset: 0, index: 1)
        encoder.setBuffer(grid.deltaN, offset: 0, index: 2)
        encoder.setBuffer(grid.gradDeltaN, offset: 0, index: 3)
        var z = UInt32(zSlice)
        encoder.setBytes(&z, length: 4, index: 4)
        encoder.dispatchThreads(gridSize2D, threadsPerThreadgroup: threadgroupSize2D)
    }

    // MARK: - Buffer Utilities

    func zeroGradients() {
        // Zero delta_n gradients
        let dnPtr = grid.gradDeltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels)
        memset(dnPtr, 0, config.totalVoxels * 4)

        // Zero logit gradients
        let logitPtr = grid.gradLogits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels)
        memset(logitPtr, 0, config.totalVoxels * config.nLevels * 4)
    }

    func scaleGradients(by factor: Float) {
        let ptr = grid.gradDeltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels)
        var scale = factor
        vDSP_vsmul(ptr, 1, &scale, ptr, 1, vDSP_Length(config.totalVoxels))
    }

    func copyField(from src: MTLBuffer, to dst: MTLBuffer) {
        let bytes = config.fieldSize * 8  // complex float = 8 bytes
        memcpy(dst.contents(), src.contents(), bytes)
    }
}
