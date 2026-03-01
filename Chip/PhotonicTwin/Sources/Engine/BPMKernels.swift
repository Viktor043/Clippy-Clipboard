import Metal
import Foundation

/// Encodes BPM forward/backward passes and FFT operations into Metal command buffers.
/// All dispatch sizes account for the batch dimension.
extension MetalEngine {

    // MARK: - Threadgroup sizes

    private var threads1D: Int { 256 }

    private var fieldThreads: MTLSize {
        MTLSize(width: config.nx, height: config.ny, depth: config.batchSize)
    }
    private var fieldThreadgroup: MTLSize {
        MTLSize(width: 16, height: 16, depth: 1)
    }

    private var batchFieldCount: Int { config.batchSize * config.fieldSize }

    private func threads1DGrid(_ count: Int) -> MTLSize {
        MTLSize(width: count, height: 1, depth: 1)
    }
    private func threads1DGroup(_ count: Int) -> MTLSize {
        MTLSize(width: min(threads1D, count), height: 1, depth: 1)
    }

    // MARK: - FFT encoding

    /// Encode a batched 2D FFT (forward or inverse) into the command encoder.
    /// Uses Stockham radix-2 ping-pong between `field` and `temp`.
    /// After completion, result is in `field` buffer.
    func encodeFFT2D(
        encoder: MTLComputeCommandEncoder,
        field: MTLBuffer,
        temp: MTLBuffer,
        inverse: Bool
    ) {
        let sign: Float = inverse ? -1.0 : 1.0
        assert(config.nx == config.ny, "Non-square grids require separate row/col FFT logic")
        let log2nx = Int(log2(Double(config.nx)))

        var src = field
        var dst = temp

        // Row FFTs: log2(NX) passes (NX == NY required for correctness)
        let rowThreads = config.batchSize * config.nx * config.ny / 2
        for pass in 0..<log2nx {
            var passVal = UInt32(pass)
            var signVal = sign

            encoder.setComputePipelineState(fftRowsPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(dst, offset: 0, index: 1)
            encoder.setBytes(&passVal, length: 4, index: 2)
            encoder.setBytes(&signVal, length: 4, index: 3)
            encoder.dispatchThreads(
                threads1DGrid(rowThreads),
                threadsPerThreadgroup: threads1DGroup(rowThreads)
            )

            swap(&src, &dst)
        }

        // If odd number of row passes, result is in temp — copy back to field
        if log2nx % 2 != 0 {
            encoder.setComputePipelineState(copyBufferPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(field, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
            src = field
            dst = temp
        } else {
            // Result is already in field after even passes
            src = field
            dst = temp
        }

        // Column FFTs: log2(NX) passes (NX == NY required for correctness)
        let colThreads = config.batchSize * config.nx * config.ny / 2
        for pass in 0..<log2nx {
            var passVal = UInt32(pass)
            var signVal = sign

            encoder.setComputePipelineState(fftColsPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(dst, offset: 0, index: 1)
            encoder.setBytes(&passVal, length: 4, index: 2)
            encoder.setBytes(&signVal, length: 4, index: 3)
            encoder.dispatchThreads(
                threads1DGrid(colThreads),
                threadsPerThreadgroup: threads1DGroup(colThreads)
            )

            swap(&src, &dst)
        }

        // If odd number of col passes, copy back
        if log2nx % 2 != 0 {
            encoder.setComputePipelineState(copyBufferPipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(field, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
        }

        // Normalize for inverse FFT
        if inverse {
            encoder.setComputePipelineState(fftNormalizePipeline)
            encoder.setBuffer(field, offset: 0, index: 0)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
        }
    }

    // MARK: - Forward BPM pass

    /// Byte stride between pulses in the fieldCache buffer.
    /// Layout: [nPulses][B * (NZ+1) * NX * NY] float2
    var pulseCacheStride: Int {
        config.batchSize * (config.nz + 1) * config.fieldSize * 8
    }

    /// Encode the full forward BPM pass for all batch samples.
    /// Phase screen → FFT → propagator multiply → IFFT, repeated for each z-slice.
    /// Caches fields at each slice for the adjoint backward pass.
    /// pulseIndex offsets into fieldCache for recurrent mode (default 0).
    func encodeForwardPass(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        step: Int,
        effectiveNZ: Int? = nil,
        pulseIndex: Int = 0
    ) {
        let nz = effectiveNZ ?? config.nz
        var bpmParams = BPMParamsGPU(k0: Float(config.k0), dz: Float(config.dz), I_sat: config.saturationIntensity)
        let cacheOffset = pulseIndex * pulseCacheStride

        for z in 0..<nz {
            // Cache field before this slice
            var zIdx = UInt32(z)
            encoder.setComputePipelineState(cacheFieldPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(frame.fieldCache, offset: cacheOffset, index: 1)
            encoder.setBytes(&zIdx, length: 4, index: 2)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )

            // Phase screen (with saturable absorption)
            encoder.setComputePipelineState(phaseScreenPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(deltaN, offset: 0, index: 1)
            encoder.setBytes(&zIdx, length: 4, index: 2)
            encoder.setBytes(&bpmParams, length: MemoryLayout<BPMParamsGPU>.size, index: 3)
            encoder.setBuffer(deltaKappa, offset: 0, index: 4)
            encoder.dispatchThreads(fieldThreads, threadsPerThreadgroup: fieldThreadgroup)

            // FFT forward
            encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: false)

            // Multiply by propagator
            encoder.setComputePipelineState(elemMulPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(propagator, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )

            // IFFT
            encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: true)
        }

        // Cache final output field (z = nz)
        var nzIdx = UInt32(nz)
        encoder.setComputePipelineState(cacheFieldPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.fieldCache, offset: cacheOffset, index: 1)
        encoder.setBytes(&nzIdx, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    // MARK: - Backward BPM pass (adjoint method)

    /// Encode the backward adjoint BPM z-loop only (no reduce/scale).
    /// Computes per-batch gradients for one pulse. Writes to per_batch_grad buffers.
    func encodeBackwardBPM(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        effectiveNZ: Int? = nil,
        pulseIndex: Int = 0
    ) {
        let nz = effectiveNZ ?? config.nz
        var bpmParams = BPMParamsGPU(k0: Float(config.k0), dz: Float(config.dz), I_sat: config.saturationIntensity)
        let cacheOffset = pulseIndex * pulseCacheStride

        for z in stride(from: nz - 1, through: 0, by: -1) {
            encodeFFT2D(encoder: encoder, field: frame.adjoint, temp: frame.fftTemp, inverse: false)

            encoder.setComputePipelineState(elemMulConjPipeline)
            encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
            encoder.setBuffer(propagator, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )

            encodeFFT2D(encoder: encoder, field: frame.adjoint, temp: frame.fftTemp, inverse: true)

            var zIdx = UInt32(z)
            encoder.setComputePipelineState(gradientReversePhasePipeline)
            encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
            encoder.setBuffer(frame.fieldCache, offset: cacheOffset, index: 1)
            encoder.setBuffer(deltaN, offset: 0, index: 2)
            encoder.setBuffer(frame.perBatchGrad, offset: 0, index: 3)
            encoder.setBytes(&zIdx, length: 4, index: 4)
            encoder.setBytes(&bpmParams, length: MemoryLayout<BPMParamsGPU>.size, index: 5)
            encoder.setBuffer(deltaKappa, offset: 0, index: 6)
            encoder.setBuffer(frame.perBatchGradKappa, offset: 0, index: 7)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
        }
    }

    /// Reduce per-batch gradients and accumulate into gradDeltaN/gradDeltaKappa.
    /// Uses += so can be called multiple times for recurrent pulses.
    func encodeReduceGradients(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        effectiveNZ: Int? = nil
    ) {
        let nz = effectiveNZ ?? config.nz
        let activeVoxels = nz * config.fieldSize
        var batchSize = UInt32(config.batchSize)

        encoder.setComputePipelineState(reduceBatchGradsPipeline)
        encoder.setBuffer(frame.perBatchGrad, offset: 0, index: 0)
        encoder.setBuffer(gradDeltaN, offset: 0, index: 1)
        encoder.setBytes(&batchSize, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(activeVoxels),
            threadsPerThreadgroup: threads1DGroup(activeVoxels)
        )

        encoder.setComputePipelineState(reduceBatchGradsPipeline)
        encoder.setBuffer(frame.perBatchGradKappa, offset: 0, index: 0)
        encoder.setBuffer(gradDeltaKappa, offset: 0, index: 1)
        encoder.setBytes(&batchSize, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(activeVoxels),
            threadsPerThreadgroup: threads1DGroup(activeVoxels)
        )
    }

    /// Scale gradients by 1/batchSize.
    func encodeScaleGradients(
        encoder: MTLComputeCommandEncoder,
        effectiveNZ: Int? = nil
    ) {
        let nz = effectiveNZ ?? config.nz
        let activeVoxels = nz * config.fieldSize
        var scale = 1.0 / Float(config.batchSize)
        var activeCount = UInt32(activeVoxels)

        encoder.setComputePipelineState(scaleBufferPipeline)
        encoder.setBuffer(gradDeltaN, offset: 0, index: 0)
        encoder.setBytes(&scale, length: 4, index: 1)
        encoder.setBytes(&activeCount, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(activeVoxels),
            threadsPerThreadgroup: threads1DGroup(activeVoxels)
        )

        encoder.setComputePipelineState(scaleBufferPipeline)
        encoder.setBuffer(gradDeltaKappa, offset: 0, index: 0)
        encoder.setBytes(&scale, length: 4, index: 1)
        encoder.setBytes(&activeCount, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(activeVoxels),
            threadsPerThreadgroup: threads1DGroup(activeVoxels)
        )
    }

    /// Combined backward pass (single-pulse convenience wrapper).
    func encodeBackwardPass(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        effectiveNZ: Int? = nil
    ) {
        encodeBackwardBPM(encoder: encoder, frame: frame, effectiveNZ: effectiveNZ)
        encodeReduceGradients(encoder: encoder, frame: frame, effectiveNZ: effectiveNZ)
        encodeScaleGradients(encoder: encoder, effectiveNZ: effectiveNZ)
    }

    // MARK: - Discrete D2NN forward/backward passes
    // Reuses existing phase screen and propagator kernels.
    // Key trick: pass BPMParams(k0=1, dz=1) so phase screen applies exp(i*deltaN) directly.
    // The propagator uses the precomputed propagator buffer (which uses discreteLayerSpacing).
    // Architecture: input → [propagate] → L0 → [propagate] → L1 → ... → Ln-1 → [propagate] → output
    // That's nLayers + 1 propagation gaps for nLayers phase masks.

    /// Encode the discrete D2NN forward pass.
    func encodeDiscreteForwardPass(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        step: Int,
        effectiveNZ: Int? = nil
    ) {
        // Phase screen params: k0=1, dz=1 makes exp(i*k0*deltaN*dz) = exp(i*deltaN)
        // Pass I_sat from config for saturable absorption support
        var bpmParams = BPMParamsGPU(k0: 1.0, dz: 1.0, I_sat: config.saturationIntensity)
        let nLayers = effectiveNZ ?? config.nz

        // Initial propagation: input face → first layer
        encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: false)
        encoder.setComputePipelineState(elemMulPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(propagator, offset: 0, index: 1)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
        encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: true)

        // For each discrete layer: cache → phase mask → propagate
        for z in 0..<nLayers {
            // Cache field arriving at this layer (for backward gradient computation)
            var zIdx = UInt32(z)
            encoder.setComputePipelineState(cacheFieldPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(frame.fieldCache, offset: 0, index: 1)
            encoder.setBytes(&zIdx, length: 4, index: 2)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )

            // Phase mask: field *= exp(i * phase[z])
            // Using existing phase_screen kernel with k0=1, dz=1
            encoder.setComputePipelineState(phaseScreenPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(deltaN, offset: 0, index: 1)
            encoder.setBytes(&zIdx, length: 4, index: 2)
            encoder.setBytes(&bpmParams, length: MemoryLayout<BPMParamsGPU>.size, index: 3)
            encoder.setBuffer(deltaKappa, offset: 0, index: 4)
            encoder.dispatchThreads(fieldThreads, threadsPerThreadgroup: fieldThreadgroup)

            // Free-space propagation to next layer (or output)
            encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: false)
            encoder.setComputePipelineState(elemMulPipeline)
            encoder.setBuffer(frame.fields, offset: 0, index: 0)
            encoder.setBuffer(propagator, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
            encodeFFT2D(encoder: encoder, field: frame.fields, temp: frame.fftTemp, inverse: true)
        }

        // Cache output field (z = nLayers)
        var nLayersIdx = UInt32(nLayers)
        encoder.setComputePipelineState(cacheFieldPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.fieldCache, offset: 0, index: 1)
        encoder.setBytes(&nLayersIdx, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Encode the discrete D2NN backward (adjoint) pass.
    func encodeDiscreteBackwardPass(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        effectiveNZ: Int? = nil
    ) {
        var bpmParams = BPMParamsGPU(k0: 1.0, dz: 1.0, I_sat: config.saturationIntensity)
        let nLayers = effectiveNZ ?? config.nz

        for z in stride(from: nLayers - 1, through: 0, by: -1) {
            // Reverse propagation (conjugate propagator)
            encodeFFT2D(encoder: encoder, field: frame.adjoint, temp: frame.fftTemp, inverse: false)
            encoder.setComputePipelineState(elemMulConjPipeline)
            encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
            encoder.setBuffer(propagator, offset: 0, index: 1)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
            encodeFFT2D(encoder: encoder, field: frame.adjoint, temp: frame.fftTemp, inverse: true)

            // Gradient computation + reverse phase screen
            // With k0=1, dz=1: gradient = 2*(-product.y), no k0*dz scaling
            var zIdx = UInt32(z)
            encoder.setComputePipelineState(gradientReversePhasePipeline)
            encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
            encoder.setBuffer(frame.fieldCache, offset: 0, index: 1)
            encoder.setBuffer(deltaN, offset: 0, index: 2)
            encoder.setBuffer(frame.perBatchGrad, offset: 0, index: 3)
            encoder.setBytes(&zIdx, length: 4, index: 4)
            encoder.setBytes(&bpmParams, length: MemoryLayout<BPMParamsGPU>.size, index: 5)
            encoder.setBuffer(deltaKappa, offset: 0, index: 6)
            encoder.setBuffer(frame.perBatchGradKappa, offset: 0, index: 7)
            encoder.dispatchThreads(
                threads1DGrid(batchFieldCount),
                threadsPerThreadgroup: threads1DGroup(batchFieldCount)
            )
        }

        // Note: we skip the reverse of the initial propagation step since
        // there are no learnable parameters between input and first layer.
    }

    // MARK: - Gumbel-softmax + Adam optimizer encoding
    // All methods accept activeVoxels parameter to only process z < effectiveNZ.
    // Params are passed by value (setBytes) to avoid race conditions with triple buffering.

    // MARK: - Gumbel-softmax: refractive index (deltaN)

    func encodeGumbelForward(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        var p = params
        encoder.setComputePipelineState(gumbelForwardPipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(deltaN, offset: 0, index: 1)
        encoder.setBuffer(dnLevels, offset: 0, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
        encoder.dispatchThreads(
            threads1DGrid(nVox),
            threadsPerThreadgroup: threads1DGroup(nVox)
        )
    }

    func encodeGumbelBackward(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        var p = params
        encoder.setComputePipelineState(gumbelBackwardPipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(deltaN, offset: 0, index: 1)
        encoder.setBuffer(gradDeltaN, offset: 0, index: 2)
        encoder.setBuffer(gradLogits, offset: 0, index: 3)
        encoder.setBuffer(dnLevels, offset: 0, index: 4)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
        encoder.dispatchThreads(
            threads1DGrid(nVox),
            threadsPerThreadgroup: threads1DGroup(nVox)
        )
    }

    func encodeAdamUpdate(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        let totalLogits = nVox * config.nLevels
        var p = params
        encoder.setComputePipelineState(adamUpdatePipeline)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(gradLogits, offset: 0, index: 1)
        encoder.setBuffer(momentM, offset: 0, index: 2)
        encoder.setBuffer(momentV, offset: 0, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        encoder.dispatchThreads(
            threads1DGrid(totalLogits),
            threadsPerThreadgroup: threads1DGroup(totalLogits)
        )
    }

    // MARK: - Direct Adam update (bypasses Gumbel-softmax)

    func encodeDirectAdamUpdate(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        var p = params
        encoder.setComputePipelineState(adamDirectPipeline)
        encoder.setBuffer(deltaN, offset: 0, index: 0)
        encoder.setBuffer(gradDeltaN, offset: 0, index: 1)
        encoder.setBuffer(deltaNMomentM, offset: 0, index: 2)
        encoder.setBuffer(deltaNMomentV, offset: 0, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        encoder.dispatchThreads(
            threads1DGrid(nVox),
            threadsPerThreadgroup: threads1DGroup(nVox)
        )
    }

    // MARK: - EMA weight averaging (Polyak averaging)

    /// Update EMA shadow weights: ema = decay * ema + (1-decay) * current
    func encodeEMAUpdate(
        encoder: MTLComputeCommandEncoder,
        current: MTLBuffer,
        ema: MTLBuffer,
        decay: Float,
        count: Int
    ) {
        var d = decay
        var c = UInt32(count)
        encoder.setComputePipelineState(emaUpdatePipeline)
        encoder.setBuffer(current, offset: 0, index: 0)
        encoder.setBuffer(ema, offset: 0, index: 1)
        encoder.setBytes(&d, length: 4, index: 2)
        encoder.setBytes(&c, length: 4, index: 3)
        encoder.dispatchThreads(
            threads1DGrid(count),
            threadsPerThreadgroup: threads1DGroup(count)
        )
    }

    /// Swap EMA weights into active weights (copy ema → current)
    func swapEMAToActive() {
        let tv = config.totalVoxels
        let nl = config.nLevels
        if config.useDirect {
            memcpy(deltaN.contents(), emaDeltaN.contents(), tv * 4)
        } else {
            memcpy(logits.contents(), emaLogits.contents(), tv * nl * 4)
        }
        if config.kappaMax > 0 {
            memcpy(kappaLogits.contents(), emaKappaLogits.contents(), tv * nl * 4)
        }
    }

    /// Copy active weights into EMA (reset EMA to current position)
    func resetEMAFromActive() {
        let tv = config.totalVoxels
        let nl = config.nLevels
        if config.useDirect {
            memcpy(emaDeltaN.contents(), deltaN.contents(), tv * 4)
        } else {
            memcpy(emaLogits.contents(), logits.contents(), tv * nl * 4)
        }
        if config.kappaMax > 0 {
            memcpy(emaKappaLogits.contents(), kappaLogits.contents(), tv * nl * 4)
        }
    }

    /// Save current active weights as "best" checkpoint
    func saveBestWeights() {
        let tv = config.totalVoxels
        let nl = config.nLevels
        if config.useDirect {
            memcpy(bestDeltaN.contents(), deltaN.contents(), tv * 4)
        } else {
            memcpy(bestLogits.contents(), logits.contents(), tv * nl * 4)
        }
        if config.kappaMax > 0 {
            memcpy(bestKappaLogits.contents(), kappaLogits.contents(), tv * nl * 4)
        }
    }

    /// Restore "best" checkpoint into active weights
    func restoreBestWeights() {
        let tv = config.totalVoxels
        let nl = config.nLevels
        if config.useDirect {
            memcpy(deltaN.contents(), bestDeltaN.contents(), tv * 4)
        } else {
            memcpy(logits.contents(), bestLogits.contents(), tv * nl * 4)
        }
        if config.kappaMax > 0 {
            memcpy(kappaLogits.contents(), bestKappaLogits.contents(), tv * nl * 4)
        }
    }

    // MARK: - Gumbel-softmax: absorption (deltaKappa)
    // Reuses same pipelines with different buffer bindings

    func encodeGumbelForwardKappa(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        var p = params
        encoder.setComputePipelineState(gumbelForwardPipeline)
        encoder.setBuffer(kappaLogits, offset: 0, index: 0)
        encoder.setBuffer(deltaKappa, offset: 0, index: 1)
        encoder.setBuffer(kappaLevels, offset: 0, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 3)
        encoder.dispatchThreads(
            threads1DGrid(nVox),
            threadsPerThreadgroup: threads1DGroup(nVox)
        )
    }

    func encodeGumbelBackwardKappa(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        var p = params
        encoder.setComputePipelineState(gumbelBackwardPipeline)
        encoder.setBuffer(kappaLogits, offset: 0, index: 0)
        encoder.setBuffer(deltaKappa, offset: 0, index: 1)
        encoder.setBuffer(gradDeltaKappa, offset: 0, index: 2)
        encoder.setBuffer(gradKappaLogits, offset: 0, index: 3)
        encoder.setBuffer(kappaLevels, offset: 0, index: 4)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 5)
        encoder.dispatchThreads(
            threads1DGrid(nVox),
            threadsPerThreadgroup: threads1DGroup(nVox)
        )
    }

    func encodeAdamUpdateKappa(encoder: MTLComputeCommandEncoder, params: SimParamsGPU, activeVoxels: Int? = nil) {
        let nVox = activeVoxels ?? config.totalVoxels
        let totalLogits = nVox * config.nLevels
        var p = params
        encoder.setComputePipelineState(adamUpdatePipeline)
        encoder.setBuffer(kappaLogits, offset: 0, index: 0)
        encoder.setBuffer(gradKappaLogits, offset: 0, index: 1)
        encoder.setBuffer(kappaMomentM, offset: 0, index: 2)
        encoder.setBuffer(kappaMomentV, offset: 0, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        encoder.dispatchThreads(
            threads1DGrid(totalLogits),
            threadsPerThreadgroup: threads1DGroup(totalLogits)
        )
    }

    // MARK: - I/O encoding

    func encodeInput(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame
    ) {
        var ioParams = IOParamsGPU(
            n_inputs: UInt32(config.nInputsA),  // Only active input channels
            spot_sigma: spotSigma,
            use_phase_encoding: config.usePhaseEncoding ? 1 : 0
        )
        encoder.setComputePipelineState(encodeInputPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.inputData, offset: 0, index: 1)
        encoder.setBuffer(spotCenters, offset: 0, index: 2)
        encoder.setBytes(&ioParams, length: MemoryLayout<IOParamsGPU>.size, index: 3)
        encoder.dispatchThreads(fieldThreads, threadsPerThreadgroup: fieldThreadgroup)
    }

    /// Encode image input: stamp full NX×NY image onto the optical field.
    /// For UCLA-style D2NN where the MNIST image IS the input field.
    func encodeImageInput(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame
    ) {
        var usePhase: UInt32 = config.usePhaseEncoding ? 1 : 0
        encoder.setComputePipelineState(encodeImageInputPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.inputData, offset: 0, index: 1)
        encoder.setBytes(&usePhase, length: 4, index: 2)
        encoder.dispatchThreads(fieldThreads, threadsPerThreadgroup: fieldThreadgroup)
    }

    func encodeLossAndGradient(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        normScale: Float? = nil
    ) {
        // CRITICAL: Zero the adjoint buffer before loss kernel.
        // The loss kernel uses += to accumulate adjoint from multiple detectors,
        // so non-detector pixels must be zero (they don't contribute to loss).
        // Without this, stale data in non-detector pixels corrupts the backward pass
        // when FFTs mix all pixel values together.
        let adjointCount = config.batchSize * config.fieldSize
        var zeroVal: Float = 0.0
        var count = UInt32(adjointCount * 2)  // *2 for float2 (real + imag)
        encoder.setComputePipelineState(fillBufferPipeline)
        encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
        encoder.setBytes(&zeroVal, length: 4, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, Int(count)), height: 1, depth: 1)
        )

        var detParams = DetectorParamsGPU(
            n_outputs: UInt32(config.nOutputsA),
            n_detectors: UInt32(config.nDetectors),
            use_differential: config.useDifferentialDetection ? 1 : 0,
            norm_scale: normScale ?? config.normScale
        )
        encoder.setComputePipelineState(lossAndGradientPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.adjoint, offset: 0, index: 1)
        encoder.setBuffer(frame.losses, offset: 0, index: 2)
        encoder.setBuffer(frame.accuracyCount, offset: 0, index: 3)
        encoder.setBuffer(frame.targetClasses, offset: 0, index: 4)
        encoder.setBuffer(detectorMaskData, offset: 0, index: 5)
        encoder.setBuffer(detectorOffsets, offset: 0, index: 6)
        encoder.setBytes(&detParams, length: MemoryLayout<DetectorParamsGPU>.size, index: 7)
        // One thread per batch sample
        encoder.dispatchThreads(
            MTLSize(width: config.batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(config.batchSize, 32), height: 1, depth: 1)
        )
    }

    /// MSE loss for regression mode. Computes |E_out - E_target|² and seeds adjoint.
    func encodeMSELossAndGradient(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame
    ) {
        // No need to zero adjoint — the MSE kernel writes every pixel (not +=)
        encoder.setComputePipelineState(mseLossPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.adjoint, offset: 0, index: 1)
        encoder.setBuffer(frame.losses, offset: 0, index: 2)
        encoder.setBuffer(frame.accuracyCount, offset: 0, index: 3)
        encoder.setBuffer(frame.targetField, offset: 0, index: 4)
        // One thread per batch sample
        encoder.dispatchThreads(
            MTLSize(width: config.batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(config.batchSize, 32), height: 1, depth: 1)
        )
    }

    // MARK: - Recurrent pulse encoding

    /// Cache the input field encoding (before first pulse) for feedback mixing
    func encodeCacheInput(encoder: MTLComputeCommandEncoder, frame: TrainingFrame) {
        encoder.setComputePipelineState(copyBufferPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.inputFieldCache, offset: 0, index: 1)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Soft threshold on output field. Caches pre-threshold field for backward.
    func encodeSoftThreshold(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        pulseIndex: Int
    ) {
        var thresh = thresholdBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
        var steep: Float = 20.0  // sigmoid steepness
        let cacheOffset = pulseIndex * batchFieldCount * 8  // bytes

        encoder.setComputePipelineState(softThresholdPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.thresholdCache, offset: cacheOffset, index: 1)
        encoder.setBytes(&thresh, length: 4, index: 2)
        encoder.setBytes(&steep, length: 4, index: 3)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Feedback mixing: field = ratio * thresholded_output + (1 - ratio) * input
    /// Caches thresholded output for backward.
    func encodeFeedbackMix(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        pulseIndex: Int
    ) {
        var ratio = feedbackRatio.contents().bindMemory(to: Float.self, capacity: 1).pointee
        let cacheOffset = pulseIndex * batchFieldCount * 8

        // Save post-threshold field (before mixing) for backward
        encoder.setComputePipelineState(copyBufferPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.feedbackCache, offset: cacheOffset, index: 1)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )

        // Mix: field = ratio * field + (1 - ratio) * inputFieldCache
        encoder.setComputePipelineState(feedbackMixPipeline)
        encoder.setBuffer(frame.fields, offset: 0, index: 0)
        encoder.setBuffer(frame.inputFieldCache, offset: 0, index: 1)
        encoder.setBytes(&ratio, length: 4, index: 2)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Backward through feedback mixing
    func encodeFeedbackMixBackward(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        pulseIndex: Int
    ) {
        var ratio = feedbackRatio.contents().bindMemory(to: Float.self, capacity: 1).pointee
        let cacheOffset = pulseIndex * batchFieldCount * 8

        encoder.setComputePipelineState(feedbackMixBackwardPipeline)
        encoder.setBuffer(frame.adjoint, offset: 0, index: 0)         // adj_mixed → becomes adj_feedback
        encoder.setBuffer(frame.fftTemp, offset: 0, index: 1)         // reuse fftTemp as adj_input scratch
        encoder.setBuffer(frame.feedbackCache, offset: cacheOffset, index: 2)
        encoder.setBuffer(frame.inputFieldCache, offset: 0, index: 3)
        encoder.setBytes(&ratio, length: 4, index: 4)
        encoder.setBuffer(gradFeedbackRatio, offset: 0, index: 5)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Backward through soft threshold
    func encodeSoftThresholdBackward(
        encoder: MTLComputeCommandEncoder,
        frame: TrainingFrame,
        pulseIndex: Int
    ) {
        var thresh = thresholdBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee
        var steep: Float = 20.0
        let cacheOffset = pulseIndex * batchFieldCount * 8

        encoder.setComputePipelineState(softThresholdBackwardPipeline)
        encoder.setBuffer(frame.adjoint, offset: 0, index: 0)
        encoder.setBuffer(frame.thresholdCache, offset: cacheOffset, index: 1)
        encoder.setBytes(&thresh, length: 4, index: 2)
        encoder.setBytes(&steep, length: 4, index: 3)
        encoder.setBuffer(gradThreshold, offset: 0, index: 4)
        encoder.dispatchThreads(
            threads1DGrid(batchFieldCount),
            threadsPerThreadgroup: threads1DGroup(batchFieldCount)
        )
    }

    /// Adam update for scalar learnable parameters (threshold, feedback ratio)
    func encodeScalarAdamUpdate(
        encoder: MTLComputeCommandEncoder,
        param: MTLBuffer,
        grad: MTLBuffer,
        momentM: MTLBuffer,
        momentV: MTLBuffer,
        params: SimParamsGPU
    ) {
        var p = params
        encoder.setComputePipelineState(adamUpdatePipeline)
        encoder.setBuffer(param, offset: 0, index: 0)
        encoder.setBuffer(grad, offset: 0, index: 1)
        encoder.setBuffer(momentM, offset: 0, index: 2)
        encoder.setBuffer(momentV, offset: 0, index: 3)
        encoder.setBytes(&p, length: MemoryLayout<SimParamsGPU>.stride, index: 4)
        // Single element
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
    }
}

// MARK: - GPU struct mirrors

struct BPMParamsGPU {
    var k0: Float
    var dz: Float
    var I_sat: Float
}

struct IOParamsGPU {
    var n_inputs: UInt32
    var spot_sigma: Float
    var use_phase_encoding: UInt32
}

struct DetectorParamsGPU {
    var n_outputs: UInt32
    var n_detectors: UInt32
    var use_differential: UInt32
    var norm_scale: Float
}
