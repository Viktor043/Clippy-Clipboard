import Metal
import Foundation

/// Central Metal infrastructure: device, command queue, compiled pipelines, and all GPU buffers.
final class MetalEngine {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let config: SimConfig

    // MARK: - Compute pipelines (16 kernels)
    // FFT
    var fftRowsPipeline: MTLComputePipelineState!
    var fftColsPipeline: MTLComputePipelineState!
    var fftNormalizePipeline: MTLComputePipelineState!
    var copyBufferPipeline: MTLComputePipelineState!
    // BPM
    var phaseScreenPipeline: MTLComputePipelineState!
    var elemMulPipeline: MTLComputePipelineState!
    var elemMulConjPipeline: MTLComputePipelineState!
    var cacheFieldPipeline: MTLComputePipelineState!
    var gradientReversePhasePipeline: MTLComputePipelineState!
    // Training
    var gumbelForwardPipeline: MTLComputePipelineState!
    var gumbelBackwardPipeline: MTLComputePipelineState!
    var adamUpdatePipeline: MTLComputePipelineState!
    var reduceBatchGradsPipeline: MTLComputePipelineState!
    var scaleBufferPipeline: MTLComputePipelineState!
    var fillBufferPipeline: MTLComputePipelineState!
    var adamDirectPipeline: MTLComputePipelineState!
    var emaUpdatePipeline: MTLComputePipelineState!
    // I/O
    var encodeInputPipeline: MTLComputePipelineState!
    var encodeImageInputPipeline: MTLComputePipelineState!
    var lossAndGradientPipeline: MTLComputePipelineState!
    var mseLossPipeline: MTLComputePipelineState!
    // Recurrent pulse
    var softThresholdPipeline: MTLComputePipelineState!
    var softThresholdBackwardPipeline: MTLComputePipelineState!
    var feedbackMixPipeline: MTLComputePipelineState!
    var feedbackMixBackwardPipeline: MTLComputePipelineState!

    // MARK: - Persistent buffers (survive across steps)
    var logits: MTLBuffer!           // [totalVoxels * nLevels] float — refractive index
    var momentM: MTLBuffer!          // [totalVoxels * nLevels] float
    var momentV: MTLBuffer!          // [totalVoxels * nLevels] float
    var kappaLogits: MTLBuffer!      // [totalVoxels * nLevels] float — absorption
    var kappaMomentM: MTLBuffer!     // [totalVoxels * nLevels] float
    var kappaMomentV: MTLBuffer!     // [totalVoxels * nLevels] float
    // Direct mode: Adam moments for deltaN (totalVoxels, not totalVoxels*nLevels)
    var deltaNMomentM: MTLBuffer!    // [totalVoxels] float
    var deltaNMomentV: MTLBuffer!    // [totalVoxels] float
    // EMA shadow weights (Polyak averaging)
    var emaLogits: MTLBuffer!        // [totalVoxels * nLevels] float — EMA of logits
    var emaKappaLogits: MTLBuffer!   // [totalVoxels * nLevels] float — EMA of kappa logits
    var emaDeltaN: MTLBuffer!        // [totalVoxels] float — EMA deltaN for direct mode
    // Best-accuracy checkpoint (saved on peak accuracy hits)
    var bestLogits: MTLBuffer!       // [totalVoxels * nLevels] float — logits at best accuracy
    var bestKappaLogits: MTLBuffer!  // [totalVoxels * nLevels] float
    var bestDeltaN: MTLBuffer!       // [totalVoxels] float — for direct mode

    // MARK: - Shared buffers (recomputed each step)
    var deltaN: MTLBuffer!           // [totalVoxels] float
    var gradDeltaN: MTLBuffer!       // [totalVoxels] float
    var gradLogits: MTLBuffer!       // [totalVoxels * nLevels] float
    var deltaKappa: MTLBuffer!       // [totalVoxels] float — absorption values
    var gradDeltaKappa: MTLBuffer!   // [totalVoxels] float
    var gradKappaLogits: MTLBuffer!  // [totalVoxels * nLevels] float
    var propagator: MTLBuffer!       // [fieldSize] float2 (complex)
    var dnLevels: MTLBuffer!         // [nLevels] float
    var kappaLevels: MTLBuffer!      // [nLevels] float — [0, kappaMax]
    var paramsBuffer: MTLBuffer!     // SimParams struct

    // MARK: - Recurrent pulse buffers (learnable scalars)
    var thresholdBuffer: MTLBuffer!    // [1] float — learnable threshold
    var feedbackRatio: MTLBuffer!      // [1] float — learnable feedback ratio
    var gradThreshold: MTLBuffer!      // [1] float — gradient accumulator
    var gradFeedbackRatio: MTLBuffer!  // [1] float — gradient accumulator
    var thresholdMomentM: MTLBuffer!   // [1] float — Adam moment
    var thresholdMomentV: MTLBuffer!   // [1] float — Adam moment
    var ratioMomentM: MTLBuffer!       // [1] float — Adam moment
    var ratioMomentV: MTLBuffer!       // [1] float — Adam moment

    // MARK: - Triple-buffered transient frames
    struct TrainingFrame {
        let fields: MTLBuffer        // [B * fieldSize] float2
        let adjoint: MTLBuffer       // [B * fieldSize] float2
        let fftTemp: MTLBuffer       // [B * fieldSize] float2
        let fieldCache: MTLBuffer    // [B * nPulses * (NZ+1) * fieldSize] float2
        let perBatchGrad: MTLBuffer  // [B * totalVoxels] float — dn gradient
        let perBatchGradKappa: MTLBuffer // [B * totalVoxels] float — kappa gradient
        let inputData: MTLBuffer     // [B * nInputs] float
        let targetClasses: MTLBuffer // [B] uint32
        let losses: MTLBuffer        // [B] float
        let accuracyCount: MTLBuffer // [1] uint32
        // Recurrent pulse buffers
        let inputFieldCache: MTLBuffer   // [B * fieldSize] float2 — cached input encoding
        let thresholdCache: MTLBuffer    // [B * nPulses * fieldSize] float2 — pre-threshold fields
        let feedbackCache: MTLBuffer     // [B * nPulses * fieldSize] float2 — post-threshold fields
        // Regression mode target (e.g., exact FFT output)
        let targetField: MTLBuffer       // [B * fieldSize] float2 — target complex field
    }
    var frames: [TrainingFrame] = []

    // MARK: - I/O geometry (CPU-computed, uploaded once)
    var spotCenters: MTLBuffer!      // [(nInputs) * 2] float (cx, cy pairs)
    var spotSigma: Float = 0
    var detectorMaskData: MTLBuffer! // flattened pixel indices per detector
    var detectorOffsets: MTLBuffer!  // [nOutputsA + 1] uint (prefix sums into mask data)
    var detectorCount: MTLBuffer!    // [nOutputsA] uint (pixels per detector)

    init(config: SimConfig) throws {
        self.config = config

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw EngineError.noDevice
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw EngineError.noCommandQueue
        }
        self.commandQueue = queue

        // Compile shader library with function constants
        let constants = MTLFunctionConstantValues()
        var nx = UInt32(config.nx)
        var ny = UInt32(config.ny)
        var nz = UInt32(config.nz)
        var batch = UInt32(config.batchSize)
        var nlevels = UInt32(config.nLevels)
        constants.setConstantValue(&nx, type: .uint, index: 0)
        constants.setConstantValue(&ny, type: .uint, index: 1)
        constants.setConstantValue(&nz, type: .uint, index: 2)
        constants.setConstantValue(&batch, type: .uint, index: 3)
        constants.setConstantValue(&nlevels, type: .uint, index: 4)

        // Load shader source from SPM resource bundle and compile at runtime
        let shaderFiles = ["Common.metal", "FFT.metal", "BPM.metal", "Training.metal", "IO.metal"]
        var source = ""
        for file in shaderFiles {
            let name = file.replacingOccurrences(of: ".metal", with: "")
            if let url = Bundle.module.url(forResource: name, withExtension: "metal") {
                if let content = try? String(contentsOf: url) {
                    source += content + "\n"
                }
            }
        }
        if source.isEmpty {
            throw EngineError.noShaders
        }
        let compileOpts = MTLCompileOptions()
        self.library = try device.makeLibrary(source: source, options: compileOpts)
        try compilePipelines(constants: constants)
        try allocateBuffers()
        precomputePropagator()
        precomputeDnLevels()
        precomputeKappaLevels()
    }

    // MARK: - Pipeline compilation

    private func compilePipelines(constants: MTLFunctionConstantValues) throws {
        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            let fn = try library.makeFunction(name: name, constantValues: constants)
            return try device.makeComputePipelineState(function: fn)
        }

        fftRowsPipeline = try makePipeline("fft_stockham_rows_batched")
        fftColsPipeline = try makePipeline("fft_stockham_cols_batched")
        fftNormalizePipeline = try makePipeline("fft_normalize_batched")
        copyBufferPipeline = try makePipeline("copy_complex_buffer")
        phaseScreenPipeline = try makePipeline("phase_screen_forward_batched")
        elemMulPipeline = try makePipeline("elementwise_multiply_batched")
        elemMulConjPipeline = try makePipeline("elementwise_multiply_conj_batched")
        cacheFieldPipeline = try makePipeline("cache_field_batched")
        gradientReversePhasePipeline = try makePipeline("gradient_reverse_phase_batched")
        gumbelForwardPipeline = try makePipeline("gumbel_softmax_forward")
        gumbelBackwardPipeline = try makePipeline("gumbel_softmax_backward")
        adamUpdatePipeline = try makePipeline("adam_update")
        reduceBatchGradsPipeline = try makePipeline("reduce_batch_gradients")
        scaleBufferPipeline = try makePipeline("scale_buffer")
        fillBufferPipeline = try makePipeline("fill_buffer")
        adamDirectPipeline = try makePipeline("adam_update_direct")
        emaUpdatePipeline = try makePipeline("ema_update")
        encodeInputPipeline = try makePipeline("encode_input_batched")
        encodeImageInputPipeline = try makePipeline("encode_image_input_batched")
        lossAndGradientPipeline = try makePipeline("compute_loss_and_gradient_batched")
        mseLossPipeline = try makePipeline("compute_mse_loss_batched")
        softThresholdPipeline = try makePipeline("soft_threshold_batched")
        softThresholdBackwardPipeline = try makePipeline("soft_threshold_backward_batched")
        feedbackMixPipeline = try makePipeline("feedback_mix_batched")
        feedbackMixBackwardPipeline = try makePipeline("feedback_mix_backward_batched")
    }

    // MARK: - Buffer allocation

    private func allocateBuffers() throws {
        let B = config.batchSize
        let fs = config.fieldSize
        let tv = config.totalVoxels
        let nl = config.nLevels

        func alloc(_ count: Int, _ stride: Int, label: String) throws -> MTLBuffer {
            let size = count * stride
            guard let buf = device.makeBuffer(length: max(size, 4), options: .storageModeShared) else {
                throw EngineError.bufferAllocation(label, size)
            }
            buf.label = label
            return buf
        }

        // Persistent — refractive index
        logits = try alloc(tv * nl, 4, label: "logits")
        momentM = try alloc(tv * nl, 4, label: "momentM")
        momentV = try alloc(tv * nl, 4, label: "momentV")
        // Persistent — absorption
        kappaLogits = try alloc(tv * nl, 4, label: "kappaLogits")
        kappaMomentM = try alloc(tv * nl, 4, label: "kappaMomentM")
        kappaMomentV = try alloc(tv * nl, 4, label: "kappaMomentV")

        // Direct mode Adam moments (totalVoxels, not totalVoxels*nLevels)
        deltaNMomentM = try alloc(tv, 4, label: "deltaNMomentM")
        deltaNMomentV = try alloc(tv, 4, label: "deltaNMomentV")

        // EMA shadow weights (Polyak averaging)
        emaLogits = try alloc(tv * nl, 4, label: "emaLogits")
        emaKappaLogits = try alloc(tv * nl, 4, label: "emaKappaLogits")
        emaDeltaN = try alloc(tv, 4, label: "emaDeltaN")
        // Best-accuracy checkpoint
        bestLogits = try alloc(tv * nl, 4, label: "bestLogits")
        bestKappaLogits = try alloc(tv * nl, 4, label: "bestKappaLogits")
        bestDeltaN = try alloc(tv, 4, label: "bestDeltaN")

        // Shared — refractive index
        deltaN = try alloc(tv, 4, label: "deltaN")
        gradDeltaN = try alloc(tv, 4, label: "gradDeltaN")
        gradLogits = try alloc(tv * nl, 4, label: "gradLogits")
        // Shared — absorption
        deltaKappa = try alloc(tv, 4, label: "deltaKappa")
        gradDeltaKappa = try alloc(tv, 4, label: "gradDeltaKappa")
        gradKappaLogits = try alloc(tv * nl, 4, label: "gradKappaLogits")
        // Shared — common
        propagator = try alloc(fs, 8, label: "propagator")       // complex = 2 floats
        dnLevels = try alloc(nl, 4, label: "dnLevels")
        kappaLevels = try alloc(nl, 4, label: "kappaLevels")
        paramsBuffer = try alloc(1, MemoryLayout<SimParamsGPU>.stride, label: "params")

        // Recurrent pulse learnable parameters
        let nP = config.nPulses
        thresholdBuffer = try alloc(1, 4, label: "threshold")
        feedbackRatio = try alloc(1, 4, label: "feedbackRatio")
        gradThreshold = try alloc(1, 4, label: "gradThreshold")
        gradFeedbackRatio = try alloc(1, 4, label: "gradFeedbackRatio")
        thresholdMomentM = try alloc(1, 4, label: "thresholdMomentM")
        thresholdMomentV = try alloc(1, 4, label: "thresholdMomentV")
        ratioMomentM = try alloc(1, 4, label: "ratioMomentM")
        ratioMomentV = try alloc(1, 4, label: "ratioMomentV")

        // Triple-buffered frames
        for i in 0..<3 {
            let frame = TrainingFrame(
                fields: try alloc(B * fs, 8, label: "fields_\(i)"),
                adjoint: try alloc(B * fs, 8, label: "adjoint_\(i)"),
                fftTemp: try alloc(B * fs, 8, label: "fftTemp_\(i)"),
                fieldCache: try alloc(B * nP * (config.nz + 1) * fs, 8, label: "fieldCache_\(i)"),
                perBatchGrad: try alloc(B * tv, 4, label: "perBatchGrad_\(i)"),
                perBatchGradKappa: try alloc(B * tv, 4, label: "perBatchGradKappa_\(i)"),
                inputData: try alloc(config.useImageInput ? B * fs : B * config.nInputsA, 4, label: "inputData_\(i)"),
                targetClasses: try alloc(B, 4, label: "targets_\(i)"),
                losses: try alloc(B, 4, label: "losses_\(i)"),
                accuracyCount: try alloc(1, 4, label: "accuracy_\(i)"),
                inputFieldCache: try alloc(B * fs, 8, label: "inputFieldCache_\(i)"),
                thresholdCache: try alloc(B * max(nP, 1) * fs, 8, label: "thresholdCache_\(i)"),
                feedbackCache: try alloc(B * max(nP, 1) * fs, 8, label: "feedbackCache_\(i)"),
                targetField: try alloc(B * fs, 8, label: "targetField_\(i)")
            )
            frames.append(frame)
        }

        // Initialize logits with small random values
        initializeLogits()
        initializeKappaLogits()

        // Initialize EMA buffers as copy of initial logits
        memcpy(emaLogits.contents(), logits.contents(), tv * nl * 4)
        memcpy(emaKappaLogits.contents(), kappaLogits.contents(), tv * nl * 4)
        memset(emaDeltaN.contents(), 0, tv * 4)
        // Initialize best buffers
        memcpy(bestLogits.contents(), logits.contents(), tv * nl * 4)
        memcpy(bestKappaLogits.contents(), kappaLogits.contents(), tv * nl * 4)
        memset(bestDeltaN.contents(), 0, tv * 4)

        // Zero moments
        memset(momentM.contents(), 0, tv * nl * 4)
        memset(momentV.contents(), 0, tv * nl * 4)
        memset(kappaMomentM.contents(), 0, tv * nl * 4)
        memset(kappaMomentV.contents(), 0, tv * nl * 4)
        memset(deltaNMomentM.contents(), 0, tv * 4)
        memset(deltaNMomentV.contents(), 0, tv * 4)

        // Initialize recurrent pulse parameters
        thresholdBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee = config.thresholdInit
        feedbackRatio.contents().bindMemory(to: Float.self, capacity: 1).pointee = config.feedbackRatioInit
        memset(gradThreshold.contents(), 0, 4)
        memset(gradFeedbackRatio.contents(), 0, 4)
        memset(thresholdMomentM.contents(), 0, 4)
        memset(thresholdMomentV.contents(), 0, 4)
        memset(ratioMomentM.contents(), 0, 4)
        memset(ratioMomentV.contents(), 0, 4)

        // Zero shared gradient buffers (critical: uninitialized memory may contain NaN,
        // and NaN * 0 = NaN, so the scale_buffer zeroing approach fails without this)
        memset(gradDeltaN.contents(), 0, tv * 4)
        memset(gradDeltaKappa.contents(), 0, tv * 4)
        memset(gradLogits.contents(), 0, tv * nl * 4)
        memset(gradKappaLogits.contents(), 0, tv * nl * 4)
        memset(deltaN.contents(), 0, tv * 4)
        memset(deltaKappa.contents(), 0, tv * 4)

        // Re-initialize deltaN AFTER zeroing (the memset above would overwrite earlier init)
        if config.useDirect {
            if config.useDiscreteD2NN {
                initializeDiscretePhases()
            } else {
                initializeDeltaNDirect()
            }
        }
    }

    private func initializeLogits() {
        let ptr = logits.contents().bindMemory(to: Float.self, capacity: config.totalVoxels * config.nLevels)
        let zeroLevel = config.nLevels / 2  // Level closest to dn=0
        var rng = config.seed

        for v in 0..<config.totalVoxels {
            for j in 0..<config.nLevels {
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                let noise = Float(rng & 0xFFFFFF) / Float(0xFFFFFF) * 0.02 - 0.01
                let idx = v * config.nLevels + j
                if j == zeroLevel || j == zeroLevel - 1 {
                    // Mild bias toward zero-dn: at temp=0.5, bias=1.5 → ~57% mass
                    // on near-zero levels. Low enough for Adam to overcome within
                    // ~150 steps at lr=0.01 (150*0.01=1.5 logit shift).
                    ptr[idx] = 1.5 + noise
                } else {
                    ptr[idx] = noise
                }
            }
        }
    }

    /// Initialize kappa logits biased toward zero absorption (level 0)
    private func initializeKappaLogits() {
        let ptr = kappaLogits.contents().bindMemory(to: Float.self, capacity: config.totalVoxels * config.nLevels)
        var rng = config.seed &+ 12345  // Different seed stream from dn logits

        for v in 0..<config.totalVoxels {
            for j in 0..<config.nLevels {
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                let noise = Float(rng & 0xFFFFFF) / Float(0xFFFFFF) * 0.02 - 0.01
                let idx = v * config.nLevels + j
                if j == 0 {
                    // Bias toward zero absorption: starts mostly transparent
                    // At temp=0.5, bias=2.0 → ~88% mass on zero kappa
                    ptr[idx] = 2.0 + noise
                } else {
                    ptr[idx] = noise
                }
            }
        }
    }

    /// Initialize deltaN directly with small random values for direct optimization mode
    private func initializeDeltaNDirect() {
        let ptr = deltaN.contents().bindMemory(to: Float.self, capacity: config.totalVoxels)
        let dnMax = Float(config.dnMax)
        var rng = config.seed &+ 99999
        for i in 0..<config.totalVoxels {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            // Small random values in [-0.1*dnMax, +0.1*dnMax]
            let noise = Float(rng & 0xFFFFFF) / Float(0xFFFFFF) * 0.2 - 0.1
            ptr[i] = noise * dnMax
        }
    }

    /// Initialize phases for discrete D2NN mode
    private func initializeDiscretePhases() {
        let ptr = deltaN.contents().bindMemory(to: Float.self, capacity: config.totalVoxels)
        if config.useZeroInit {
            // Zero init: neutral glass, light passes straight through.
            // Optimizer gradually introduces structure from a clean starting point.
            for i in 0..<config.totalVoxels { ptr[i] = 0.0 }
            print("[DiscreteD2NN] Zero-initialized \(config.totalVoxels) phases (neutral glass)")
        } else {
            var rng = config.seed &+ 77777
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            for i in 0..<config.totalVoxels {
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                let noise = Float(rng & 0xFFFFFF) / Float(0xFFFFFF) * 0.2 - 0.1
                ptr[i] = noise
                minVal = min(minVal, noise)
                maxVal = max(maxVal, noise)
            }
            print("[DiscreteD2NN] Random-initialized \(config.totalVoxels) phases in [\(minVal), \(maxVal)] rad")
        }
        print("[DiscreteD2NN] N_F=\(Int(Double(config.nx) * Double(config.nx) * config.voxelPitch * config.voxelPitch / (config.wavelength * config.dz))), layerSpacing=\(config.dz*1e6)um, nLayers=\(config.nDiscreteLayers)")
    }

    // MARK: - Propagator precomputation

    /// Recompute physics buffers (propagator + dn_levels) from config.
    /// Call when physics params change live (dnMax, dz, wavelength, etc.)
    func recomputePhysics(config: SimConfig) {
        let k = 2.0 * Double.pi * config.nBase / config.wavelength
        let dx = config.voxelPitch
        let dy = config.voxelPitch
        let dz = config.dz
        let nx = config.nx, ny = config.ny

        // Propagator
        let propPtr = propagator.contents().bindMemory(to: SIMD2<Float>.self, capacity: nx * ny)
        for ix in 0..<nx {
            for iy in 0..<ny {
                let fx = ix < nx / 2 ? Double(ix) : Double(ix - nx)
                let fy = iy < ny / 2 ? Double(iy) : Double(iy - ny)
                let kx = 2.0 * .pi * fx / (Double(nx) * dx)
                let ky = 2.0 * .pi * fy / (Double(ny) * dy)
                let kzSq = k * k - kx * kx - ky * ky
                if kzSq > 0 {
                    let kz = kzSq.squareRoot()
                    let phase = (kz - k) * dz
                    propPtr[ix * ny + iy] = SIMD2<Float>(Float(cos(phase)), Float(sin(phase)))
                } else {
                    propPtr[ix * ny + iy] = SIMD2<Float>(0, 0)
                }
            }
        }

        // dn_levels
        let dnPtr = dnLevels.contents().bindMemory(to: Float.self, capacity: config.nLevels)
        for i in 0..<config.nLevels {
            dnPtr[i] = Float(config.dnMax) * (2.0 * Float(i) / Float(config.nLevels - 1) - 1.0)
        }

        // kappa_levels
        let kapPtr = kappaLevels.contents().bindMemory(to: Float.self, capacity: config.nLevels)
        for i in 0..<config.nLevels {
            kapPtr[i] = Float(config.kappaMax) * Float(i) / Float(config.nLevels - 1)
        }
    }

    private func precomputePropagator() {
        let nx = config.nx
        let ny = config.ny
        let k = 2.0 * Double.pi * config.nBase / config.wavelength
        let dx = config.voxelPitch
        let dy = config.voxelPitch
        let dz = config.dz

        let ptr = propagator.contents().bindMemory(to: SIMD2<Float>.self, capacity: nx * ny)

        for ix in 0..<nx {
            for iy in 0..<ny {
                // FFT frequency indices
                let fx = ix < nx / 2 ? Double(ix) : Double(ix - nx)
                let fy = iy < ny / 2 ? Double(iy) : Double(iy - ny)

                let kx = 2.0 * .pi * fx / (Double(nx) * dx)
                let ky = 2.0 * .pi * fy / (Double(ny) * dy)
                let kzSq = k * k - kx * kx - ky * ky

                if kzSq > 0 {
                    let kz = kzSq.squareRoot()
                    let phase = (kz - k) * dz  // Paraxial reference frame: subtract on-axis k
                    ptr[ix * ny + iy] = SIMD2<Float>(Float(cos(phase)), Float(sin(phase)))
                } else {
                    ptr[ix * ny + iy] = SIMD2<Float>(0, 0)  // evanescent → zero
                }
            }
        }
    }

    private func precomputeDnLevels() {
        let ptr = dnLevels.contents().bindMemory(to: Float.self, capacity: config.nLevels)
        for i in 0..<config.nLevels {
            // Span [-dnMax, +dnMax] so the medium can both increase and decrease index
            ptr[i] = Float(config.dnMax) * (2.0 * Float(i) / Float(config.nLevels - 1) - 1.0)
        }
    }

    private func precomputeKappaLevels() {
        let ptr = kappaLevels.contents().bindMemory(to: Float.self, capacity: config.nLevels)
        for i in 0..<config.nLevels {
            // Span [0, kappaMax] — absorption is always non-negative
            ptr[i] = Float(config.kappaMax) * Float(i) / Float(config.nLevels - 1)
        }
    }

    // MARK: - Create params struct for a given step (value type, no shared buffer race)

    /// Create GPU params from a config snapshot. Pass `liveConfig` to use live-tuned values.
    func makeParams(step: Int, lr: Float? = nil, from overrideConfig: SimConfig? = nil) -> SimParamsGPU {
        let c = overrideConfig ?? config
        let currentLR = lr ?? c.learningRate(at: step)
        return SimParamsGPU(
            k0: Float(c.k0),
            dz: Float(c.dz),
            dn_max: Float(c.dnMax),
            n_base: Float(c.nBase),
            temperature: c.temperature(at: step),
            lr: currentLR,
            beta1: c.adamBeta1,
            beta2: c.adamBeta2,
            epsilon: c.adamEps,
            sparsity_lambda: c.sparsityLambda,
            step: UInt32(step),
            seed: UInt32(c.seed & 0xFFFFFFFF),
            inv_batch: 1.0 / Float(c.batchSize),
            grad_clip_max: c.gradClipMax
        )
    }
}

// MARK: - GPU-side params struct (must match Metal SimParams layout)
struct SimParamsGPU {
    var k0: Float
    var dz: Float
    var dn_max: Float
    var n_base: Float
    var temperature: Float
    var lr: Float
    var beta1: Float
    var beta2: Float
    var epsilon: Float
    var sparsity_lambda: Float
    var step: UInt32
    var seed: UInt32
    var inv_batch: Float
    var grad_clip_max: Float
}

// MARK: - Errors
enum EngineError: Error, CustomStringConvertible {
    case noDevice
    case noCommandQueue
    case noShaders
    case bufferAllocation(String, Int)
    case pipelineCompilation(String)

    var description: String {
        switch self {
        case .noDevice: return "No Metal device found"
        case .noCommandQueue: return "Failed to create command queue"
        case .noShaders: return "No Metal shader files found"
        case .bufferAllocation(let name, let size):
            return "Failed to allocate buffer '\(name)' (\(size) bytes)"
        case .pipelineCompilation(let name):
            return "Failed to compile pipeline '\(name)'"
        }
    }
}
