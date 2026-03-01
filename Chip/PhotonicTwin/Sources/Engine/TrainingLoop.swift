import Foundation
import Metal

/// Orchestrates the triple-buffered training loop.
/// Supports single-pass and recurrent pulse modes.
final class TrainingLoop {
    let engine: MetalEngine
    let liveConfig: LiveConfig
    let monitor: TrainingMonitor
    let refModel: ReferenceModel
    let ioEncoder: IOEncoder
    let checkpoint: CheckpointManager

    // MNIST data (when useImageInput is true)
    private var mnistImages: [[Float]]?
    private var mnistLabels: [UInt8]?
    private var mnistRng: UInt64 = 42

    // Regression data generators (FFT or matrix transform)
    private var fftDataGen: FFTDataGenerator?
    private var matrixDataGen: MatrixTransformDataGenerator?

    /// Snapshot of initial config for fixed params (grid size, batch size, etc.)
    let baseConfig: SimConfig

    private var startStep: Int = 0
    private var startTime: CFAbsoluteTime = 0
    private let semaphore = DispatchSemaphore(value: 3)
    private var shouldStop = false
    private var lastLoss: Float = 0
    private var lastAccuracy: Float = 0
    private var lastStepsPerSec: Double = 0

    private var lastCompletedFrameIdx: Int = -1

    private var speedWindowStart: CFAbsoluteTime = 0
    private var speedWindowStep: Int = 0

    // EMA tracking
    private var emaInitialized = false
    private var bestEMAAccuracy: Float = 0
    private var bestEMAAccuracyStep: Int = 0

    // Best weights tracking (saved on new smoothed accuracy peaks)
    private var bestSmoothedAccForSave: Float = 0
    private var bestSmoothedAccSaveStep: Int = 0

    // Adaptive depth growth
    private var currentNZ: Int
    private var bestRollingLoss: Float = Float.infinity
    private var rollingLossSum: Float = 0
    private var rollingLossCount: Int = 0
    private var stepsWithoutImprovement: Int = 0

    init(engine: MetalEngine, config: SimConfig, monitor: TrainingMonitor, liveConfig: LiveConfig? = nil) {
        self.engine = engine
        self.baseConfig = config
        self.liveConfig = liveConfig ?? LiveConfig(config)
        self.monitor = monitor
        self.currentNZ = config.clampedDepth(config.initialDepth)
        self.refModel = ReferenceModel(
            nInputs: config.nInputsA,
            nOutputs: config.nOutputsA,
            hiddenSize: 32,
            seed: config.seed,
            linear: config.useLinearTeacher
        )
        self.ioEncoder = IOEncoder(config: config)
        self.checkpoint = CheckpointManager(config: config)

        ioEncoder.uploadToEngine(engine)

        // Load MNIST dataset if configured
        if config.useImageInput && !config.mnistDataDir.isEmpty && !config.useRegressionMode {
            do {
                let loader = try MNISTLoader(directory: config.mnistDataDir, split: "train")
                let (imgs, lbls) = loader.filtered(classes: config.mnistClasses)
                self.mnistImages = imgs
                self.mnistLabels = lbls
                print("[MNIST] Ready: \(imgs.count) images, \(config.mnistClasses.count) classes, grid=\(config.nx)×\(config.ny)")
            } catch {
                print("[MNIST] Failed to load: \(error)")
            }
        }

        // Regression data generator (FFT or matrix transform)
        if config.useRegressionMode {
            let target = config.regressionTarget.lowercased()
            if target == "fft" {
                self.fftDataGen = FFTDataGenerator(
                    nx: config.nx, ny: config.ny,
                    inputSize: config.fftInputSize, seed: UInt64(config.seed))
                print("[Regression] FFT mode: \(config.nx)×\(config.ny), inputSize=\(config.fftInputSize)")
            } else {
                // Matrix transform regression
                let matType: MatrixTransformDataGenerator.MatrixType
                if target.hasPrefix("custom:") {
                    // Load custom matrix from binary file
                    let path = String(config.regressionTarget.dropFirst("custom:".count))
                    let N = config.nx * config.ny
                    guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                        fatalError("[Regression] Cannot read custom matrix file: \(path)")
                    }
                    let expectedBytes = N * N * MemoryLayout<Float>.size
                    guard fileData.count == expectedBytes else {
                        fatalError("[Regression] Custom matrix file has \(fileData.count) bytes, "
                            + "expected \(expectedBytes) for \(N)×\(N) Float32")
                    }
                    let floats = fileData.withUnsafeBytes { ptr in
                        Array(ptr.bindMemory(to: Float.self))
                    }
                    let fileName = (path as NSString).lastPathComponent
                    matType = .custom(data: floats, name: fileName)
                    print("[Regression] Loaded custom matrix: \(fileName) (\(N)×\(N))")
                } else {
                    switch target {
                    case "randomunitary":
                        matType = .randomUnitary(seed: UInt64(config.seed) &+ 12345)
                    case "hadamard":
                        matType = .hadamard
                    case "identity":
                        matType = .identity
                    default:
                        print("[Regression] Unknown target '\(target)', defaulting to randomUnitary")
                        matType = .randomUnitary(seed: UInt64(config.seed) &+ 12345)
                    }
                }
                let gen = MatrixTransformDataGenerator(
                    nx: config.nx, ny: config.ny,
                    matrixType: matType, seed: UInt64(config.seed))
                self.matrixDataGen = gen
                let orthErr = gen.verifyOrthogonality()
                print("[Regression] Matrix mode: \(gen.matrixType) \(config.nx)×\(config.ny), "
                    + "dim=\(gen.matrixDim), orthogonality error=\(orthErr)")
            }
        }
    }

    /// Convenience: current config snapshot from LiveConfig (use for tunable params)
    private var config: SimConfig { liveConfig.config }

    func start(fromStep: Int = 0) {
        startStep = fromStep
        shouldStop = false

        let bc = baseConfig
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.monitor.isRunning = true
            self.monitor.totalSteps = self.config.nSteps
            var summary = "\(bc.nx)×\(bc.ny)×\(bc.nz) B=\(bc.batchSize) lr=\(self.config.lr)"
            if bc.useDiscreteD2NN { summary += " DISCRETE(\(bc.nDiscreteLayers)L)" }
            if bc.useImageInput { summary += " MNIST(\(bc.mnistClasses.count)cls)" }
            if bc.saturationIntensity > 0 { summary += " SAT=\(bc.saturationIntensity)" }
            if bc.nPulses > 1 { summary += " P=\(bc.nPulses)" }
            if bc.useLinearTeacher { summary += " LIN" }
            self.monitor.configSummary = summary
            self.monitor.log("Training: \(bc.nx)×\(bc.ny)×\(bc.nz), \(self.config.nSteps) steps")
            if bc.saturationIntensity > 0 {
                self.monitor.log("Saturable absorption: I_sat=\(bc.saturationIntensity)")
            }
            if bc.nPulses > 1 {
                self.monitor.log("Recurrent: \(bc.nPulses) pulses, threshold=\(bc.thresholdInit), ratio=\(bc.feedbackRatioInit)")
            }
            self.monitor.log("Adaptive depth: nz=\(self.currentNZ)→\(bc.nz), patience=\(bc.depthGrowthPatience)")
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.runTrainingLoop()
        }
    }

    func pause() { shouldStop = true }
    func resume() { start(fromStep: startStep) }

    // MARK: - Main training loop

    private func runTrainingLoop() {
        // Run gradient check at startup for discrete D2NN mode
        if baseConfig.useDiscreteD2NN {
            runGradientCheck()
        }

        startTime = CFAbsoluteTimeGetCurrent()
        var lastUIUpdate: CFAbsoluteTime = 0
        var lastFieldUpdate: CFAbsoluteTime = 0
        let nPulses = baseConfig.nPulses
        let isRecurrent = nPulses > 1

        // Read nSteps from live config each iteration (can be extended while running)
        var step = startStep
        while step < config.nSteps {
            if shouldStop { break }

            semaphore.wait()

            let now = CFAbsoluteTimeGetCurrent()
            let completedIdx = lastCompletedFrameIdx
            if completedIdx >= 0 {
                if now - lastUIUpdate > 1.0 {
                    lastUIUpdate = now
                    postMetrics(step: step - 1, frame: engine.frames[completedIdx])
                }
                // Periodic console logging for terminal monitoring
                if step % baseConfig.logEvery == 0 && step > 4 {
                    logToConsole(step: step - 1, frame: engine.frames[completedIdx])
                }
                if now - lastFieldUpdate > 15.0 {
                    lastFieldUpdate = now
                    postFieldVisualization(frame: engine.frames[completedIdx])
                }
            }

            // Hot-reload I/O geometry if changed via live tuning
            if liveConfig.ioGeometryDirty {
                reloadIOGeometry()
                liveConfig.clearIOGeometryDirty()
            }

            // Hot-reload physics buffers (propagator, dn_levels) if changed via live tuning
            if liveConfig.physicsDirty {
                reloadPhysics()
                liveConfig.clearPhysicsDirty()
            }

            let frameIdx = step % 3
            let frame = engine.frames[frameIdx]

            if baseConfig.useRegressionMode, (fftDataGen != nil || matrixDataGen != nil) {
                packRegressionBatch(frame: frame)
            } else if baseConfig.useImageInput, let imgs = mnistImages, let lbls = mnistLabels {
                packMNISTBatch(images: imgs, labels: lbls, frame: frame)
            } else {
                let (inputs, targets) = refModel.generateBatch(size: baseConfig.batchSize)
                packInputData(inputs: inputs, targets: targets, frame: frame)
            }

            // Read tunable params fresh from LiveConfig each step
            let liveSnapshot = liveConfig.config
            let currentLR = liveSnapshot.learningRate(at: step)
            let stepParams = engine.makeParams(step: step, lr: currentLR, from: liveSnapshot)

            guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
                semaphore.signal()
                step += 1
                continue
            }
            cmdBuf.label = "train_step_\(step)"

            guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
                semaphore.signal()
                step += 1
                continue
            }

            // Zero accuracy counter
            memcpy(frame.accuracyCount.contents(), [UInt32(0)] as [UInt32], 4)

            let effNZ = currentNZ
            let activeVoxelCount = effNZ * baseConfig.fieldSize

            // Zero grad buffers
            encodeZeroGrads(encoder: encoder, activeVoxelCount: activeVoxelCount)

            // Zero recurrent gradient accumulators
            if isRecurrent {
                var zeroVal: Float = 0.0
                var one = UInt32(1)
                encoder.setComputePipelineState(engine.fillBufferPipeline)
                encoder.setBuffer(engine.gradThreshold, offset: 0, index: 0)
                encoder.setBytes(&zeroVal, length: 4, index: 1)
                encoder.setBytes(&one, length: 4, index: 2)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.setBuffer(engine.gradFeedbackRatio, offset: 0, index: 0)
                encoder.setBytes(&zeroVal, length: 4, index: 1)
                encoder.setBytes(&one, length: 4, index: 2)
                encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }

            let hasKappa = baseConfig.kappaMax > 0
            let isDirect = baseConfig.useDirect
            let isDiscrete = baseConfig.useDiscreteD2NN

            let isRegression = baseConfig.useRegressionMode

            if isDiscrete {
                // --- Discrete D2NN mode ---
                // Phase is the learnable parameter (deltaN buffer stores phases)
                if !isRegression {
                    // Classification: encode input spots/images
                    if baseConfig.useImageInput {
                        engine.encodeImageInput(encoder: encoder, frame: frame)
                    } else {
                        engine.encodeInput(encoder: encoder, frame: frame)
                    }
                }
                // Regression: input already written to frame.fields by packRegressionBatch
                engine.encodeDiscreteForwardPass(encoder: encoder, frame: frame, step: step, effectiveNZ: effNZ)
                if isRegression {
                    engine.encodeMSELossAndGradient(encoder: encoder, frame: frame)
                } else {
                    engine.encodeLossAndGradient(encoder: encoder, frame: frame, normScale: liveSnapshot.normScale)
                }
                engine.encodeDiscreteBackwardPass(encoder: encoder, frame: frame, effectiveNZ: effNZ)
                engine.encodeReduceGradients(encoder: encoder, frame: frame)
                engine.encodeScaleGradients(encoder: encoder)
                engine.encodeDirectAdamUpdate(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
            } else {
                // --- Continuous BPM mode (original) ---
                // Forward pass
                if !isDirect {
                    engine.encodeGumbelForward(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                }
                if hasKappa {
                    engine.encodeGumbelForwardKappa(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                }
                if !isRegression {
                    engine.encodeInput(encoder: encoder, frame: frame)
                }

                if isRecurrent {
                    engine.encodeCacheInput(encoder: encoder, frame: frame)
                    for p in 0..<nPulses {
                        if p > 0 {
                            engine.encodeSoftThreshold(encoder: encoder, frame: frame, pulseIndex: p - 1)
                            engine.encodeFeedbackMix(encoder: encoder, frame: frame, pulseIndex: p - 1)
                        }
                        engine.encodeForwardPass(encoder: encoder, frame: frame, step: step,
                                                effectiveNZ: effNZ, pulseIndex: p)
                    }
                } else {
                    engine.encodeForwardPass(encoder: encoder, frame: frame, step: step, effectiveNZ: effNZ)
                }

                // Loss + adjoint seed
                if isRegression {
                    engine.encodeMSELossAndGradient(encoder: encoder, frame: frame)
                } else {
                    engine.encodeLossAndGradient(encoder: encoder, frame: frame, normScale: liveSnapshot.normScale)
                }

                // Backward pass
                if isRecurrent {
                    for p in stride(from: nPulses - 1, through: 0, by: -1) {
                        engine.encodeBackwardBPM(encoder: encoder, frame: frame,
                                                effectiveNZ: effNZ, pulseIndex: p)
                        engine.encodeReduceGradients(encoder: encoder, frame: frame, effectiveNZ: effNZ)
                        if p > 0 {
                            engine.encodeFeedbackMixBackward(encoder: encoder, frame: frame, pulseIndex: p - 1)
                            engine.encodeSoftThresholdBackward(encoder: encoder, frame: frame, pulseIndex: p - 1)
                        }
                    }
                    engine.encodeScaleGradients(encoder: encoder, effectiveNZ: effNZ)
                    engine.encodeScalarAdamUpdate(encoder: encoder,
                        param: engine.thresholdBuffer, grad: engine.gradThreshold,
                        momentM: engine.thresholdMomentM, momentV: engine.thresholdMomentV,
                        params: stepParams)
                    engine.encodeScalarAdamUpdate(encoder: encoder,
                        param: engine.feedbackRatio, grad: engine.gradFeedbackRatio,
                        momentM: engine.ratioMomentM, momentV: engine.ratioMomentV,
                        params: stepParams)
                } else {
                    engine.encodeBackwardPass(encoder: encoder, frame: frame, effectiveNZ: effNZ)
                }

                if isDirect {
                    engine.encodeDirectAdamUpdate(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                } else {
                    engine.encodeGumbelBackward(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                    engine.encodeAdamUpdate(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                }
                if hasKappa {
                    engine.encodeGumbelBackwardKappa(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                    engine.encodeAdamUpdateKappa(encoder: encoder, params: stepParams, activeVoxels: activeVoxelCount)
                }
            }

            // EMA weight update (Polyak averaging) — runs on GPU, negligible cost
            let emaConfig = liveSnapshot
            if emaConfig.useEMA {
                let emaDecay = emaConfig.emaDecay
                if isDirect {
                    engine.encodeEMAUpdate(encoder: encoder,
                        current: engine.deltaN, ema: engine.emaDeltaN,
                        decay: emaDecay, count: activeVoxelCount)
                } else {
                    let totalLogits = activeVoxelCount * baseConfig.nLevels
                    engine.encodeEMAUpdate(encoder: encoder,
                        current: engine.logits, ema: engine.emaLogits,
                        decay: emaDecay, count: totalLogits)
                }
                if hasKappa {
                    let totalLogits = activeVoxelCount * baseConfig.nLevels
                    engine.encodeEMAUpdate(encoder: encoder,
                        current: engine.kappaLogits, ema: engine.emaKappaLogits,
                        decay: emaDecay, count: totalLogits)
                }
                if !emaInitialized {
                    emaInitialized = true
                    DispatchQueue.main.async { [weak self] in
                        self?.monitor.log("EMA weight averaging enabled (decay=\(emaDecay))")
                    }
                }
            }

            encoder.endEncoding()

            let capturedStep = step
            let capturedFrameIdx = frameIdx
            let isDebugStep = capturedStep < 5

            if isDebugStep {
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                debugCheckNaN(step: capturedStep, frame: engine.frames[capturedFrameIdx])
                lastCompletedFrameIdx = capturedFrameIdx
                startStep = capturedStep + 1
                semaphore.signal()
            } else {
                cmdBuf.addCompletedHandler { [weak self] _ in
                    guard let self else { return }
                    self.lastCompletedFrameIdx = capturedFrameIdx
                    self.startStep = capturedStep + 1
                    self.semaphore.signal()
                }
                cmdBuf.commit()
            }

            if step > 0 && step % baseConfig.checkpointInterval == 0 && !isDebugStep {
                cmdBuf.waitUntilCompleted()
                saveCheckpoint(step: step)
            }

            // EMA auto-swap: periodically replace active weights with EMA weights
            let emaSwapInterval = config.emaSwapInterval
            if config.useEMA && emaSwapInterval > 0 && step > 0 && step % emaSwapInterval == 0 && !isDebugStep {
                cmdBuf.waitUntilCompleted()
                engine.swapEMAToActive()
                // Also reset EMA to new position
                engine.resetEMAFromActive()
                // Zero momentum for clean restart from EMA position
                let nLogits = baseConfig.totalVoxels * baseConfig.nLevels
                memset(engine.momentM.contents(), 0, nLogits * 4)
                memset(engine.momentV.contents(), 0, nLogits * 4)
                DispatchQueue.main.async { [weak self] in
                    self?.monitor.log("EMA→active swap at step \(step) (momentum reset)")
                }
            }

            step += 1
        }

        for _ in 0..<3 { semaphore.wait() }
        for _ in 0..<3 { semaphore.signal() }

        saveCheckpoint(step: startStep)

        let wasStoppedManually = shouldStop
        DispatchQueue.main.async { [weak self] in
            self?.monitor.isRunning = false
            self?.monitor.log("Training complete at step \(self?.startStep ?? 0)")
            self?.monitor.finishCurrentRun(status: wasStoppedManually ? .stopped : .completed)
        }
    }

    // MARK: - Live I/O geometry reload

    /// Recompute and re-upload I/O geometry from the current LiveConfig
    private func reloadIOGeometry() {
        let currentConfig = liveConfig.config
        // Create a new IOEncoder with updated geometry but same grid dimensions
        var modifiedConfig = baseConfig
        modifiedConfig.inputXFraction = currentConfig.inputXFraction
        modifiedConfig.outputXFraction = currentConfig.outputXFraction
        let newIO = IOEncoder(config: modifiedConfig)
        newIO.uploadToEngine(engine)

        DispatchQueue.main.async { [weak self] in
            self?.monitor.log("I/O geometry reloaded: in=\(String(format: "%.2f", currentConfig.inputXFraction)) out=\(String(format: "%.2f", currentConfig.outputXFraction))")
        }
    }

    // MARK: - Live physics reload

    /// Recompute propagator and dn_levels from updated physics parameters
    private func reloadPhysics() {
        let currentConfig = liveConfig.config
        // Merge live physics params into baseConfig (preserving grid dimensions)
        var modifiedConfig = baseConfig
        modifiedConfig.dnMax = currentConfig.dnMax
        modifiedConfig.dz = currentConfig.dz
        modifiedConfig.kappaMax = currentConfig.kappaMax
        modifiedConfig.saturationIntensity = currentConfig.saturationIntensity

        engine.recomputePhysics(config: modifiedConfig)

        DispatchQueue.main.async { [weak self] in
            self?.monitor.log("Physics reloaded: dnMax=\(String(format: "%.4f", currentConfig.dnMax)) dz=\(String(format: "%.1fμm", currentConfig.dz * 1e6)) κ=\(String(format: "%.4f", currentConfig.kappaMax))")
        }
    }

    // MARK: - Gradient Check

    /// Run finite-difference gradient check at startup to verify the adjoint backward pass.
    /// Picks several random voxels, perturbs each by ±epsilon, and compares
    /// (L(phi+eps) - L(phi-eps)) / (2*eps) to the adjoint gradient.
    private func runGradientCheck() {
        dbg("[GradCheck] ===== Starting finite-difference gradient verification =====")

        let frame = engine.frames[0]
        let tv = baseConfig.totalVoxels
        let fs = baseConfig.fieldSize
        let B = baseConfig.batchSize

        // 1. Pack a fixed batch (will be reused for all forward passes)
        if baseConfig.useImageInput, let imgs = mnistImages, let lbls = mnistLabels {
            // Save RNG state and pack a deterministic batch
            let savedRng = mnistRng
            packMNISTBatch(images: imgs, labels: lbls, frame: frame)
            mnistRng = savedRng  // restore so training starts clean
        } else {
            let (inputs, targets) = refModel.generateBatch(size: B)
            packInputData(inputs: inputs, targets: targets, frame: frame)
        }

        // Use real propagator for gradient check (identity propagator is degenerate)
        let propPtr = engine.propagator.contents().bindMemory(to: SIMD2<Float>.self, capacity: fs)
        var savedProp: [SIMD2<Float>] = []
        for i in 0..<fs { savedProp.append(propPtr[i]) }
        dbg("[GradCheck] [DIAG] Using REAL propagator for gradient check")
        // Propagator diagnostic: check if H is approximately real (would make conj(H)=H → no adjoint effect)
        var propRealEnergy: Double = 0, propImagEnergy: Double = 0
        var propMinPhase: Double = .infinity, propMaxPhase: Double = -.infinity
        for i in 0..<fs {
            let h = propPtr[i]
            propRealEnergy += Double(h.x * h.x)
            propImagEnergy += Double(h.y * h.y)
            if h.x != 0 || h.y != 0 {
                let phase = atan2(Double(h.y), Double(h.x))
                propMinPhase = min(propMinPhase, phase)
                propMaxPhase = max(propMaxPhase, phase)
            }
        }
        let imagFraction = propImagEnergy / (propRealEnergy + propImagEnergy + 1e-30)
        dbg("[GradCheck] [DIAG] Propagator: imagFraction=\(String(format: "%.4f", imagFraction)) phaseRange=[\(String(format: "%.4f", propMinPhase)), \(String(format: "%.4f", propMaxPhase))]")
        // Sample a few propagator values
        let sampleIndices = [0, 1, fs/4, fs/2, fs*3/4, fs-1]
        for idx in sampleIndices {
            let h = propPtr[idx]
            dbg("[GradCheck] [DIAG]   H[\(idx)]=(\(String(format: "%+.6f", h.x)), \(String(format: "%+.6f", h.y))) |H|=\(String(format: "%.4f", sqrt(h.x*h.x + h.y*h.y)))")
        }

        // ADJOINT PROPERTY TEST: verify <P†(y), x> = <y, P(x)> for the propagator
        // This is the DEFINITIVE test that IFFT(conj(H)*FFT(·)) is the correct adjoint of IFFT(H*FFT(·))
        do {
            // Use batch 0 of the fields and adjoint buffers as scratch (will be overwritten later)
            let fieldPtr = frame.fields.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)
            let adjPtr = frame.adjoint.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)

            // Create random test vectors x and y (only batch 0)
            srand48(42)
            var xVals: [SIMD2<Float>] = []
            var yVals: [SIMD2<Float>] = []
            for i in 0..<fs {
                let xr = Float(drand48() * 2 - 1)
                let xi = Float(drand48() * 2 - 1)
                let yr = Float(drand48() * 2 - 1)
                let yi = Float(drand48() * 2 - 1)
                xVals.append(SIMD2<Float>(xr, xi))
                yVals.append(SIMD2<Float>(yr, yi))
                fieldPtr[i] = SIMD2<Float>(xr, xi)  // x in fields[batch 0]
                adjPtr[i] = SIMD2<Float>(yr, yi)     // y in adjoint[batch 0]
            }
            // Zero other batches
            for b in 1..<B {
                for i in 0..<fs {
                    fieldPtr[b * fs + i] = .zero
                    adjPtr[b * fs + i] = .zero
                }
            }

            // Forward propagation: P(x) = IFFT(H * FFT(x))
            do {
                guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else { fatalError("Failed to create encoder") }
                engine.encodeFFT2D(encoder: enc, field: frame.fields, temp: frame.fftTemp, inverse: false)
                enc.setComputePipelineState(engine.elemMulPipeline)
                enc.setBuffer(frame.fields, offset: 0, index: 0)
                enc.setBuffer(engine.propagator, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: B * fs, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, B * fs), height: 1, depth: 1))
                engine.encodeFFT2D(encoder: enc, field: frame.fields, temp: frame.fftTemp, inverse: true)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }
            // Read P(x) from fields[batch 0]
            var px: [SIMD2<Float>] = []
            for i in 0..<fs { px.append(fieldPtr[i]) }

            // Backward propagation: P†(y) = IFFT(conj(H) * FFT(y))
            do {
                guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else { fatalError("Failed to create encoder") }
                engine.encodeFFT2D(encoder: enc, field: frame.adjoint, temp: frame.fftTemp, inverse: false)
                enc.setComputePipelineState(engine.elemMulConjPipeline)
                enc.setBuffer(frame.adjoint, offset: 0, index: 0)
                enc.setBuffer(engine.propagator, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: B * fs, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(256, B * fs), height: 1, depth: 1))
                engine.encodeFFT2D(encoder: enc, field: frame.adjoint, temp: frame.fftTemp, inverse: true)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }
            // Read P†(y) from adjoint[batch 0]
            var pdy: [SIMD2<Float>] = []
            for i in 0..<fs { pdy.append(adjPtr[i]) }

            // Compute inner products: <P†(y), x> and <y, P(x)>
            // <a, b> = sum conj(a) * b = sum (a.x*b.x + a.y*b.y) + i*(a.x*b.y - a.y*b.x)
            var ipLeft_r: Double = 0, ipLeft_i: Double = 0   // <P†(y), x>
            var ipRight_r: Double = 0, ipRight_i: Double = 0  // <y, P(x)>
            for i in 0..<fs {
                // <P†(y), x> = sum conj(P†(y)[i]) * x[i]
                ipLeft_r += Double(pdy[i].x * xVals[i].x + pdy[i].y * xVals[i].y)
                ipLeft_i += Double(pdy[i].x * xVals[i].y - pdy[i].y * xVals[i].x)
                // <y, P(x)> = sum conj(y[i]) * P(x)[i]
                ipRight_r += Double(yVals[i].x * px[i].x + yVals[i].y * px[i].y)
                ipRight_i += Double(yVals[i].x * px[i].y - yVals[i].y * px[i].x)
            }
            let relErr_r = abs(ipLeft_r - ipRight_r) / (abs(ipLeft_r) + abs(ipRight_r) + 1e-15)
            let relErr_i = abs(ipLeft_i - ipRight_i) / (abs(ipLeft_i) + abs(ipRight_i) + 1e-15)
            dbg("[GradCheck] [ADJOINT-TEST] <P†(y),x> = (\(String(format: "%.6f", ipLeft_r)), \(String(format: "%.6f", ipLeft_i)))")
            dbg("[GradCheck] [ADJOINT-TEST] <y,P(x)>  = (\(String(format: "%.6f", ipRight_r)), \(String(format: "%.6f", ipRight_i)))")
            dbg("[GradCheck] [ADJOINT-TEST] relErr = (\(String(format: "%.2e", relErr_r)), \(String(format: "%.2e", relErr_i))) \(relErr_r < 1e-3 && relErr_i < 1e-3 ? "PASS" : "FAIL")")
        }

        // Helper: run forward + loss synchronously, return average loss
        // Uses Double accumulation to avoid float32 precision issues in the gradient check.
        func runForwardAndLoss() -> Double {
            guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return Double.nan }

            // Encode input → field
            if baseConfig.useImageInput {
                engine.encodeImageInput(encoder: enc, frame: frame)
            } else {
                engine.encodeInput(encoder: enc, frame: frame)
            }

            // Forward pass through discrete layers
            engine.encodeDiscreteForwardPass(encoder: enc, frame: frame, step: 0)

            // Loss computation
            memcpy(frame.accuracyCount.contents(), [UInt32(0)] as [UInt32], 4)
            engine.encodeLossAndGradient(encoder: enc, frame: frame)

            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Read average loss with double precision accumulation
            let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: B)
            var total: Double = 0
            for b in 0..<B { total += Double(lossPtr[b]) }
            return total / Double(B)
        }

        // Helper: run forward + loss, return per-sample losses array
        func runForwardAndLossPerSample() -> [Double] {
            guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return [] }
            if baseConfig.useImageInput {
                engine.encodeImageInput(encoder: enc, frame: frame)
            } else {
                engine.encodeInput(encoder: enc, frame: frame)
            }
            engine.encodeDiscreteForwardPass(encoder: enc, frame: frame, step: 0)
            memcpy(frame.accuracyCount.contents(), [UInt32(0)] as [UInt32], 4)
            engine.encodeLossAndGradient(encoder: enc, frame: frame)
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: B)
            return (0..<B).map { Double(lossPtr[$0]) }
        }

        // Helper: run forward + loss + backward, return (loss, gradient for voxel idx)
        // Also captures adjoint BEFORE backward pass (adjoint seed) for diagnostics.
        var adjointSeedSample: SIMD2<Float> = .zero  // diagnostic: adjoint at test pixel before backprop
        var adjointAfterBackprop: SIMD2<Float> = .zero  // diagnostic: adjoint at test pixel after backprop
        var diagOutputField: SIMD2<Float> = .zero  // output field at test pixel
        var diagTarget: UInt32 = 0  // target class for batch 0
        var diagLoss0: Float = 0  // loss for batch 0
        var diagPixel: Int = 0  // which pixel to diagnose

        func runFullPassAndGrad(voxelIdx: Int) -> (Double, Float) {
            let activeVoxels = tv
            diagPixel = voxelIdx % fs

            // Step 1: Forward + Loss (to capture adjoint seed)
            do {
                guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else { return (Double.nan, Float.nan) }

                // Zero grad buffers
                var zeroVal: Float = 0.0
                var count = UInt32(activeVoxels)
                let tg = MTLSize(width: min(256, activeVoxels), height: 1, depth: 1)
                let grid = MTLSize(width: activeVoxels, height: 1, depth: 1)
                enc.setComputePipelineState(engine.fillBufferPipeline)
                enc.setBuffer(engine.gradDeltaN, offset: 0, index: 0)
                enc.setBytes(&zeroVal, length: 4, index: 1)
                enc.setBytes(&count, length: 4, index: 2)
                enc.dispatchThreads(grid, threadsPerThreadgroup: tg)

                if baseConfig.useImageInput {
                    engine.encodeImageInput(encoder: enc, frame: frame)
                } else {
                    engine.encodeInput(encoder: enc, frame: frame)
                }
                engine.encodeDiscreteForwardPass(encoder: enc, frame: frame, step: 0)
                memcpy(frame.accuracyCount.contents(), [UInt32(0)] as [UInt32], 4)
                engine.encodeLossAndGradient(encoder: enc, frame: frame)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }

            // Read adjoint seed and output field (before backward pass)
            let adjPtr = frame.adjoint.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)
            adjointSeedSample = adjPtr[0 * fs + diagPixel]  // batch 0
            // Also read output field at this pixel (for adjoint seed verification)
            let fieldsPtr = frame.fields.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)
            diagOutputField = fieldsPtr[0 * fs + diagPixel]  // batch 0
            // Read target class and loss for batch 0
            let targetPtr = frame.targetClasses.contents().bindMemory(to: UInt32.self, capacity: B)
            diagTarget = targetPtr[0]
            diagLoss0 = frame.losses.contents().bindMemory(to: Float.self, capacity: B)[0]

            // Step 2: Backward pass
            do {
                guard let cmdBuf = engine.commandQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else { return (Double.nan, Float.nan) }
                engine.encodeDiscreteBackwardPass(encoder: enc, frame: frame)
                engine.encodeReduceGradients(encoder: enc, frame: frame)
                engine.encodeScaleGradients(encoder: enc)
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
            }

            // Read adjoint after backward (should differ from seed due to back-propagation)
            adjointAfterBackprop = adjPtr[0 * fs + diagPixel]

            // Read loss
            let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: B)
            var total: Double = 0
            for b in 0..<B { total += Double(lossPtr[b]) }
            let loss = total / Double(B)

            let gradPtr = engine.gradDeltaN.contents().bindMemory(to: Float.self, capacity: tv)
            let grad = gradPtr[voxelIdx]
            return (loss, grad)
        }

        let dnPtr = engine.deltaN.contents().bindMemory(to: Float.self, capacity: tv)
        // Use large epsilon for adequate SNR with float32 GPU computation.
        // On large grids (128×128), perturbing 1 voxel creates tiny loss changes
        // that are near float32 precision limits at small eps.
        // eps=0.1 gives good SNR while sin(0.1)≈0.0998 (0.17% nonlinearity).
        let eps: Float = 0.1

        // Test voxels across layers, avoiding grid center (symmetry artifacts)
        let nz = baseConfig.nz
        let testVoxels: [(String, Int)] = [
            ("layer0_q1", 0 * fs + (baseConfig.nx/4) * baseConfig.ny + baseConfig.ny/4),
            ("layer0_q3", 0 * fs + (3*baseConfig.nx/4) * baseConfig.ny + baseConfig.ny/4),
            ("layer0_off", 0 * fs + 5 * baseConfig.ny + 7),
            ("mid_q1", (nz/2) * fs + (baseConfig.nx/4) * baseConfig.ny + baseConfig.ny/4),
            ("mid_off", (nz/2) * fs + 10 * baseConfig.ny + 3),
            ("last_q1", (nz-1) * fs + (baseConfig.nx/4) * baseConfig.ny + baseConfig.ny/4),
            ("last_q3", (nz-1) * fs + (3*baseConfig.nx/4) * baseConfig.ny + 3*baseConfig.ny/4),
            ("last_off", (nz-1) * fs + 5 * baseConfig.ny + 7),
        ]

        dbg("[GradCheck] eps=\(eps), batch=\(B), nVoxels=\(tv), nLayers=\(nz), nx=\(baseConfig.nx)")

        // One-time comprehensive loss verification for batch 0 using first test voxel
        do {
            let firstVoxelIdx = testVoxels[0].1
            let _ = runFullPassAndGrad(voxelIdx: firstVoxelIdx)
            // Now frame has the output field, losses, adjoint, etc.
            let outFields = frame.fields.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)
            let targetPtr = frame.targetClasses.contents().bindMemory(to: UInt32.self, capacity: B)
            let lossGPU = frame.losses.contents().bindMemory(to: Float.self, capacity: B)[0]
            let target0 = Int(targetPtr[0])
            let nOutputsA = baseConfig.nOutputsA

            // Read detector mask data
            let maskOffsetsPtr = engine.detectorOffsets.contents().bindMemory(to: UInt32.self, capacity: nOutputsA + 1)
            let maskPixelsPtr = engine.detectorMaskData.contents().bindMemory(to: UInt32.self, capacity: Int(maskOffsetsPtr[nOutputsA]))

            // Compute detector scores for batch 0
            var scores: [Double] = []
            for c in 0..<nOutputsA {
                let start = Int(maskOffsetsPtr[c])
                let end = Int(maskOffsetsPtr[c + 1])
                var intensity: Double = 0
                for p in start..<end {
                    let pixel = Int(maskPixelsPtr[p])
                    let f = outFields[0 * fs + pixel]
                    intensity += Double(f.x * f.x + f.y * f.y)
                }
                scores.append(intensity)
            }

            let normScale = Double(baseConfig.normScale)
            let normScores = scores.map { $0 * normScale }
            let maxS = normScores.max() ?? 0
            let exps = normScores.map { exp($0 - maxS) }
            let sumExp = exps.reduce(0, +)
            let softmax = exps.map { $0 / sumExp }

            let alpha = 0.1
            var cpuLoss: Double = 0
            var dLdScores: [Double] = []
            for c in 0..<nOutputsA {
                let smoothT = (c == target0) ? (1.0 - alpha + alpha / Double(nOutputsA)) : (alpha / Double(nOutputsA))
                cpuLoss -= smoothT * log(softmax[c] + 1e-10)
                dLdScores.append((softmax[c] - smoothT) * normScale)
            }

            dbg("[GradCheck] [LOSS-VERIFY] target=\(target0) nDet=\(nOutputsA)")
            dbg("[GradCheck] [LOSS-VERIFY] scores: \(scores.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
            dbg("[GradCheck] [LOSS-VERIFY] normScores: \(normScores.map { String(format: "%.4f", $0) }.joined(separator: ", "))")
            dbg("[GradCheck] [LOSS-VERIFY] softmax: \(softmax.map { String(format: "%.6f", $0) }.joined(separator: ", "))")
            dbg("[GradCheck] [LOSS-VERIFY] cpuLoss=\(String(format: "%.6f", cpuLoss)) gpuLoss=\(String(format: "%.6f", lossGPU))")
            dbg("[GradCheck] [LOSS-VERIFY] dL/dScore: \(dLdScores.map { String(format: "%+.6f", $0) }.joined(separator: ", "))")

            // Verify adjoint seed at a detector pixel (first pixel of detector 0)
            let det0Start = Int(maskOffsetsPtr[0])
            let det0Pixel = Int(maskPixelsPtr[det0Start])
            let adjSeed0 = frame.adjoint.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)[0 * fs + det0Pixel]
            let eOut0 = outFields[0 * fs + det0Pixel]
            dbg("[GradCheck] [LOSS-VERIFY] det0_pixel=\(det0Pixel)")
            dbg("[GradCheck] [LOSS-VERIFY]   E_out=(\(String(format: "%+.6f", eOut0.x)), \(String(format: "%+.6f", eOut0.y)))")
            dbg("[GradCheck] [LOSS-VERIFY]   adjSeed=(\(String(format: "%+.6f", adjSeed0.x)), \(String(format: "%+.6f", adjSeed0.y)))")
            if abs(eOut0.x) > 1e-6 && abs(eOut0.y) > 1e-6 {
                let expectedAdj_x = Float(dLdScores[0]) * eOut0.x
                let expectedAdj_y = Float(dLdScores[0]) * eOut0.y
                dbg("[GradCheck] [LOSS-VERIFY]   expected_adj=(\(String(format: "%+.6f", expectedAdj_x)), \(String(format: "%+.6f", expectedAdj_y)))")
                dbg("[GradCheck] [LOSS-VERIFY]   match_x=\(abs(adjSeed0.x - expectedAdj_x) < 0.0001 ? "YES" : "NO(\(String(format: "%.6f", adjSeed0.x - expectedAdj_x)))") match_y=\(abs(adjSeed0.y - expectedAdj_y) < 0.0001 ? "YES" : "NO(\(String(format: "%.6f", adjSeed0.y - expectedAdj_y)))")")
            }
        }

        // Collect all gradient pairs for cosine similarity
        var numGrads: [Double] = []
        var adjGrads: [Double] = []
        var nOK = 0, nWarn = 0, nFail = 0

        for (name, voxelIdx) in testVoxels {
            guard voxelIdx < tv else { continue }

            let originalPhase = dnPtr[voxelIdx]

            // Get adjoint gradient + diagnostic readback
            let (loss0, adjGrad) = runFullPassAndGrad(voxelIdx: voxelIdx)

            // Diagnostic: read per_batch_grad for this voxel across batch elements
            let layer = voxelIdx / fs
            let pixel = voxelIdx % fs
            let pbGradPtr = frame.perBatchGrad.contents().bindMemory(to: Float.self, capacity: B * tv)
            var pbGradSamples: [Float] = []
            var pbGradSum: Double = 0
            for b in 0..<B {
                let val = pbGradPtr[b * tv + voxelIdx]
                pbGradSamples.append(val)
                pbGradSum += Double(val)
            }
            let manualReducedGrad = pbGradSum / Double(B)

            // Diagnostic: read adjoint at the voxel position (batch 0)
            let adjPtr = frame.adjoint.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * fs)
            let adj0 = adjPtr[0 * fs + pixel]  // batch 0, pixel in the field

            // Diagnostic: read field cache at this z-layer (batch 0)
            let cachePtr = frame.fieldCache.contents().bindMemory(to: SIMD2<Float>.self, capacity: B * (nz + 1) * fs)
            let cached0 = cachePtr[0 * (nz + 1) * fs + layer * fs + pixel]

            // Diagnostic: read output field at detector positions (batch 0) — use z=nLayers cache
            let output0_det0_pix = cachePtr[0 * (nz + 1) * fs + nz * fs + 0]  // first pixel of output

            dbg("[GradCheck] [DIAG] \(name): layer=\(layer) pixel=\(pixel)")
            dbg("[GradCheck] [DIAG]   perBatchGrad[b=0..3]: \(pbGradSamples.prefix(4).map { String(format: "%+.6f", $0) }.joined(separator: ", "))")
            dbg("[GradCheck] [DIAG]   manualReducedGrad(avg): \(String(format: "%+.8f", manualReducedGrad))")
            dbg("[GradCheck] [DIAG]   gradDeltaN[idx]: \(String(format: "%+.8f", adjGrad))")
            dbg("[GradCheck] [DIAG]   adjoint[b=0,pixel]: (\(String(format: "%+.6f", adj0.x)), \(String(format: "%+.6f", adj0.y)))")
            dbg("[GradCheck] [DIAG]   fieldCache[b=0,z=\(layer),pixel]: (\(String(format: "%+.6f", cached0.x)), \(String(format: "%+.6f", cached0.y)))")
            dbg("[GradCheck] [DIAG]   adjointSeed[b=0]: (\(String(format: "%+.6f", adjointSeedSample.x)), \(String(format: "%+.6f", adjointSeedSample.y)))")
            dbg("[GradCheck] [DIAG]   adjointAfterBackprop[b=0]: (\(String(format: "%+.6f", adjointAfterBackprop.x)), \(String(format: "%+.6f", adjointAfterBackprop.y)))")
            dbg("[GradCheck] [DIAG]   outputField[b=0]: (\(String(format: "%+.6f", diagOutputField.x)), \(String(format: "%+.6f", diagOutputField.y)))")
            dbg("[GradCheck] [DIAG]   target[b=0]=\(diagTarget) loss[b=0]=\(String(format: "%.4f", diagLoss0))")
            // Verify adjoint seed: if this pixel is in a detector, adjointSeed should = dL_dscore * E_out
            // dL_dscore is real, so adjointSeed.x/E_out.x should equal adjointSeed.y/E_out.y
            if abs(diagOutputField.x) > 1e-6 && abs(diagOutputField.y) > 1e-6 {
                let ratio_x = adjointSeedSample.x / diagOutputField.x
                let ratio_y = adjointSeedSample.y / diagOutputField.y
                dbg("[GradCheck] [DIAG]   adjSeed/Eout: x_ratio=\(String(format: "%+.6f", ratio_x)) y_ratio=\(String(format: "%+.6f", ratio_y)) \(abs(ratio_x - ratio_y) < 0.001 ? "CONSISTENT" : "INCONSISTENT")")
            }

            // CPU-side gradient verification for batch 0:
            // adj0 is adjoint AFTER reverse phase screen: adj_after = adj_before * conj(phasor)
            // So adj_before = adj_after * phasor (since |phasor|=1)
            // Then expected_grad = -2 * Im(conj(adj_before) * field_after)
            let dn = originalPhase  // deltaN at this voxel
            let cosP = cos(dn)  // k0=1, dz=1 → phase = dn
            let sinP = sin(dn)
            let phasorR = cosP, phasorI = sinP
            // adj_before = adj0 * phasor (undo conj(phasor) multiplication)
            let adjBeforeR = adj0.x * phasorR - adj0.y * phasorI
            let adjBeforeI = adj0.x * phasorI + adj0.y * phasorR
            // field_after = cached0 * phasor
            let fieldAfterR = cached0.x * phasorR - cached0.y * phasorI
            let fieldAfterI = cached0.x * phasorI + cached0.y * phasorR
            // conj(adj_before) * field_after
            let prodR = adjBeforeR * fieldAfterR + adjBeforeI * fieldAfterI
            let prodI = adjBeforeR * fieldAfterI - adjBeforeI * fieldAfterR
            // expected gradient = -2 * Im(conj(adj_before) * field_after)
            let cpuGrad = -2.0 * prodI
            dbg("[GradCheck] [DIAG]   cpuGrad[b=0]: \(String(format: "%+.6f", cpuGrad)) vs perBatchGrad[b=0]: \(String(format: "%+.6f", pbGradSamples[0]))")

            // Perturb +eps — get per-sample losses
            dnPtr[voxelIdx] = originalPhase + eps
            let lossesPlus = runForwardAndLossPerSample()
            let lossPlus = lossesPlus.reduce(0.0, +) / Double(B)

            // Perturb -eps — get per-sample losses
            dnPtr[voxelIdx] = originalPhase - eps
            let lossesMinus = runForwardAndLossPerSample()
            let lossMinus = lossesMinus.reduce(0.0, +) / Double(B)

            // Restore
            dnPtr[voxelIdx] = originalPhase

            // Per-sample gradient comparison (first voxel only, to keep output manageable)
            if name == "layer0_q1" || name == "last_q1" {
                dbg("[GradCheck] [PER-SAMPLE] \(name):")
                var sampleMatch = 0
                var sampleFail = 0
                for b in 0..<min(B, 8) {
                    let numG = (lossesPlus[b] - lossesMinus[b]) / Double(2 * eps)
                    let adjG = Double(pbGradSamples[b])
                    let agree = (numG > 0) == (adjG > 0) || abs(numG) < 1e-6
                    if agree { sampleMatch += 1 } else { sampleFail += 1 }
                    dbg("[GradCheck] [PER-SAMPLE]   b=\(b): numGrad=\(String(format:"%+.6f", numG)) adjGrad=\(String(format:"%+.6f", adjG)) \(agree ? "✓" : "✗")")
                }
                dbg("[GradCheck] [PER-SAMPLE]   match=\(sampleMatch) fail=\(sampleFail)")
            }

            // Central difference (all in Double for precision)
            let numGrad = (lossPlus - lossMinus) / Double(2 * eps)
            let adjGradD = Double(adjGrad)
            numGrads.append(numGrad)
            adjGrads.append(adjGradD)

            let absErr = abs(numGrad - adjGradD)
            let relErr = absErr / (abs(numGrad) + abs(adjGradD) + 1e-8)

            let status: String
            if relErr < 0.1 { status = "OK"; nOK += 1 }
            else if relErr < 0.3 { status = "WARN"; nWarn += 1 }
            else { status = "FAIL"; nFail += 1 }
            dbg("[GradCheck] [\(status)] \(name) (idx=\(voxelIdx)): numGrad=\(String(format:"%+.6f", numGrad)) adjGrad=\(String(format:"%+.6f", adjGrad)) relErr=\(String(format:"%.2f", relErr)) delta=\(String(format:"%.2e", lossPlus - lossMinus))")
        }

        // Cosine similarity: measures if gradients point in the same direction
        // This is the most important metric — if cos_sim > 0.9, training will work.
        var dot: Double = 0, normN: Double = 0, normA: Double = 0
        for i in 0..<numGrads.count {
            dot += numGrads[i] * adjGrads[i]
            normN += numGrads[i] * numGrads[i]
            normA += adjGrads[i] * adjGrads[i]
        }
        let cosSim = dot / (sqrt(normN * normA) + 1e-15)
        dbg("[GradCheck] Cosine similarity: \(String(format:"%.4f", cosSim)) (\(cosSim > 0.9 ? "GOOD" : cosSim > 0.5 ? "MARGINAL" : "BAD"))")
        dbg("[GradCheck] ===== Summary: \(nOK) OK, \(nWarn) WARN, \(nFail) FAIL =====")

        // Restore original propagator
        for i in 0..<fs { propPtr[i] = savedProp[i] }
        dbg("[GradCheck] [DIAG] Propagator restored to original values")
    }

    // MARK: - Helpers

    private func encodeZeroGrads(encoder: MTLComputeCommandEncoder, activeVoxelCount: Int) {
        var zeroVal: Float = 0.0
        var count = UInt32(activeVoxelCount)
        let tg = MTLSize(width: min(256, activeVoxelCount), height: 1, depth: 1)
        let grid = MTLSize(width: activeVoxelCount, height: 1, depth: 1)

        encoder.setComputePipelineState(engine.fillBufferPipeline)
        encoder.setBuffer(engine.gradDeltaN, offset: 0, index: 0)
        encoder.setBytes(&zeroVal, length: 4, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)

        encoder.setBuffer(engine.gradDeltaKappa, offset: 0, index: 0)
        encoder.setBytes(&zeroVal, length: 4, index: 1)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(grid, threadsPerThreadgroup: tg)
    }

    private func packInputData(inputs: [[Float]], targets: [Int], frame: MetalEngine.TrainingFrame) {
        let nActive = baseConfig.nInputsA
        let inputPtr = frame.inputData.contents().bindMemory(to: Float.self, capacity: baseConfig.batchSize * nActive)
        let targetPtr = frame.targetClasses.contents().bindMemory(to: UInt32.self, capacity: baseConfig.batchSize)

        for b in 0..<baseConfig.batchSize {
            for i in 0..<nActive {
                inputPtr[b * nActive + i] = inputs[b][i]
            }
            targetPtr[b] = UInt32(targets[b])
        }
    }

    /// Pack a batch of MNIST images into the GPU buffer.
    /// Images are resized from 28×28 to NX×NY and written as [B * NX * NY] floats.
    private func packMNISTBatch(images: [[Float]], labels: [UInt8], frame: MetalEngine.TrainingFrame) {
        let B = baseConfig.batchSize
        let nx = baseConfig.nx
        let ny = baseConfig.ny
        let npixels = nx * ny
        let inputPtr = frame.inputData.contents().bindMemory(to: Float.self, capacity: B * npixels)
        let targetPtr = frame.targetClasses.contents().bindMemory(to: UInt32.self, capacity: B)
        let nImages = images.count

        for b in 0..<B {
            // Random sample with simple xorshift
            mnistRng ^= mnistRng << 13
            mnistRng ^= mnistRng >> 7
            mnistRng ^= mnistRng << 17
            let idx = Int(mnistRng % UInt64(nImages))

            let resized = MNISTLoader.resize(image: images[idx], targetW: nx, targetH: ny)
            let offset = b * npixels
            for p in 0..<npixels {
                inputPtr[offset + p] = resized[p]
            }
            targetPtr[b] = UInt32(labels[idx])
        }
    }

    /// Pack a regression batch: write complex input to fields, target transform to targetField.
    /// Supports both FFT (via fftDataGen) and matrix transforms (via matrixDataGen).
    private func packRegressionBatch(frame: MetalEngine.TrainingFrame) {
        let B = baseConfig.batchSize
        let N = baseConfig.fieldSize  // NX * NY

        let (inputs, targets): ([Float], [Float])
        if let gen = fftDataGen {
            (inputs, targets) = gen.generateBatch(batchSize: B)
        } else if let gen = matrixDataGen {
            (inputs, targets) = gen.generateBatch(batchSize: B)
        } else {
            return
        }

        // Write input complex field directly to frame.fields (float2 = 2 floats per pixel)
        let fieldPtr = frame.fields.contents().bindMemory(to: Float.self, capacity: B * N * 2)
        memcpy(fieldPtr, inputs, B * N * 2 * MemoryLayout<Float>.size)

        // Write target to frame.targetField
        let targetPtr = frame.targetField.contents().bindMemory(to: Float.self, capacity: B * N * 2)
        memcpy(targetPtr, targets, B * N * 2 * MemoryLayout<Float>.size)
    }

    // MARK: - Metrics

    private func postMetrics(step: Int, frame: MetalEngine.TrainingFrame) {
        let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: baseConfig.batchSize)
        let accPtr = frame.accuracyCount.contents().bindMemory(to: UInt32.self, capacity: 1)

        var totalLoss: Float = 0
        for b in 0..<baseConfig.batchSize { totalLoss += lossPtr[b] }
        let avgLoss = totalLoss / Float(baseConfig.batchSize)
        let accuracy: Float
        if baseConfig.useRegressionMode {
            // For regression: report PSNR as "accuracy" (higher is better)
            // PSNR = 10 * log10(1.0 / MSE), normalized to [0,1] as PSNR/100
            let mse = max(avgLoss, 1e-10)
            let psnr = 10.0 * log10(1.0 / mse)
            accuracy = min(max(psnr, 0), 100) / 100.0
        } else {
            accuracy = Float(accPtr[0]) / Float(baseConfig.batchSize)
        }

        let now = CFAbsoluteTimeGetCurrent()
        let windowElapsed = now - speedWindowStart
        let windowSteps = step - speedWindowStep
        let stepsPerSec: Double
        if windowElapsed > 0.5 && windowSteps > 0 {
            stepsPerSec = Double(windowSteps) / windowElapsed
        } else {
            stepsPerSec = lastStepsPerSec
        }
        speedWindowStart = now
        speedWindowStep = step
        let temp = config.temperature(at: step)

        self.lastLoss = avgLoss
        self.lastAccuracy = accuracy
        self.lastStepsPerSec = stepsPerSec

        // Adaptive depth growth
        rollingLossSum += avgLoss
        rollingLossCount += 1
        if rollingLossCount >= 50 {
            let rollingAvg = rollingLossSum / Float(rollingLossCount)
            if rollingAvg < bestRollingLoss - baseConfig.depthGrowthThreshold {
                bestRollingLoss = rollingAvg
                stepsWithoutImprovement = 0
            } else {
                stepsWithoutImprovement += rollingLossCount
            }
            rollingLossSum = 0
            rollingLossCount = 0

            if stepsWithoutImprovement >= baseConfig.depthGrowthPatience && currentNZ < baseConfig.nz {
                let newNZ = baseConfig.clampedDepth(currentNZ + baseConfig.depthGrowthIncrement)
                if newNZ > currentNZ {
                    let oldNZ = currentNZ
                    currentNZ = newNZ
                    stepsWithoutImprovement = 0
                    bestRollingLoss = Float.infinity
                    DispatchQueue.main.async { [weak self] in
                        self?.monitor.log("Depth grown: nz \(oldNZ) → \(newNZ)")
                    }
                }
            }
        }

        // Save "best weights" when smoothed accuracy hits new high
        // Use smoothed accuracy (not raw) to avoid saving on noisy spikes
        let emaA: Float = 0.05  // same as monitor's emaAlpha
        let currentSmoothed = emaA * accuracy + (1.0 - emaA) * bestSmoothedAccForSave
        if step > 500 && currentSmoothed > bestSmoothedAccForSave + 0.01 {
            bestSmoothedAccForSave = currentSmoothed
            bestSmoothedAccSaveStep = step
            engine.saveBestWeights()
            DispatchQueue.main.async { [weak self] in
                self?.monitor.log("Best weights saved at step \(step) (smoothed acc \(String(format: "%.1f", currentSmoothed * 100))%)")
            }
        }
        // Update local smoothed tracker (simpler than full EMA, just for save threshold)
        bestSmoothedAccForSave = max(bestSmoothedAccForSave, currentSmoothed * 0.999)

        DispatchQueue.main.async { [weak self] in
            self?.monitor.addMetrics(step: step, loss: avgLoss, accuracy: accuracy,
                                     temperature: temp, stepsPerSec: stepsPerSec)
        }
    }

    private func postFieldVisualization(frame: MetalEngine.TrainingFrame) {
        let fieldPtr = frame.fields.contents().bindMemory(to: SIMD2<Float>.self, capacity: baseConfig.batchSize * baseConfig.fieldSize)
        var magnitude = [Float](repeating: 0, count: baseConfig.fieldSize)
        for i in 0..<baseConfig.fieldSize {
            let f = fieldPtr[i]
            magnitude[i] = f.x * f.x + f.y * f.y
        }

        let dnPtr = engine.deltaN.contents().bindMemory(to: Float.self, capacity: baseConfig.totalVoxels)
        let z = baseConfig.nz / 2
        var slice = [Float](repeating: 0, count: baseConfig.fieldSize)
        let offset = z * baseConfig.fieldSize
        for i in 0..<baseConfig.fieldSize { slice[i] = dnPtr[offset + i] }

        DispatchQueue.main.async { [weak self] in
            self?.monitor.outputFieldMagnitude = magnitude
            self?.monitor.voxelSlice = slice
        }
    }

    // MARK: - Console logging

    private func logToConsole(step: Int, frame: MetalEngine.TrainingFrame) {
        let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: baseConfig.batchSize)
        let accPtr = frame.accuracyCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        var totalLoss: Float = 0
        for b in 0..<baseConfig.batchSize { totalLoss += lossPtr[b] }
        let avgLoss = totalLoss / Float(baseConfig.batchSize)
        let lr = config.learningRate(at: step)
        if baseConfig.useRegressionMode {
            let mse = avgLoss
            let psnr = 10.0 * log10(1.0 / max(mse, 1e-10))
            dbg("[STEP] step=\(step) MSE=\(String(format: "%.6f", mse)) PSNR=\(String(format: "%.1f", psnr))dB lr=\(String(format: "%.5f", lr)) nz=\(currentNZ)")
        } else {
            let accuracy = Float(accPtr[0]) / Float(baseConfig.batchSize)
            let temp = config.temperature(at: step)
            dbg("[STEP] step=\(step) loss=\(String(format: "%.4f", avgLoss)) acc=\(String(format: "%.1f", accuracy * 100))% T=\(String(format: "%.3f", temp)) lr=\(String(format: "%.5f", lr)) nz=\(currentNZ)")
        }
    }

    // MARK: - Debug

    private static let debugLogPath = NSHomeDirectory() + "/.photonic_debug.log"

    func dbg(_ msg: String) {
        let line = msg + "\n"
        FileHandle.standardError.write(Data(line.utf8))
        if let fh = FileHandle(forWritingAtPath: Self.debugLogPath) {
            fh.seekToEndOfFile()
            fh.write(Data(line.utf8))
            fh.closeFile()
        } else {
            FileManager.default.createFile(atPath: Self.debugLogPath, contents: Data(line.utf8))
        }
    }

    private func debugCheckNaN(step: Int, frame: MetalEngine.TrainingFrame) {
        func checkBuffer(_ buf: MTLBuffer, name: String, count: Int) {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            var nanCount = 0
            var infCount = 0
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            var sumAbs: Double = 0
            for i in 0..<count {
                let v = ptr[i]
                if v.isNaN { nanCount += 1 }
                if v.isInfinite { infCount += 1 }
                if v.isFinite { minVal = min(minVal, v); maxVal = max(maxVal, v); sumAbs += Double(abs(v)) }
            }
            let mean = sumAbs / Double(max(count, 1))
            if nanCount > 0 || infCount > 0 {
                dbg("[NaN] step=\(step) \(name): \(nanCount) NaN, \(infCount) Inf in \(count)")
            } else {
                dbg("[OK] step=\(step) \(name): (\(minVal)...\(maxVal)) mean=\(String(format: "%.6g", mean)) [\(count)]")
            }
        }

        // Compact per-step loss for trend tracking
        let lossPtr = frame.losses.contents().bindMemory(to: Float.self, capacity: baseConfig.batchSize)
        var totalLoss: Float = 0
        for b in 0..<baseConfig.batchSize { totalLoss += lossPtr[b] }
        let avgLoss = totalLoss / Float(baseConfig.batchSize)
        let accPtr = frame.accuracyCount.contents().bindMemory(to: UInt32.self, capacity: 1)
        let accuracy = Float(accPtr[0]) / Float(baseConfig.batchSize)
        dbg("[STEP] step=\(step) loss=\(String(format: "%.4f", avgLoss)) acc=\(String(format: "%.1f", accuracy * 100))%")

        // Full diagnostics only for first 3 steps
        guard step < 3 else { return }

        let tv = baseConfig.totalVoxels
        let nl = baseConfig.nLevels
        checkBuffer(engine.deltaN, name: "deltaN", count: tv)
        checkBuffer(engine.gradDeltaN, name: "gradDeltaN", count: tv)
        checkBuffer(engine.gradLogits, name: "gradLogits", count: tv * nl)
        checkBuffer(engine.logits, name: "logits", count: tv * nl)
        checkBuffer(frame.losses, name: "losses", count: baseConfig.batchSize)

        // Check adjoint magnitude
        let adjCount = baseConfig.batchSize * baseConfig.fieldSize * 2
        let adjPtr = frame.adjoint.contents().bindMemory(to: Float.self, capacity: adjCount)
        var adjSumAbs: Double = 0
        var adjMax: Float = 0
        for i in 0..<adjCount {
            let v = abs(adjPtr[i])
            adjSumAbs += Double(v)
            adjMax = max(adjMax, v)
        }
        dbg("[OK] step=\(step) adjoint: mean=\(String(format: "%.6g", adjSumAbs/Double(adjCount))) max=\(adjMax)")

        // Check field imaginary component magnitude
        let fieldCount = baseConfig.batchSize * baseConfig.fieldSize
        let fieldPtr = frame.fields.contents().bindMemory(to: SIMD2<Float>.self, capacity: fieldCount)
        var realSum: Double = 0
        var imagSum: Double = 0
        for i in 0..<fieldCount {
            let f = fieldPtr[i]
            realSum += Double(f.x * f.x)
            imagSum += Double(f.y * f.y)
        }
        let imagFrac = imagSum / max(realSum + imagSum, 1e-30)
        dbg("[OK] step=\(step) field: real_energy=\(String(format:"%.4g", realSum)) imag_energy=\(String(format:"%.4g", imagSum)) imag_fraction=\(String(format:"%.4g", imagFrac))")
    }

    // MARK: - Checkpointing

    private func saveCheckpoint(step: Int) {
        do {
            let metrics = CheckpointManager.MetricsSnapshot(
                loss: lastLoss, accuracy: lastAccuracy, stepsPerSecond: lastStepsPerSec)
            try checkpoint.save(
                step: step,
                logits: engine.logits, momentM: engine.momentM, momentV: engine.momentV,
                kappaLogits: engine.kappaLogits,
                kappaMomentM: engine.kappaMomentM, kappaMomentV: engine.kappaMomentV,
                config: baseConfig, metrics: metrics)
            DispatchQueue.main.async { [weak self] in
                self?.monitor.log("Checkpoint saved: step \(step)")
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.monitor.log("Checkpoint failed: \(error)")
            }
        }
    }
}
