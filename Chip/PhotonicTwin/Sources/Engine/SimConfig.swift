import Foundation

/// All simulation parameters — physics, grid, training, and runtime.
struct SimConfig {
    // MARK: - Grid dimensions (must be power of 2)
    var nx: Int = 128
    var ny: Int = 128
    var nz: Int = 40

    // MARK: - Quantization
    let nLevels: Int = 16          // 4-bit: 16 refractive index levels

    // MARK: - Physics
    var wavelength: Double = 1.55e-6    // 1550 nm telecom C-band
    var nBase: Double = 1.45            // fused silica
    var dnMax: Double = 0.007           // max refractive index change
    var kappaMax: Double = 0.0          // absorption coefficient: 0 = phase-only (default)
    var saturationIntensity: Float = 0  // I_sat: 0 = linear absorption, >0 = saturable (optical ReLU)
    let voxelPitch: Double = 2e-6      // 2 μm xy resolution
    var dz: Double = 70e-6             // 70 μm slice thickness (Fresnel # ≈ 15 for 128² × 40)

    var k0: Double { 2.0 * .pi / wavelength }

    // MARK: - I/O channels
    var nInputsA: Int = 4      // current token inputs
    var nInputsB: Int = 4      // state vector inputs
    var nOutputsA: Int = 4     // response outputs
    var nOutputsB: Int = 4     // updated state outputs

    var nInputs: Int { nInputsA + nInputsB }
    var nOutputs: Int { nOutputsA + nOutputsB }

    // MARK: - I/O geometry
    var inputXFraction: Float = 0.5    // horizontal position of input spots (0=left, 1=right)
    var outputXFraction: Float = 0.5   // horizontal position of output detectors
    var useDifferentialDetection: Bool = false  // 2 detectors per class (pos-neg)
    var nDetectors: Int { useDifferentialDetection ? nOutputsA * 2 : nOutputsA }
    var usePhaseEncoding: Bool = false  // encode inputs as phase, not amplitude (Ozcan 2024)
    var spotSigmaOverride: Float = 0   // 0 = auto (nx/(4*sqrt(nIn))), >0 = explicit sigma in pixels
    var detectorRadiusOverride: Int = 0 // 0 = auto (2*sigma), >0 = explicit radius in pixels
    var normScale: Float = 10.0        // score amplification before softmax (live-tunable)

    // MARK: - Regression mode (optical FFT, matrix transform, etc.)
    var useRegressionMode: Bool = false // true = MSE regression instead of classification
    var fftInputSize: Int = 64          // size of random input region for FFT experiment

    /// Matrix regression: which transform target to learn
    /// "fft" = 2D DFT (default), "randomUnitary" = random orthogonal matrix,
    /// "hadamard" = Walsh-Hadamard, "identity" = identity (baseline test),
    /// "custom:/path/to/matrix.bin" = load custom N×N Float32 binary
    var regressionTarget: String = "fft"

    // MARK: - Training
    var nSteps: Int = 50_000
    var batchSize: Int = 16
    var lr: Float = 0.003
    var minLR: Float = 0.0003
    var warmupSteps: Int = 200
    var adamBeta1: Float = 0.9
    var adamBeta2: Float = 0.999
    var adamEps: Float = 1e-8
    var sparsityLambda: Float = 5e-5
    var gradClipMax: Float = 10.0     // per-element gradient clipping (0 = disabled)

    // MARK: - Gumbel-softmax annealing
    var initialTemp: Float = 0.5       // Moderate exploration at start
    var annealRate: Float = 1e-4       // reaches 0.25 at ~7K, 0.15 at ~12K
    var minTemp: Float = 0.15          // Don't go too low — prevents gradient collapse

    // MARK: - Adaptive depth growth
    var initialDepth: Int = 40          // start at full depth (strong init keeps glass transparent)
    var depthGrowthPatience: Int = 500  // steps without improvement before growing
    var depthGrowthThreshold: Float = 0.002 // minimum loss improvement
    var depthGrowthIncrement: Int = 2   // slices to add when growing

    // MARK: - Recurrent pulse sequence
    var nPulses: Int = 1                // 1 = single-pass (default), >1 = recurrent
    var thresholdInit: Float = 0.1      // initial threshold for output gating (learnable)
    var feedbackRatioInit: Float = 0.3  // initial mix ratio: output * ratio + input * (1 - ratio)

    // MARK: - EMA weight averaging (Polyak averaging)
    var useEMA: Bool = true            // maintain EMA shadow weights
    var emaDecay: Float = 0.999        // EMA decay factor (0.999 = ~1000-step window)
    var emaSwapInterval: Int = 0       // if > 0, auto-swap EMA→active every N steps (0 = manual only)
    var emaEvalInterval: Int = 500     // evaluate with EMA weights every N steps

    // MARK: - Discrete D2NN mode (UCLA-style thin phase masks)
    var useDiscreteD2NN: Bool = false  // true = discrete phase layers, false = continuous BPM
    var nDiscreteLayers: Int = 5       // number of thin phase mask layers
    var discreteLayerSpacing: Double = 0.0  // physical spacing between layers (m), auto-computed if 0

    // MARK: - Optimization mode
    var useDirect: Bool = false  // true = optimize deltaN directly (1 param/voxel), false = Gumbel-softmax (16 logits/voxel)

    // MARK: - MNIST image input mode (UCLA-style)
    var useImageInput: Bool = false     // true = MNIST images as full-field input
    var mnistClasses: [Int] = [0,1,2,3] // which digit classes to use
    var mnistDataDir: String = ""       // path to MNIST data directory
    var useZeroInit: Bool = false       // true = start from neutral glass (all phases = 0)

    // MARK: - Teacher model
    var useLinearTeacher: Bool = false  // true = linear teacher (4×4 matrix), false = MLP (4→32→4)

    // MARK: - LR schedule
    enum LRSchedule: Int { case cosine = 0, constant = 1 }
    var lrSchedule: LRSchedule = .cosine

    // MARK: - Runtime
    var seed: UInt64 = 42
    var checkpointInterval: Int = 1000
    var maxCheckpoints: Int = 5
    var logEvery: Int = 50
    var uiUpdateInterval: Double = 0.5  // seconds

    // MARK: - Derived
    var fieldSize: Int { nx * ny }
    var totalVoxels: Int { nx * ny * nz }

    func temperature(at step: Int) -> Float {
        max(minTemp, initialTemp * exp(-annealRate * Float(step)))
    }

    /// LR schedule with linear warmup
    func learningRate(at step: Int) -> Float {
        if step < warmupSteps {
            return minLR + (lr - minLR) * Float(step) / Float(warmupSteps)
        }
        switch lrSchedule {
        case .constant:
            return lr
        case .cosine:
            let progress = Float(step - warmupSteps) / Float(max(nSteps - warmupSteps, 1))
            return minLR + 0.5 * (lr - minLR) * (1.0 + cos(Float.pi * progress))
        }
    }

    /// Clamp depth to valid range
    func clampedDepth(_ depth: Int) -> Int {
        min(max(depth, 1), nz)
    }

    /// Auto-scale dz and dnMax for the current grid to achieve target Fresnel number.
    /// Call after setting nx/ny/nz to get physics that actually allows diffraction.
    mutating func autoScalePhysics(fresnelTarget: Double = 15.0) {
        let a = Double(nx) * voxelPitch  // grid physical size
        dz = a * a / (wavelength * fresnelTarget * Double(nz))
        dnMax = 2.0 / (k0 * dz)  // ~2 rad max phase per slice
    }

    /// Auto-scale for discrete D2NN mode: compute layer spacing for target Fresnel number.
    /// Sets nz = nDiscreteLayers and dz = discreteLayerSpacing for propagator.
    /// Phase is the learnable parameter (dnMax = π), no k0*dz conversion needed.
    /// UCLA D2NN used F ≈ 3 per layer for full diffraction mixing between layers.
    mutating func autoScaleDiscrete(fresnelTarget: Double = 3.0) {
        let a = Double(nx) * voxelPitch  // grid physical size
        discreteLayerSpacing = a * a / (wavelength * fresnelTarget)
        nz = nDiscreteLayers
        dz = discreteLayerSpacing  // propagator uses this
        dnMax = .pi                // phase range [-π, π]
        useDirect = true           // direct phase optimization (no Gumbel-softmax)
        kappaMax = 0               // phase-only (no absorption)
    }

    /// Memory estimate in bytes for all GPU buffers
    var estimatedMemoryBytes: Int {
        let persistent = totalVoxels * nLevels * 4 * 3 * 2 // (logits + momentM + momentV) × 2 (dn + kappa)
        let shared = totalVoxels * 4 * 4 + totalVoxels * nLevels * 4 * 2 + fieldSize * 8  // deltaN + gradDeltaN + deltaKappa + gradDeltaKappa + gradLogits×2 + propagator
        let perFrame = batchSize * fieldSize * 8 * 3     // fields + adjoint + fftTemp
            + batchSize * (nz + 1) * fieldSize * 8       // fieldCache
            + batchSize * totalVoxels * 4 * 2            // perBatchGrad + perBatchGradKappa
        return persistent + shared + perFrame * 3         // triple buffered
    }

    var estimatedMemoryMB: Double {
        Double(estimatedMemoryBytes) / 1_048_576.0
    }
}

// MARK: - Presets

extension SimConfig {
    static var quick: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64; c.nz = 20
        c.nSteps = 30_000; c.batchSize = 64
        c.lr = 0.003; c.minLR = 0.0003
        c.useDirect = false               // Gumbel-softmax: 16 logits/voxel
        c.useLinearTeacher = true
        c.initialTemp = 1.0
        c.minTemp = 1.0
        c.annealRate = 0.0
        c.initialDepth = 20
        c.warmupSteps = 200
        c.autoScalePhysics()  // 64×64×20: dz≈35μm, dnMax≈0.014
        return c
    }

    static var medium: SimConfig {
        var c = SimConfig()
        // 128×128×40 = 655K voxels, 10.5M logits
        c.nSteps = 30_000
        c.batchSize = 32
        c.lr = 0.003; c.minLR = 0.0003
        c.useDirect = false               // Gumbel-softmax: essential at larger grids
        c.useLinearTeacher = true
        c.initialTemp = 1.0; c.minTemp = 1.0
        c.annealRate = 0.0
        c.initialDepth = 40
        c.warmupSteps = 200
        c.autoScalePhysics()  // 128×128×40: dz≈70μm, dnMax≈0.007
        return c
    }

    static var large: SimConfig {
        var c = SimConfig()
        c.nx = 256; c.ny = 256; c.nz = 40
        c.nSteps = 50_000; c.batchSize = 16
        c.lr = 0.002; c.annealRate = 1e-4
        c.autoScalePhysics()  // dz≈282μm, dnMax≈0.00175
        return c
    }

    static var maxQuality: SimConfig {
        var c = SimConfig()
        c.nx = 512; c.ny = 512; c.nz = 40
        c.nSteps = 30_000; c.batchSize = 8
        c.lr = 0.001; c.annealRate = 1e-4
        c.autoScalePhysics()  // dz≈1127μm, dnMax≈0.00044
        return c
    }

    // MARK: - Experiment presets

    /// Saturable absorption with MLP teacher: nonlinear optical ReLU
    /// Tests whether saturable absorption (nonlinear element) enables classification
    /// of MLP teacher targets that phase-only glass cannot implement.
    /// Key experiment: can optical nonlinearity break the linear barrier?
    static var saturableAbsorption: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32; c.nz = 10
        c.autoScalePhysics()
        c.kappaMax = 0.083 / (c.k0 * c.dz)    // ~8% max attenuation per voxel
        c.saturationIntensity = 0.5              // I_sat for nonlinear absorption
        c.nSteps = 30_000; c.batchSize = 128
        c.lr = 0.003; c.minLR = 0.0003
        c.useDirect = true                       // direct deltaN (phase); kappa still uses Gumbel
        c.useLinearTeacher = false                // MLP teacher: nonlinear targets!
        c.initialTemp = 1.0; c.minTemp = 1.0
        c.annealRate = 0.0
        c.warmupSteps = 200
        c.initialDepth = 10
        return c
    }

    /// Recurrent pulse experiment on 32×32×10: 3 passes with threshold feedback
    static var recurrent3Pulse: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32; c.nz = 10
        c.autoScalePhysics()
        c.nPulses = 3
        c.thresholdInit = 0.1
        c.feedbackRatioInit = 0.3
        c.nSteps = 30_000; c.batchSize = 128
        c.lr = 0.003; c.minLR = 0.0003
        c.useDirect = true
        c.useLinearTeacher = true    // linear teacher
        c.initialTemp = 1.0; c.minTemp = 1.0
        c.annealRate = 0.0
        c.warmupSteps = 200
        c.initialDepth = 10
        return c
    }

    /// Combined: saturable absorption + 3 recurrent pulses on 32×32×10
    static var recurrentSaturable: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32; c.nz = 10
        c.autoScalePhysics()
        c.kappaMax = 0.083 / (c.k0 * c.dz)
        c.saturationIntensity = 0.5
        c.nPulses = 3
        c.thresholdInit = 0.1
        c.feedbackRatioInit = 0.3
        c.nSteps = 30_000; c.batchSize = 128
        c.lr = 0.003; c.minLR = 0.0003
        c.useDirect = true
        c.useLinearTeacher = false   // MLP teacher: test if combined nonlinearities break linear barrier
        c.initialTemp = 1.0; c.minTemp = 1.0
        c.annealRate = 0.0
        c.warmupSteps = 200
        c.initialDepth = 10
        return c
    }

    // MARK: - D2NN-inspired presets (separated I/O + differential detection)

    /// D2NN debug: 32×32×10 with separated I/O for fast iteration
    static var d2nnDebug: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32; c.nz = 10
        c.nSteps = 30_000; c.batchSize = 128
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .constant
        c.useDirect = false
        c.useLinearTeacher = true
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.initialDepth = 10; c.warmupSteps = 200
        c.inputXFraction = 0.2
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true  // Phase encoding (Ozcan 2024)
        c.spotSigmaOverride = 2.0   // Small spots → fast diffraction → light reaches detectors
        c.detectorRadiusOverride = 4
        c.autoScalePhysics()
        return c
    }

    /// D2NN: 64×64×20 with separated I/O
    static var d2nn: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64; c.nz = 20
        c.nSteps = 30_000; c.batchSize = 64
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .constant
        c.useDirect = false
        c.useLinearTeacher = true
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.initialDepth = 20; c.warmupSteps = 200
        c.inputXFraction = 0.2
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true  // Phase encoding (Ozcan 2024)
        c.spotSigmaOverride = 2.0   // Small spots → fast diffraction → light reaches detectors
        c.detectorRadiusOverride = 6
        c.autoScalePhysics()
        return c
    }

    /// D2NN + differential detection: 64×64×20, separated I/O, 2 detectors per class
    static var d2nnDifferential: SimConfig {
        var c = d2nn
        c.useDifferentialDetection = true
        c.usePhaseEncoding = true  // Phase encoding (Ozcan 2024)
        return c
    }

    /// D2NN deep: 64×64×40, separated I/O + differential — 2× more depth for better routing
    static var d2nnDeep: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64; c.nz = 40
        c.nSteps = 30_000; c.batchSize = 64
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .constant
        c.useDirect = false
        c.useLinearTeacher = true
        c.useDifferentialDetection = true
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.initialDepth = 40; c.warmupSteps = 200
        c.inputXFraction = 0.2
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true  // Phase encoding (Ozcan 2024)
        c.spotSigmaOverride = 2.0
        c.detectorRadiusOverride = 6
        c.autoScalePhysics()
        return c
    }

    /// D2NN 128: 128×128×40, separated I/O + differential — full scale
    static var d2nn128: SimConfig {
        var c = SimConfig()
        c.nx = 128; c.ny = 128; c.nz = 40
        c.nSteps = 60_000; c.batchSize = 32
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .constant
        c.useDirect = false
        c.useLinearTeacher = true
        c.useDifferentialDetection = true
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.initialDepth = 40; c.warmupSteps = 200
        c.inputXFraction = 0.2
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true  // Phase encoding (Ozcan 2024)
        c.spotSigmaOverride = 2.0   // Small spots: z_R=47μm, fills 94% of grid
        c.detectorRadiusOverride = 8
        c.autoScalePhysics()
        return c
    }

    /// UCLA D2NN replication: continuous phase, separated I/O, differential detection
    /// Mimics Lin/Ozcan Science 2018 setup adapted to BPM.
    /// Key: NO Gumbel-softmax — direct continuous deltaN optimization.
    /// Small input spots (sigma=2px) so light diffracts to fill entire volume.
    static var uclaReplication: SimConfig {
        var c = SimConfig()
        c.nx = 128; c.ny = 128; c.nz = 20  // 20 layers (vs UCLA's 5 thick layers)
        c.nSteps = 60_000; c.batchSize = 32
        c.lr = 0.001; c.minLR = 0.0001      // lower lr for direct mode stability
        c.lrSchedule = .constant
        c.useDirect = true                    // KEY: continuous deltaN, no quantization
        c.useLinearTeacher = true
        c.useDifferentialDetection = true     // UCLA 2019 differential detection
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.initialDepth = 20; c.warmupSteps = 500
        c.inputXFraction = 0.2               // separated I/O (UCLA style)
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true             // phase encoding
        c.gradClipMax = 1.0                   // aggressive clipping for direct mode at scale
        c.spotSigmaOverride = 2.0            // Small spots → light fills grid via diffraction
        c.detectorRadiusOverride = 8
        c.useEMA = true; c.emaDecay = 0.999
        c.autoScalePhysics()
        return c
    }

    /// UCLA D2NN replication small: 64×64×20, faster iteration
    static var uclaReplicationSmall: SimConfig {
        var c = uclaReplication
        c.nx = 64; c.ny = 64
        c.batchSize = 64
        c.nSteps = 30_000
        c.spotSigmaOverride = 2.0
        c.detectorRadiusOverride = 6
        c.autoScalePhysics()
        return c
    }

    /// Tiny debug preset: small grid for rapid convergence testing
    /// 32×32×10 = 10K voxels (vs 655K for medium).
    /// Uses linear teacher (4×4 matrix) — achievable by linear photonic system.
    static var debug: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32; c.nz = 10
        c.nSteps = 30_000; c.batchSize = 128
        c.lr = 0.003; c.minLR = 0.0003    // direct mode: lower lr for stability
        c.useDirect = true                 // direct deltaN optimization
        c.initialTemp = 1.0               // unused in direct mode
        c.minTemp = 1.0
        c.annealRate = 0.0
        c.useLinearTeacher = true  // linear teacher for 4-class
        c.initialDepth = 10
        c.warmupSteps = 200
        c.autoScalePhysics()
        return c
    }

    // MARK: - Discrete D2NN presets (UCLA-style thin phase masks)

    /// Discrete D2NN: 128×128, 5 layers, UCLA-style architecture
    /// Key: discrete phase masks with free-space propagation between them.
    /// Learnable parameter is phase φ ∈ [-π, π] per pixel per layer.
    /// Should match or exceed UCLA's 91.75% (10-class) ≈ ~98% (4-class).
    static var discreteD2NN: SimConfig {
        var c = SimConfig()
        c.nx = 128; c.ny = 128
        c.nDiscreteLayers = 5
        c.useDiscreteD2NN = true
        c.nSteps = 60_000; c.batchSize = 32
        c.lr = 0.001; c.minLR = 0.0001
        c.lrSchedule = .cosine
        c.useLinearTeacher = true
        c.useDifferentialDetection = true
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.warmupSteps = 500
        c.inputXFraction = 0.2
        c.outputXFraction = 0.8
        c.usePhaseEncoding = true
        c.gradClipMax = 0.1
        c.spotSigmaOverride = 2.0
        c.detectorRadiusOverride = 8
        c.useEMA = true; c.emaDecay = 0.999
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// Discrete D2NN small: 64×64, 5 layers, faster iteration
    static var discreteD2NNSmall: SimConfig {
        var c = discreteD2NN
        c.nx = 64; c.ny = 64
        c.batchSize = 64
        c.nSteps = 30_000
        c.spotSigmaOverride = 2.0
        c.detectorRadiusOverride = 4
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// Discrete D2NN large: 200×200, 5 layers, closest to UCLA's original 200×200 grid
    static var discreteD2NNLarge: SimConfig {
        var c = discreteD2NN
        c.nx = 256; c.ny = 256  // 256 (power of 2, close to 200)
        c.batchSize = 16
        c.nSteps = 60_000
        c.spotSigmaOverride = 3.0
        c.detectorRadiusOverride = 12
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    // MARK: - MNIST D2NN presets (UCLA-style with real image input)

    /// MNIST D2NN: 64×64, 5 layers, MNIST 4-class image input, UCLA-style
    /// Grid detector layout across full output plane, no differential (simpler, more area per class)
    static var mnistD2NN: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64
        c.nDiscreteLayers = 5
        c.useDiscreteD2NN = true
        c.useImageInput = true
        c.mnistClasses = [0, 1, 2, 3]
        c.mnistDataDir = NSHomeDirectory() + "/Documents/photonic-twin/data/mnist"
        c.nSteps = 50_000; c.batchSize = 64
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .cosine
        c.useLinearTeacher = true  // unused but set for compat
        c.useDifferentialDetection = false  // 4 standard detectors (larger area each)
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.warmupSteps = 200
        c.inputXFraction = 0.5   // unused for image mode
        c.outputXFraction = 0.5  // unused for image mode (grid layout)
        c.nOutputsA = c.mnistClasses.count
        c.usePhaseEncoding = false  // amplitude input (UCLA default)
        c.gradClipMax = 0.0
        c.spotSigmaOverride = 2.0
        // Detector radius ~10% of grid size. UCLA used ~20 at 200×200.
        c.detectorRadiusOverride = 8
        c.useEMA = true; c.emaDecay = 0.999
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// MNIST D2NN small: 32×32, 5 layers, fastest iteration
    static var mnistD2NNSmall: SimConfig {
        var c = mnistD2NN
        c.nx = 32; c.ny = 32
        c.batchSize = 128
        c.nSteps = 30_000
        c.detectorRadiusOverride = 4  // proportional: 32*0.125
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// MNIST D2NN large: 128×128, 5 layers, higher resolution
    static var mnistD2NNLarge: SimConfig {
        var c = mnistD2NN
        c.nx = 128; c.ny = 128
        c.batchSize = 32
        c.nSteps = 60_000
        c.detectorRadiusOverride = 16  // proportional: 128*0.125
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// MNIST 10-class: 128×128, 5 layers, full 10-digit MNIST (UCLA comparison)
    static var mnist10Class: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.detectorRadiusOverride = 12  // smaller to avoid overlap in 4×3 grid
        c.normScale = 0.1  // 10-class raw scores ~100-300, need low normScale to avoid softmax saturation
        c.nSteps = 100_000
        return c
    }

    /// MNIST 10-class small: 64×64, 5 layers, full 10-digit MNIST (fast iteration)
    static var mnist10ClassSmall: SimConfig {
        var c = mnistD2NN
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.detectorRadiusOverride = 6  // proportional to 64×64 grid
        c.normScale = 0.1
        c.nSteps = 100_000
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        return c
    }

    /// MNIST 10-class differential on small grid: 64×64, 5 layers, 20 detectors (r=4)
    static var mnist10ClassSmallDiff: SimConfig {
        var c = mnistD2NN
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.useDifferentialDetection = true  // 20 detectors
        c.detectorRadiusOverride = 4  // small to fit 5×4 grid on 64×64
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.gradClipMax = 1.0
        c.nSteps = 100_000
        return c
    }

    /// MNIST 10-class with phase encoding: 128×128, 5 layers
    /// Phase encoding: input pixel value → phase angle (unit amplitude)
    /// Literature suggests this gives the phase screens more to work with
    static var mnist10ClassPhase: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.detectorRadiusOverride = 12
        c.usePhaseEncoding = true  // KEY: phase encoding
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.nSteps = 100_000
        return c
    }

    /// MNIST 10-class with larger detectors: 128×128, 5 layers
    /// Larger detector radius captures more signal
    static var mnist10ClassLargeDet: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.detectorRadiusOverride = 20  // much larger detectors
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.nSteps = 100_000
        return c
    }

    /// MNIST 10-class optimized: 128×128, 5 layers, differential detection, batch=128
    /// Based on literature: SCE loss + differential + batch 128 → 97-98%
    static var mnist10ClassDiff: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.useDifferentialDetection = true  // 20 detectors (2 per class)
        c.detectorRadiusOverride = 8  // smaller for 5×4 grid of 20 detectors
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.gradClipMax = 1.0
        c.nSteps = 100_000
        return c
    }

    /// MNIST 10-class optimized, no differential: 128×128, batch=128
    /// Control experiment: same as mnist10ClassDiff but without differential detection
    static var mnist10ClassBatch128: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.detectorRadiusOverride = 12
        c.batchSize = 128
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.nSteps = 100_000
        return c
    }

    // MARK: - Optical FFT glass block presets

    /// Optical FFT: optimize glass to compute 2D DFT of complex input fields.
    /// 128×128 grid, 15 initial slices (progressive growth to 40), complex refractive index.
    /// Training: random complex 64×64 inputs → exact 2D FFT targets (MSE loss).
    static var opticalFFT: SimConfig {
        var c = SimConfig()
        c.nx = 128; c.ny = 128
        c.nDiscreteLayers = 40     // max depth for progressive growth
        c.useRegressionMode = true
        c.fftInputSize = 64
        c.useDiscreteD2NN = true
        c.batchSize = 16
        c.lr = 0.001
        c.minLR = 0.0001
        c.lrSchedule = .cosine
        c.warmupSteps = 500
        c.nSteps = 200_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0        // no gradient clipping; let Adam handle scaling
        c.logEvery = 50
        c.checkpointInterval = 5000
        c.initialDepth = 15        // start with 15 slices, grow to 40
        c.depthGrowthPatience = 2000
        c.depthGrowthThreshold = 0.001
        c.autoScaleDiscrete()  // sets nz=40, useDirect=true, kappaMax=0
        // Override after autoScaleDiscrete (which forces kappaMax=0, phase-only)
        c.kappaMax = 0.01          // enable absorption (complex refractive index)
        c.saturationIntensity = 5.0 // saturable absorption (optical nonlinearity)
        return c
    }

    /// Optical FFT small: 64×64 grid, faster iteration for debugging.
    static var opticalFFTSmall: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64
        c.nDiscreteLayers = 20
        c.useRegressionMode = true
        c.fftInputSize = 32
        c.useDiscreteD2NN = true
        c.batchSize = 32
        c.lr = 0.003
        c.minLR = 0.0003
        c.lrSchedule = .cosine
        c.warmupSteps = 300
        c.nSteps = 100_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0        // no gradient clipping
        c.logEvery = 50
        c.initialDepth = 10
        c.depthGrowthPatience = 1500
        c.depthGrowthThreshold = 0.001
        c.autoScaleDiscrete()
        c.kappaMax = 0.01
        c.saturationIntensity = 5.0
        return c
    }

    /// Optical FFT tiny deep: 16×16 grid, 64 layers.
    /// MPLC scaling law: N modes need ~2N-3N planes. 16×16=256 modes → 64 planes.
    static var opticalFFTTiny: SimConfig {
        var c = SimConfig()
        c.nx = 16; c.ny = 16
        c.nDiscreteLayers = 64
        c.useRegressionMode = true
        c.fftInputSize = 16      // full grid input (no zero-padding)
        c.useDiscreteD2NN = true
        c.batchSize = 128        // large batch for better gradient SNR
        c.lr = 0.003
        c.minLR = 0.0003
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 100_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 20
        c.autoScaleDiscrete()
        c.initialDepth = c.nz    // use ALL layers from the start (no progressive growth)
        c.kappaMax = 0.0         // phase-only for simplicity
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT medium: 32×32 grid, 96 layers.
    /// MPLC: 32×32=1024 modes → 96 planes (~3N).
    static var opticalFFTMedium: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32
        c.nDiscreteLayers = 96
        c.useRegressionMode = true
        c.fftInputSize = 32      // full grid input
        c.useDiscreteD2NN = true
        c.batchSize = 64
        c.lr = 0.003
        c.minLR = 0.0003
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 200_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 50
        c.autoScaleDiscrete()
        c.initialDepth = c.nz    // use ALL layers from the start
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT micro: 4×4 grid, 32 layers.
    /// Proof of concept: 16 modes → 32 layers = 2× MPLC minimum.
    /// DFT structure: 2D FFT = 2+2 = 4 butterfly stages → should converge fast.
    static var opticalFFT4x4: SimConfig {
        var c = SimConfig()
        c.nx = 4; c.ny = 4
        c.nDiscreteLayers = 32
        c.useRegressionMode = true
        c.fftInputSize = 4
        c.useDiscreteD2NN = true
        c.batchSize = 256
        c.lr = 0.003
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 50_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT 8×8: 8×8 grid, 128 layers.
    /// 64 modes → 128 layers = 2× MPLC minimum.
    static var opticalFFT8x8: SimConfig {
        var c = SimConfig()
        c.nx = 8; c.ny = 8
        c.nDiscreteLayers = 128
        c.useRegressionMode = true
        c.fftInputSize = 8
        c.useDiscreteD2NN = true
        c.batchSize = 256
        c.lr = 0.003
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 100_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT 16×16 extra-deep: 256 layers.
    /// 256 modes → 256 layers = 1× MPLC minimum. 4× deeper than opticalFFTTiny (64L).
    /// Tests whether more layers can break through the 77% MSE reduction plateau.
    static var opticalFFTTinyDeep: SimConfig {
        var c = SimConfig()
        c.nx = 16; c.ny = 16
        c.nDiscreteLayers = 256
        c.useRegressionMode = true
        c.fftInputSize = 16
        c.useDiscreteD2NN = true
        c.batchSize = 64
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 200_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 20
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT 16×16 ultra-deep: 512 layers.
    /// 256 modes → 512 layers = 2× MPLC minimum (matches 4×4 and 8×8 convergence ratio).
    /// Expected to reach 70+ dB like smaller grids.
    static var opticalFFTTinyUltra: SimConfig {
        var c = SimConfig()
        c.nx = 16; c.ny = 16
        c.nDiscreteLayers = 512
        c.useRegressionMode = true
        c.fftInputSize = 16
        c.useDiscreteD2NN = true
        c.batchSize = 32
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 300_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 20
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Optical FFT 32×32 deep: 32×32 grid, 512 layers.
    /// 1024 modes → 512 layers = 0.5× MPLC minimum. Tests whether DFT structure allows fewer layers.
    static var opticalFFT32x32Deep: SimConfig {
        var c = SimConfig()
        c.nx = 32; c.ny = 32
        c.nDiscreteLayers = 512
        c.useRegressionMode = true
        c.fftInputSize = 32
        c.useDiscreteD2NN = true
        c.batchSize = 32
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 300_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 50
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    // MARK: - Matrix regression presets (arbitrary unitary transforms)

    /// Matrix regression 8×8: train glass to implement a random 64×64 unitary matrix.
    /// Proves glass can implement ANY unitary, not just FFT.
    /// 64 modes × 2 = 128 layers (2× MPLC ratio, same as FFT proof).
    static var matrixUnitary8x8: SimConfig {
        var c = SimConfig()
        c.nx = 8; c.ny = 8
        c.nDiscreteLayers = 128
        c.useRegressionMode = true
        c.regressionTarget = "randomUnitary"
        c.fftInputSize = 8       // full grid
        c.useDiscreteD2NN = true
        c.batchSize = 256
        c.lr = 0.003
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 100_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// Matrix regression: Hadamard 8×8 (128 layers).
    /// Hadamard is a specific structured unitary — should be easier than random.
    static var matrixHadamard8x8: SimConfig {
        var c = matrixUnitary8x8
        c.regressionTarget = "hadamard"
        return c
    }

    /// Matrix regression: identity 8×8 (128 layers).
    /// Baseline test — glass should learn to be transparent (trivial).
    static var matrixIdentity8x8: SimConfig {
        var c = matrixUnitary8x8
        c.regressionTarget = "identity"
        return c
    }

    /// 16×16 random unitary matrix: 256 modes × 2 = 512 layers (2× MPLC ratio)
    static var matrixUnitary16x16: SimConfig {
        var c = SimConfig()
        c.nx = 16; c.ny = 16
        c.nDiscreteLayers = 512
        c.useRegressionMode = true
        c.regressionTarget = "randomUnitary"
        c.fftInputSize = 16
        c.useDiscreteD2NN = true
        c.batchSize = 32
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 100_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// 16×16 Hadamard matrix: 256 modes × 2 = 512 layers
    static var matrixHadamard16x16: SimConfig {
        var c = matrixUnitary16x16
        c.regressionTarget = "hadamard"
        return c
    }

    /// TinyStories-1M layer6 W_V nearest unitary: 64×64 → 8×8 grid, 128 layers
    /// Best candidate (cond=88.45, most unitary-like in the model)
    static var tinyStoriesWV: SimConfig {
        var c = matrixUnitary8x8
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_V_unitary.bin"
        c.regressionTarget = "custom:\(path)"
        c.nSteps = 200_000
        return c
    }

    /// TinyStories-1M layer6 W_V RAW (non-unitary, uses amplitude control)
    /// kappaMax > 0 enables absorption → glass can learn non-unitary transforms
    static var tinyStoriesWVRaw: SimConfig {
        var c = matrixUnitary8x8
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_V_raw.bin"
        c.regressionTarget = "custom:\(path)"
        c.kappaMax = 0.1  // Enable amplitude modulation for non-unitary
        c.nSteps = 200_000
        return c
    }

    /// TinyStories-1M layer6 W_Q nearest unitary (cond=2477.74)
    static var tinyStoriesWQ: SimConfig {
        var c = matrixUnitary8x8
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_Q_unitary.bin"
        c.regressionTarget = "custom:\(path)"
        c.nSteps = 200_000
        return c
    }

    /// TinyStories-1M layer6 W_K nearest unitary (cond=1561.47)
    static var tinyStoriesWK: SimConfig {
        var c = matrixUnitary8x8
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_K_unitary.bin"
        c.regressionTarget = "custom:\(path)"
        c.nSteps = 200_000
        return c
    }

    /// TinyStories-1M layer6 W_O nearest unitary (cond=1218.97)
    static var tinyStoriesWO: SimConfig {
        var c = matrixUnitary8x8
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_O_unitary.bin"
        c.regressionTarget = "custom:\(path)"
        c.nSteps = 200_000
        return c
    }

    // MARK: - 64×64 glass block presets (proving LLM-scale blocks)

    /// 64×64 random unitary with 4096 layers (1× MPLC ratio).
    /// Tests if 1× ratio gives ≥50dB PSNR (sufficient for LLM inference).
    static var matrixUnitary64x64: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64
        c.nDiscreteLayers = 4096   // 1× ratio
        c.useRegressionMode = true
        c.regressionTarget = "randomUnitary"
        c.fftInputSize = 64
        c.useDiscreteD2NN = true
        c.batchSize = 8            // small batch for memory
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 50_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    /// 64×64 actual TinyStories W_V unitary (best-conditioned matrix).
    /// Uses layer6 W_V nearest unitary at full 64×64 scale.
    static var matrixTinyStories64x64: SimConfig {
        var c = matrixUnitary64x64
        let path = NSHomeDirectory() + "/Documents/photonic-twin/data/weights/layer6_W_V_unitary.bin"
        c.regressionTarget = "custom:\(path)"
        return c
    }

    /// 64×64 random unitary with 2048 layers (0.5× MPLC ratio).
    /// Tests minimum viable depth for 64×64 blocks.
    static var matrixUnitary64x64Shallow: SimConfig {
        var c = matrixUnitary64x64
        c.nDiscreteLayers = 2048
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        return c
    }

    /// 64×64 random unitary with 512 layers (minimal test).
    /// Proves the engine handles 64×64 grids. PSNR will be limited but nonzero.
    static var matrixUnitary64x64Tiny: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64
        c.nDiscreteLayers = 512    // 0.125× ratio — minimal
        c.useRegressionMode = true
        c.regressionTarget = "randomUnitary"
        c.fftInputSize = 64
        c.useDiscreteD2NN = true
        c.batchSize = 1            // minimal memory
        c.lr = 0.001
        c.lrSchedule = .constant
        c.warmupSteps = 0
        c.nSteps = 50_000
        c.useEMA = true; c.emaDecay = 0.999
        c.gradClipMax = 0.0
        c.logEvery = 10
        c.autoScaleDiscrete()
        c.initialDepth = c.nz
        c.kappaMax = 0.0
        c.saturationIntensity = 0.0
        return c
    }

    // MARK: - Fashion-MNIST presets (harder than MNIST)

    /// Fashion-MNIST 10-class with phase encoding: 128×128, 5 layers.
    /// Same format as MNIST but much harder (clothing items vs digits).
    /// Typical D2NN accuracy: ~85-90% vs 99%+ on MNIST.
    static var fashion10ClassPhase: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.mnistDataDir = NSHomeDirectory() + "/Documents/photonic-twin/data/fashion-mnist"
        c.detectorRadiusOverride = 12
        c.usePhaseEncoding = true
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.nSteps = 100_000
        return c
    }

    /// Fashion-MNIST 10-class standard (amplitude encoding): 128×128, 5 layers.
    static var fashion10Class: SimConfig {
        var c = mnistD2NNLarge
        c.mnistClasses = [0,1,2,3,4,5,6,7,8,9]
        c.nOutputsA = 10
        c.mnistDataDir = NSHomeDirectory() + "/Documents/photonic-twin/data/fashion-mnist"
        c.detectorRadiusOverride = 12
        c.batchSize = 32
        c.lr = 0.003
        c.lrSchedule = .constant
        c.normScale = 0.1
        c.nSteps = 100_000
        return c
    }

    // MARK: - Debug presets (fundamental validation)

    /// MNIST 2-class debug: simplest possible task (0 vs 1), zero init, 64×64
    /// Use this to validate: forward model, gradients, loss function.
    /// If this doesn't converge to >90%, something fundamental is broken.
    /// UCLA D2NN used Fresnel number ≈ 3 per layer spacing — critical for diffraction mixing.
    static var mnist2ClassDebug: SimConfig {
        var c = SimConfig()
        c.nx = 64; c.ny = 64
        c.nDiscreteLayers = 5
        c.useDiscreteD2NN = true
        c.useImageInput = true
        c.useZeroInit = true  // neutral glass start
        c.mnistClasses = [0, 1]  // just 0 vs 1
        c.mnistDataDir = NSHomeDirectory() + "/Documents/photonic-twin/data/mnist"
        c.nSteps = 30_000; c.batchSize = 32
        c.lr = 0.003; c.minLR = 0.0003
        c.lrSchedule = .constant  // constant lr works better (cosine gets stuck)
        c.useLinearTeacher = true
        c.useDifferentialDetection = false  // 2 standard detectors
        c.initialTemp = 1.0; c.minTemp = 1.0; c.annealRate = 0.0
        c.warmupSteps = 100
        c.inputXFraction = 0.5
        c.outputXFraction = 0.5
        c.nOutputsA = 2
        c.usePhaseEncoding = false
        c.gradClipMax = 0.0
        c.spotSigmaOverride = 2.0
        c.detectorRadiusOverride = 10  // large detectors for clear signal
        c.useEMA = false  // keep it simple
        c.normScale = 1.0  // no amplification — let gradients flow cleanly
        c.autoScaleDiscrete(fresnelTarget: 3.0)  // UCLA-matched F≈3 for full diffraction mixing
        c.initialDepth = c.nz
        return c
    }

    /// Same as 2-class debug but with random init (compare with zero init)
    static var mnist2ClassRandom: SimConfig {
        var c = mnist2ClassDebug
        c.useZeroInit = false  // random init for comparison
        return c
    }

    /// 1-layer gradient check test — simplest possible D2NN
    static var mnist2Class1Layer: SimConfig {
        var c = mnist2ClassDebug
        c.nDiscreteLayers = 1
        c.autoScaleDiscrete(fresnelTarget: 3.0)
        return c
    }
}
