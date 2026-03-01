// Photonic Neural Compute — Metal GPU BPM Digital Twin
// SNRGY Studios — February 2026
//
// Main entry point: sets up Metal, compiles shaders, wires up all components,
// and runs the BPM optimization loop to train a 3D glass refractive index
// pattern that optically performs neural network inference.
//
// Architecture:
//   Config.swift       — All simulation parameters
//   VoxelGrid.swift    — 4-bit voxel grid with Gumbel-softmax logits
//   BPMSimulator.swift — Forward/backward BPM dispatch + FFT
//   IOEncoder.swift    — Spatial light input encoding + detector readout
//   StateManager.swift — Recurrent state vector management (BPTT)
//   ReferenceModel.swift — MLP teacher model for training data
//   Shaders.metal      — All GPU compute kernels
//
// Usage:
//   swiftc -O -o photonic_twin \
//     Config.swift VoxelGrid.swift BPMSimulator.swift IOEncoder.swift \
//     StateManager.swift ReferenceModel.swift PhotonicTwin.swift \
//     -framework Metal -framework Accelerate
//   ./photonic_twin [--nx 64] [--ny 64] [--nz 40] [--steps 3000]
//
// Requires: macOS with Apple Silicon (M1/M2/M3/M4/M5)

import Foundation
import Metal
import Accelerate

// ============================================================
// Photonic Twin Engine
// ============================================================

class PhotonicTwinEngine {
    let config: SimConfig
    let device: MTLDevice
    let commandQueue: MTLCommandQueue

    // Subsystems
    let grid: VoxelGrid
    let bpm: BPMSimulator
    let ioEncoder: IOEncoder
    let stateManager: StateManager
    let refModel: ReferenceModel

    // Training state
    var rng: SeededRNG
    var currentTemp: Float

    // MARK: - Initialization

    init(config: SimConfig) throws {
        self.config = config
        self.rng = SeededRNG(seed: config.seed)
        self.currentTemp = config.initialTemp

        // --- Metal Device ---
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device found. Requires Apple Silicon Mac.")
        }
        self.device = dev
        guard let queue = dev.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue

        print("Metal device: \(dev.name)")
        print("Max buffer length: \(dev.maxBufferLength / 1_048_576) MB")

        // --- Subsystems ---
        grid = VoxelGrid(device: dev, config: config)
        bpm = BPMSimulator(device: dev, commandQueue: queue, grid: grid, config: config)
        ioEncoder = IOEncoder(config: config)
        stateManager = StateManager(config: config)
        refModel = ReferenceModel(config: config)

        // --- Compile Metal Shaders ---
        try compileShaders()

        // Print configuration
        config.printSummary()
        grid.printInfo()
        refModel.printInfo()
    }

    // MARK: - Shader Compilation

    private func compileShaders() throws {
        // Look for Shaders.metal in the same directory as the executable, or CWD
        let shaderPaths = [
            URL(fileURLWithPath: CommandLine.arguments[0])
                .deletingLastPathComponent()
                .appendingPathComponent("Shaders.metal").path,
            "Shaders.metal",
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Shaders.metal").path
        ]

        var shaderSource: String?
        for path in shaderPaths {
            if let source = try? String(contentsOfFile: path, encoding: .utf8) {
                shaderSource = source
                print("Loaded shaders from: \(path)")
                break
            }
        }

        guard let source = shaderSource else {
            fatalError("Could not find Shaders.metal. Place it next to the executable or in the current directory.")
        }

        // Function constants — baked into the GPU pipelines at compile time
        let constants = MTLFunctionConstantValues()
        var nx = UInt32(config.nx)
        var ny = UInt32(config.ny)
        var nz = UInt32(config.nz)
        var k0 = config.k0
        var dz = config.dz
        var dnMax = config.dnMax
        var nLevels = UInt32(config.nLevels)

        constants.setConstantValue(&nx,      type: .uint,  index: 0)  // NX
        constants.setConstantValue(&ny,      type: .uint,  index: 1)  // NY
        constants.setConstantValue(&nz,      type: .uint,  index: 2)  // NZ
        constants.setConstantValue(&k0,      type: .float, index: 3)  // K0
        constants.setConstantValue(&dz,      type: .float, index: 4)  // DZ_CONST
        constants.setConstantValue(&dnMax,   type: .float, index: 5)  // DN_MAX
        constants.setConstantValue(&nLevels, type: .uint,  index: 6)  // N_LEVELS

        let compileOptions = MTLCompileOptions()
        let library = try device.makeLibrary(source: source, options: compileOptions)

        try bpm.compilePipelines(library: library, constants: constants)
        print("Metal shader pipelines compiled successfully")
    }

    // MARK: - Training Loop

    func train() {
        let startTime = CFAbsoluteTimeGetCurrent()

        print("\n============================================================")
        print("TRAINING")
        print("============================================================")
        print("Optimizer:   Adam (lr=\(config.lr), beta1=\(config.adam_beta1), beta2=\(config.adam_beta2))")
        print("Steps:       \(config.nSteps)")
        print("Batch size:  \(config.batchSize)")
        print("BPTT steps:  \(config.bpttSteps)")
        print("Temperature: \(config.initialTemp) → \(config.minTemp) (anneal rate \(config.annealRate))")
        print("Sparsity:    lambda=\(config.sparsityLambda)")
        print("------------------------------------------------------------")

        for step in 0..<config.nSteps {
            // Anneal Gumbel-softmax temperature
            currentTemp = max(
                config.minTemp,
                config.initialTemp * exp(-config.annealRate * Float(step))
            )

            // Compute continuous delta_n from logits via Gumbel-softmax
            grid.computeDeltaN(temperature: currentTemp, addNoise: true)

            // Zero all gradients
            bpm.zeroGradients()

            // Generate training batch
            let bptt = config.bpttSteps
            let (batchInputs, batchTargets): ([[Float]], [Int])
            if bptt <= 1 {
                (batchInputs, batchTargets) = refModel.generateBatch(
                    size: config.batchSize, rng: &rng
                )
            } else {
                // For BPTT > 1, generate sequences and use last step's target
                let (seqInputs, seqTargets) = refModel.generateSequenceBatch(
                    batchSize: config.batchSize, seqLen: bptt, rng: &rng
                )
                // Flatten: we'll process sequences individually below
                // For now, use single-step training path
                var flatInputs = [[Float]]()
                var flatTargets = [Int]()
                for b in 0..<config.batchSize {
                    flatInputs.append(seqInputs[b][bptt - 1])
                    flatTargets.append(seqTargets[b][bptt - 1])
                }
                (batchInputs, batchTargets) = (flatInputs, flatTargets)
            }

            var stepLoss: Float = 0
            var stepCorrect = 0

            // Process each sample in the mini-batch
            for b in 0..<config.batchSize {
                if bptt <= 1 {
                    // --- Single-step training (no recurrence) ---
                    let (loss, correct) = trainSingleStep(
                        inputA: batchInputs[b],
                        targetClassA: batchTargets[b]
                    )
                    stepLoss += loss
                    if correct { stepCorrect += 1 }
                } else {
                    // --- Multi-step BPTT training ---
                    // TODO: Full BPTT implementation when bpttSteps > 1
                    // For now, fall back to single-step with state reset
                    stateManager.resetSequence()
                    let (loss, correct) = trainSingleStep(
                        inputA: batchInputs[b],
                        targetClassA: batchTargets[b]
                    )
                    stepLoss += loss
                    if correct { stepCorrect += 1 }
                }
            }

            // Average gradients over batch
            bpm.scaleGradients(by: 1.0 / Float(config.batchSize))

            // Backpropagate through Gumbel-softmax: gradDeltaN → gradLogits
            grid.backwardGumbelSoftmax(temperature: currentTemp)

            // Adam update on logits
            bpm.adamUpdate(step: step)

            // Logging
            if step % config.logEvery == 0 || step == config.nSteps - 1 {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let msPerStep = (elapsed / Double(step + 1)) * 1000

                let avgLoss = stepLoss / Float(config.batchSize)
                let accuracy = Float(stepCorrect) / Float(config.batchSize)

                print(String(format:
                    "Step %5d/%d | Loss: %.4f | Acc: %3.0f%% | Temp: %.3f | %.0f ms/step",
                    step, config.nSteps, avgLoss, accuracy * 100,
                    currentTemp, msPerStep))
            }
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        print("------------------------------------------------------------")
        print(String(format: "Training complete in %.1fs (%.0f ms/step avg)",
                      totalTime, totalTime / Double(config.nSteps) * 1000))
    }

    // MARK: - Single-Step Training

    /// Process one sample through the photonic system (no recurrence).
    /// Returns (loss, isCorrect).
    private func trainSingleStep(
        inputA: [Float], targetClassA: Int
    ) -> (Float, Bool) {
        // 1. Encode input onto the field
        //    Input A = current tokens, Input B = state vector (zeros for single step)
        let stateB: [Float]? = config.bpttSteps > 1 ? stateManager.getStateForInputB() : nil
        ioEncoder.encodeInput(inputA: inputA, inputB: stateB, into: grid.fieldBuffer)

        // 2. Forward BPM propagation through all z-slices
        bpm.forwardPass()

        // 3. Read detector intensities
        let (outputA, outputB) = ioEncoder.decodeOutput(from: grid.fieldBuffer)

        // 4. Compute loss and check accuracy
        let loss = ioEncoder.computeLoss(intensitiesA: outputA, targetClassA: targetClassA)
        let predicted = argmax(outputA)
        let correct = predicted == targetClassA

        // 5. Compute output gradient (∂L/∂h*) into adjoint buffer
        ioEncoder.computeOutputGradient(
            field: grid.fieldBuffer,
            adjoint: grid.adjointBuffer,
            intensitiesA: outputA,
            targetClassA: targetClassA
        )

        // 6. Backward BPM propagation (adjoint method)
        bpm.backwardPass()

        // 7. Update state if doing recurrence
        if config.bpttSteps > 1 {
            stateManager.recordOutputAndUpdateState(outputA: outputA, outputB: outputB)
        }

        return (loss, correct)
    }

    // MARK: - Evaluation

    func evaluate() {
        let nEval = 200

        print("\n============================================================")
        print("EVALUATION (\(nEval) samples)")
        print("============================================================")

        // Evaluate with continuous delta_n (no Gumbel noise)
        grid.computeDeltaN(temperature: currentTemp, addNoise: false)

        var evalRng = SeededRNG(seed: 99999)  // Fixed eval seed
        let (evalInputs, evalTargets) = refModel.generateBatch(size: nEval, rng: &evalRng)

        var correct = 0
        for b in 0..<nEval {
            ioEncoder.encodeInput(inputA: evalInputs[b], inputB: nil, into: grid.fieldBuffer)
            bpm.forwardPass()
            let (outputA, _) = ioEncoder.decodeOutput(from: grid.fieldBuffer)
            if argmax(outputA) == evalTargets[b] { correct += 1 }
        }

        let accuracy = Float(correct) / Float(nEval)
        let baseline = 1.0 / Float(config.nOutputsA)

        print(String(format: "Continuous accuracy:  %.1f%%", accuracy * 100))
        print(String(format: "Random baseline:      %.1f%%", baseline * 100))

        // Evaluate with hard-quantized grid
        evaluateQuantized(evalInputs: evalInputs, evalTargets: evalTargets)
    }

    private func evaluateQuantized(evalInputs: [[Float]], evalTargets: [Int]) {
        let nEval = evalInputs.count

        // Hard quantize: argmax of logits → 4-bit indices → discrete delta_n
        let (indices, stats) = grid.hardQuantize()

        // Write the hard-quantized delta_n values into the deltaN buffer
        let dnPtr = grid.deltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels)
        let levels = config.dnLevels
        for v in 0..<config.totalVoxels {
            dnPtr[v] = levels[Int(indices[v])]
        }

        // Run evaluation with quantized grid
        var correct = 0
        for b in 0..<nEval {
            ioEncoder.encodeInput(inputA: evalInputs[b], inputB: nil, into: grid.fieldBuffer)
            bpm.forwardPass()
            let (outputA, _) = ioEncoder.decodeOutput(from: grid.fieldBuffer)
            if argmax(outputA) == evalTargets[b] { correct += 1 }
        }

        let quantAccuracy = Float(correct) / Float(nEval)
        print(String(format: "Quantized accuracy:   %.1f%%", quantAccuracy * 100))
        print(String(format: "Quantization loss:    %.1f%%",
                      (1.0 - quantAccuracy / max(Float(correct > 0 ? correct : 1) / Float(nEval), 0.01)) * 100))

        stats.printSummary()
    }

    // MARK: - Save Results

    func saveResults() {
        do {
            let outputURL = URL(fileURLWithPath: config.outputDir)
            try grid.save(to: outputURL)

            // Also save the config as JSON
            let configDict: [String: Any] = [
                "nx": config.nx, "ny": config.ny, "nz": config.nz,
                "wavelength": config.wavelength,
                "nBase": config.nBase,
                "dnMax": config.dnMax,
                "voxelPitch": config.voxelPitch,
                "dz": config.dz,
                "nLevels": config.nLevels,
                "nBits": config.nBits,
                "nInputsA": config.nInputsA,
                "nInputsB": config.nInputsB,
                "nOutputsA": config.nOutputsA,
                "nOutputsB": config.nOutputsB,
                "nSteps": config.nSteps,
                "lr": config.lr,
                "batchSize": config.batchSize,
                "sparsityLambda": config.sparsityLambda,
                "bpttSteps": config.bpttSteps,
                "finalTemperature": currentTemp
            ]
            let jsonData = try JSONSerialization.data(
                withJSONObject: configDict, options: [.prettyPrinted, .sortedKeys]
            )
            let configURL = outputURL.appendingPathComponent("training_config.json")
            try jsonData.write(to: configURL)

            print("\nResults saved to \(config.outputDir)/")
        } catch {
            print("Warning: Failed to save results — \(error)")
        }
    }

    // MARK: - Utilities

    private func argmax(_ values: [Float]) -> Int {
        var maxIdx = 0
        var maxVal = values[0]
        for i in 1..<values.count {
            if values[i] > maxVal {
                maxVal = values[i]
                maxIdx = i
            }
        }
        return maxIdx
    }
}

// ============================================================
// Main Entry Point
// ============================================================

@main
struct PhotonicTwinApp {
    static func main() {
        // Parse configuration
        var config = SimConfig()
        config.parseArgs(CommandLine.arguments)

        // Create output directory
        let fm = FileManager.default
        if !fm.fileExists(atPath: config.outputDir) {
            try? fm.createDirectory(
                atPath: config.outputDir,
                withIntermediateDirectories: true
            )
        }

        do {
            // Initialize engine
            let engine = try PhotonicTwinEngine(config: config)

            // Train
            engine.train()

            // Evaluate
            engine.evaluate()

            // Save
            engine.saveResults()

            print("\n============================================================")
            print("DONE")
            print("============================================================")
        } catch {
            print("Fatal error: \(error)")
            exit(1)
        }
    }
}
