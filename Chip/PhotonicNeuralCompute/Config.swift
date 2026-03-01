// Photonic Neural Compute — Configuration
// SNRGY Studios — February 2026
//
// All simulation parameters in one place.

import Foundation

struct SimConfig {
    // --- Grid geometry ---
    var nx: Int = 64                  // Transverse grid width
    var ny: Int = 64                  // Transverse grid height
    var nz: Int = 40                  // Number of z-slices (depth)

    // --- Physics ---
    var wavelength: Float = 1.55e-6   // 1550 nm telecom C-band
    var nBase: Float = 1.45           // Fused silica base refractive index
    var dnMax: Float = 0.05           // Max refractive index change (femtosecond laser)
    var voxelPitch: Float = 2e-6      // Transverse voxel size (meters)
    var dz: Float = 10e-6             // Slice thickness (meters)

    // --- Quantization ---
    var nLevels: Int = 16             // 4-bit: 16 discrete refractive index levels
    var nBits: Int = 4                // Bits per voxel

    // --- I/O ---
    var nInputsA: Int = 4             // Input region A: current tokens
    var nInputsB: Int = 4             // Input region B: state vector
    var nOutputsA: Int = 4            // Output region A: response tokens
    var nOutputsB: Int = 4            // Output region B: updated state

    var nInputs: Int { nInputsA + nInputsB }
    var nOutputs: Int { nOutputsA + nOutputsB }

    // --- Optimizer ---
    var nSteps: Int = 3000
    var lr: Float = 0.01
    var batchSize: Int = 8
    var sparsityLambda: Float = 1e-4
    var adam_beta1: Float = 0.9
    var adam_beta2: Float = 0.999
    var adam_eps: Float = 1e-8

    // --- Gumbel-softmax annealing ---
    var initialTemp: Float = 2.0      // Starting temperature
    var annealRate: Float = 3e-4      // Exponential decay rate
    var minTemp: Float = 0.1          // Minimum temperature

    // --- Recurrent (BPTT) ---
    var bpttSteps: Int = 1            // Start with 1 (no BPTT), increase later
    var stateDim: Int = 256           // State vector dimensionality

    // --- Logging ---
    var logEvery: Int = 100
    var seed: UInt64 = 42
    var outputDir: String = "results_metal"

    // --- Derived quantities ---
    var k0: Float { 2.0 * .pi / wavelength }
    var totalVoxels: Int { nx * ny * nz }
    var fieldSize: Int { nx * ny }

    var physicalWidth_mm: Float { Float(nx) * voxelPitch * 1e3 }
    var physicalHeight_mm: Float { Float(ny) * voxelPitch * 1e3 }
    var physicalDepth_mm: Float { Float(nz) * dz * 1e3 }

    var maxPhasePerSlice: Float { k0 * dnMax * dz }  // radians

    /// 4-bit packed grid size in bytes (2 voxels per byte)
    var packedGridBytes: Int { (totalVoxels + 1) / 2 }

    /// The 16 discrete dn levels, evenly spaced from 0 to dnMax
    var dnLevels: [Float] {
        (0..<nLevels).map { Float($0) * dnMax / Float(nLevels - 1) }
    }

    /// Precomputed phase factors for the 16-entry lookup table
    func phaseLUT(k0: Float, dz: Float) -> [(Float, Float)] {
        dnLevels.map { dn in
            let phase = k0 * dn * dz
            return (cos(phase), sin(phase))  // (real, imag) of exp(i*phase)
        }
    }

    // --- Memory estimates ---
    var fieldMemoryMB: Float { Float(fieldSize * 8) / 1e6 }  // complex64
    var gridMemoryMB: Float { Float(totalVoxels * 4) / 1e6 }  // float32 logits
    var logitsMemoryMB: Float { Float(totalVoxels * nLevels * 4) / 1e6 }  // full logits
    var cacheMemoryMB: Float { Float((nz + 1) * fieldSize * 8) / 1e6 }

    var totalEstimatedMB: Float {
        fieldMemoryMB * 3 +  // field + adjoint + temp
        gridMemoryMB * 3 +   // grid + grad + moments
        logitsMemoryMB +     // per-voxel 16-class logits
        cacheMemoryMB        // field cache for adjoint
    }

    func printSummary() {
        print("============================================================")
        print("PHOTONIC NEURAL COMPUTE — Metal GPU Digital Twin")
        print("============================================================")
        print("Grid:         \(nx) x \(ny) x \(nz) voxels (\(totalVoxels) total)")
        print(String(format: "Physical:     %.1f x %.1f x %.1f mm",
                      physicalWidth_mm, physicalHeight_mm, physicalDepth_mm))
        print("Wavelength:   \(wavelength * 1e9) nm")
        print("Quantization: \(nLevels) levels (\(nBits)-bit)")
        print(String(format: "Phase/slice:  %.3f rad (max)", maxPhasePerSlice))
        print(String(format: "Phase total:  %.1f rad (max over %d slices)", maxPhasePerSlice * Float(nz), nz))
        print("I/O:          \(nInputsA)+\(nInputsB) inputs → \(nOutputsA)+\(nOutputsB) outputs")
        print("State dim:    \(stateDim)")
        print("BPTT steps:   \(bpttSteps)")
        print(String(format: "Memory est:   %.0f MB", totalEstimatedMB))
        print("============================================================")
    }
}

// MARK: - CLI Argument Parsing

extension SimConfig {
    mutating func parseArgs(_ args: [String]) {
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--nx":       i += 1; nx = Int(args[i])!
            case "--ny":       i += 1; ny = Int(args[i])!
            case "--nz":       i += 1; nz = Int(args[i])!
            case "--steps":    i += 1; nSteps = Int(args[i])!
            case "--lr":       i += 1; lr = Float(args[i])!
            case "--batch":    i += 1; batchSize = Int(args[i])!
            case "--sparsity": i += 1; sparsityLambda = Float(args[i])!
            case "--seed":     i += 1; seed = UInt64(args[i])!
            case "--output":   i += 1; outputDir = args[i]
            case "--log-every": i += 1; logEvery = Int(args[i])!
            case "--bptt":     i += 1; bpttSteps = Int(args[i])!
            case "--state-dim": i += 1; stateDim = Int(args[i])!
            case "--temp":     i += 1; initialTemp = Float(args[i])!
            default: print("Unknown argument: \(args[i])")
            }
            i += 1
        }
    }
}
