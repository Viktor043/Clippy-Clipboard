// Photonic Neural Compute — VoxelGrid
// SNRGY Studios — February 2026
//
// 3D voxel grid with 4-bit quantization (16 discrete refractive index levels).
// During optimization, each voxel has 16 logits processed through Gumbel-softmax
// to yield a continuous delta_n. For engraving, argmax gives the hard 4-bit index.
// Packed format stores 2 voxels per byte.

import Foundation
import Metal

// MARK: - Seeded RNG (xorshift64)

struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        // Avoid a zero state which is a fixed point of xorshift64
        self.state = seed == 0 ? 0xDEADBEEF : seed
    }

    mutating func next() -> UInt64 {
        state ^= state &<< 13
        state ^= state &>> 7
        state ^= state &<< 17
        return state
    }

    /// Uniform Float in [0, 1)
    mutating func nextFloat() -> Float {
        return Float(next() & 0x00FFFFFF) / Float(0x01000000)
    }

    /// Standard Gumbel sample: -log(-log(u)), u ~ Uniform(0,1)
    mutating func nextGumbel() -> Float {
        var u: Float = 0
        // Avoid u == 0 or u == 1 exactly
        repeat { u = nextFloat() } while u <= 1e-10 || u >= 1.0 - 1e-10
        return -logf(-logf(u))
    }
}

// MARK: - Quantization Statistics

struct QuantStats {
    let histogram: [Int]       // Count per level (length nLevels)
    let totalVoxels: Int
    let nLevels: Int
    let dnMax: Float

    var sparsity: Float {
        // Fraction of voxels at level 0 (dn = 0)
        return Float(histogram[0]) / Float(totalVoxels)
    }

    func printSummary() {
        print("--- Quantization Statistics ---")
        print("Total voxels: \(totalVoxels)")
        print(String(format: "Sparsity (level 0): %.1f%%", sparsity * 100))
        let step = dnMax / Float(nLevels - 1)
        for (i, count) in histogram.enumerated() {
            let dn = Float(i) * step
            let pct = 100.0 * Float(count) / Float(totalVoxels)
            print(String(format: "  Level %2d (dn=%.4f): %6d  (%.1f%%)", i, dn, count, pct))
        }
    }
}

// MARK: - VoxelGrid

class VoxelGrid {
    let config: SimConfig
    let device: MTLDevice

    // --- GPU Buffers ---

    /// Learnable parameters: totalVoxels * nLevels floats
    let logits: MTLBuffer

    /// Continuous delta_n values computed from softmax(logits) dot dnLevels: totalVoxels floats
    let deltaN: MTLBuffer

    /// Gradient of loss w.r.t. logits: totalVoxels * nLevels floats
    let gradLogits: MTLBuffer

    /// Gradient from adjoint method w.r.t. delta_n: totalVoxels floats
    let gradDeltaN: MTLBuffer

    /// Adam first moment for logits: totalVoxels * nLevels floats
    let momentM: MTLBuffer

    /// Adam second moment for logits: totalVoxels * nLevels floats
    let momentV: MTLBuffer

    /// Working field (complex): fieldSize * 2 floats
    let fieldBuffer: MTLBuffer

    /// Adjoint field (complex): fieldSize * 2 floats
    let adjointBuffer: MTLBuffer

    /// Cached forward fields for adjoint backward pass: (nz+1) * fieldSize * 2 floats
    let fieldCache: MTLBuffer

    /// Angular-spectrum propagation kernel (complex): fieldSize * 2 floats
    let propagator: MTLBuffer

    /// FFT scratch buffer for Stockham ping-pong: fieldSize * 2 floats
    let fftTemp: MTLBuffer

    /// Precomputed dn levels for GPU kernels: nLevels floats
    let dnLevelsBuffer: MTLBuffer

    /// Seeded RNG for Gumbel noise
    private var rng: SeededRNG

    // MARK: - Initialization

    init(device: MTLDevice, config: SimConfig) {
        self.config = config
        self.device = device
        self.rng = SeededRNG(seed: config.seed)

        let nL = config.nLevels
        let nV = config.totalVoxels
        let fS = config.fieldSize

        let logitsBytes   = nV * nL * MemoryLayout<Float>.size
        let gridBytes     = nV * MemoryLayout<Float>.size
        let fieldBytes    = fS * 2 * MemoryLayout<Float>.size   // complex = 2 floats
        let cacheBytes    = (config.nz + 1) * fieldBytes

        // Allocate all buffers with .storageModeShared (unified memory on Apple Silicon)
        logits       = device.makeBuffer(length: logitsBytes, options: .storageModeShared)!
        deltaN       = device.makeBuffer(length: gridBytes,   options: .storageModeShared)!
        gradLogits   = device.makeBuffer(length: logitsBytes, options: .storageModeShared)!
        gradDeltaN   = device.makeBuffer(length: gridBytes,   options: .storageModeShared)!
        momentM      = device.makeBuffer(length: logitsBytes, options: .storageModeShared)!
        momentV      = device.makeBuffer(length: logitsBytes, options: .storageModeShared)!
        fieldBuffer  = device.makeBuffer(length: fieldBytes,  options: .storageModeShared)!
        adjointBuffer = device.makeBuffer(length: fieldBytes, options: .storageModeShared)!
        fieldCache   = device.makeBuffer(length: cacheBytes,  options: .storageModeShared)!
        propagator   = device.makeBuffer(length: fieldBytes,  options: .storageModeShared)!
        fftTemp      = device.makeBuffer(length: fieldBytes,  options: .storageModeShared)!

        // Create and fill dn levels buffer for GPU kernels
        let levelsBytes = nL * MemoryLayout<Float>.size
        dnLevelsBuffer = device.makeBuffer(length: levelsBytes, options: .storageModeShared)!
        let levelsPtr = dnLevelsBuffer.contents().bindMemory(to: Float.self, capacity: nL)
        let dnLevelsArray = config.dnLevels
        for i in 0..<nL {
            levelsPtr[i] = dnLevelsArray[i]
        }

        // Initialize logits with small random values (seeded)
        initializeLogits()

        // Compute initial delta_n from logits at high temperature (flat softmax)
        computeDeltaN(temperature: config.initialTemp, addNoise: false)

        // Build angular-spectrum propagation kernel
        buildPropagator()

        // Zero Adam moments
        zeroBuffer(momentM, count: nV * nL)
        zeroBuffer(momentV, count: nV * nL)

        // Zero gradient buffers
        zeroBuffer(gradLogits, count: nV * nL)
        zeroBuffer(gradDeltaN, count: nV)
    }

    // MARK: - Logit Initialization

    /// Initialize logits with small random values centered around 0.
    /// This produces a roughly uniform distribution over the 16 levels initially.
    private func initializeLogits() {
        let ptr = logits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let count = config.totalVoxels * config.nLevels
        for i in 0..<count {
            // Small uniform noise in [-0.1, 0.1]
            let u = rng.nextFloat()          // [0, 1)
            ptr[i] = (u - 0.5) * 0.2        // [-0.1, 0.1)
        }
    }

    // MARK: - Propagator (Angular Spectrum Method)

    /// Build the free-space propagation kernel in Fourier space.
    /// H(fx, fy) = exp(i * kz * dz) where kz = sqrt(k^2 - kx^2 - ky^2).
    /// Evanescent modes (kz^2 < 0) are zeroed out.
    /// Spatial frequencies follow numpy's fftfreq convention:
    ///   freq[i] = i / N           for i < N/2
    ///   freq[i] = (i - N) / N     for i >= N/2
    func buildPropagator() {
        let ptr = propagator.contents().bindMemory(
            to: Float.self, capacity: config.fieldSize * 2
        )
        let nx = config.nx
        let ny = config.ny
        let dx = config.voxelPitch
        let dy = config.voxelPitch
        let dz = config.dz
        let k = config.k0 * config.nBase     // wavenumber in medium

        for ix in 0..<nx {
            for iy in 0..<ny {
                // numpy fftfreq: freq = [0, 1, ..., N/2-1, -N/2, ..., -1] / N
                let freqX: Float
                if ix < (nx + 1) / 2 {
                    freqX = Float(ix) / Float(nx)
                } else {
                    freqX = Float(ix - nx) / Float(nx)
                }
                let freqY: Float
                if iy < (ny + 1) / 2 {
                    freqY = Float(iy) / Float(ny)
                } else {
                    freqY = Float(iy - ny) / Float(ny)
                }

                // Spatial frequencies in 1/m, then angular frequencies
                let kx = 2.0 * Float.pi * freqX / dx
                let ky = 2.0 * Float.pi * freqY / dy
                let kzSq = k * k - kx * kx - ky * ky

                let idx = (ix * ny + iy) * 2
                if kzSq > 0 {
                    let kz = sqrtf(kzSq)
                    let phase = kz * dz
                    ptr[idx]     = cosf(phase)    // Real
                    ptr[idx + 1] = sinf(phase)    // Imag
                } else {
                    // Evanescent — zero out
                    ptr[idx]     = 0.0
                    ptr[idx + 1] = 0.0
                }
            }
        }
    }

    // MARK: - Gumbel-Softmax (CPU)

    /// Compute continuous delta_n from logits using Gumbel-softmax relaxation.
    ///
    /// For each voxel v:
    ///   1. (Optionally) sample Gumbel noise g_j for each level j
    ///   2. Compute perturbed logits: y_j = (logits[v,j] + g_j) / temperature
    ///   3. Softmax: p_j = exp(y_j) / sum_k exp(y_k)
    ///   4. Weighted sum: deltaN[v] = sum_j p_j * dnLevels[j]
    ///
    /// As temperature -> 0, this approaches argmax (hard quantization).
    /// As temperature -> inf, this approaches uniform averaging.
    ///
    /// The logits buffer is never modified; a temporary array is used for softmax.
    ///
    /// - Parameters:
    ///   - temperature: Gumbel-softmax temperature (annealed during training)
    ///   - addNoise: If true, add Gumbel noise (training). If false, just softmax (evaluation).
    func computeDeltaN(temperature: Float, addNoise: Bool = true) {
        let logitsPtr = logits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let dnPtr = deltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels
        )
        let levels = config.dnLevels
        let nL = config.nLevels
        let invTemp = 1.0 / max(temperature, 1e-6)

        // Temporary buffer for perturbed logits (one voxel at a time)
        var perturbed = [Float](repeating: 0, count: nL)

        for v in 0..<config.totalVoxels {
            let base = v * nL

            // Perturb with Gumbel noise and scale by temperature
            var maxVal: Float = -Float.greatestFiniteMagnitude
            for j in 0..<nL {
                let noise: Float = addNoise ? rng.nextGumbel() : 0
                perturbed[j] = (logitsPtr[base + j] + noise) * invTemp
                if perturbed[j] > maxVal { maxVal = perturbed[j] }
            }

            // Numerically stable softmax
            var sumExp: Float = 0
            for j in 0..<nL {
                perturbed[j] = expf(perturbed[j] - maxVal)
                sumExp += perturbed[j]
            }

            // Weighted sum: deltaN = sum_j softmax_j * dnLevel_j
            var dn: Float = 0
            let invSum = 1.0 / sumExp
            for j in 0..<nL {
                dn += (perturbed[j] * invSum) * levels[j]
            }

            dnPtr[v] = dn
        }
    }

    // MARK: - Backward: Gradient of Gumbel-softmax

    /// Backpropagate gradDeltaN through the softmax to produce gradLogits.
    ///
    /// Given dL/d(deltaN) in gradDeltaN, compute dL/d(logits).
    /// For softmax: dp_j/dz_k = p_j (delta_jk - p_k) / temperature
    /// Chain rule: dL/dz_j = sum_k (dL/d_dn * d_dn/dp_k * dp_k/dz_j)
    ///           = (1/T) * p_j * (levels_j - sum_k p_k * levels_k) * dL/d_dn
    ///           = (1/T) * p_j * (levels_j - dn) * dL/d_dn
    ///
    /// - Parameter temperature: Current annealing temperature (must match forward pass)
    func backwardGumbelSoftmax(temperature: Float) {
        let logitsPtr = logits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let gradLogitsPtr = gradLogits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let gradDnPtr = gradDeltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels
        )
        let levels = config.dnLevels
        let nL = config.nLevels
        let invTemp = 1.0 / max(temperature, 1e-6)

        var probs = [Float](repeating: 0, count: nL)

        for v in 0..<config.totalVoxels {
            let base = v * nL

            // Recompute softmax probabilities (without Gumbel noise for gradient)
            var maxVal: Float = -Float.greatestFiniteMagnitude
            for j in 0..<nL {
                let z = logitsPtr[base + j] * invTemp
                probs[j] = z
                if z > maxVal { maxVal = z }
            }
            var sumExp: Float = 0
            for j in 0..<nL {
                probs[j] = expf(probs[j] - maxVal)
                sumExp += probs[j]
            }
            let invSum = 1.0 / sumExp
            var dn: Float = 0
            for j in 0..<nL {
                probs[j] *= invSum
                dn += probs[j] * levels[j]
            }

            // dL/dz_j = (1/T) * p_j * (levels_j - dn) * dL/d_dn
            let dLdDn = gradDnPtr[v]
            for j in 0..<nL {
                gradLogitsPtr[base + j] += invTemp * probs[j] * (levels[j] - dn) * dLdDn
            }
        }
    }

    // MARK: - Hard Quantization

    /// Argmax of logits per voxel to get the hard 4-bit index.
    /// Returns the indices array and statistics about the quantized grid.
    func hardQuantize() -> (indices: [UInt8], stats: QuantStats) {
        let logitsPtr = logits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let nL = config.nLevels
        let nV = config.totalVoxels

        var indices = [UInt8](repeating: 0, count: nV)
        var histogram = [Int](repeating: 0, count: nL)

        for v in 0..<nV {
            let base = v * nL
            var bestIdx = 0
            var bestVal = logitsPtr[base]
            for j in 1..<nL {
                if logitsPtr[base + j] > bestVal {
                    bestVal = logitsPtr[base + j]
                    bestIdx = j
                }
            }
            indices[v] = UInt8(bestIdx)
            histogram[bestIdx] += 1
        }

        let stats = QuantStats(
            histogram: histogram,
            totalVoxels: nV,
            nLevels: nL,
            dnMax: config.dnMax
        )
        return (indices, stats)
    }

    // MARK: - 4-Bit Packing

    /// Pack an array of 4-bit indices (0..15) into bytes, 2 voxels per byte.
    /// Voxel at even index goes into the high nibble, odd index into the low nibble.
    /// For an odd number of voxels, the last byte's low nibble is zero-padded.
    ///
    /// Layout: byte[k] = (indices[2k] << 4) | indices[2k+1]
    ///
    /// - Parameter indices: Array of UInt8 values in [0, 15]
    /// - Returns: Packed Data with ceil(count/2) bytes
    func pack4Bit(indices: [UInt8]) -> Data {
        let count = indices.count
        let packedCount = (count + 1) / 2
        var packed = Data(count: packedCount)

        packed.withUnsafeMutableBytes { rawBuf in
            let buf = rawBuf.bindMemory(to: UInt8.self)
            for k in 0..<packedCount {
                let hi = indices[2 * k] & 0x0F
                let lo: UInt8
                if 2 * k + 1 < count {
                    lo = indices[2 * k + 1] & 0x0F
                } else {
                    lo = 0
                }
                buf[k] = (hi << 4) | lo
            }
        }

        return packed
    }

    // MARK: - Save Engraving File

    /// Save the hard-quantized engraving file and metadata JSON.
    ///
    /// Files written:
    ///   - `<dir>/engraving.4bit`  — packed 4-bit grid (2 voxels/byte)
    ///   - `<dir>/engraving.json`  — metadata (dimensions, physics, stats)
    ///
    /// - Parameter directory: URL of the output directory
    func save(to directory: URL) throws {
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )

        // Hard quantize
        let (indices, stats) = hardQuantize()

        // Pack to 4-bit
        let packedData = pack4Bit(indices: indices)

        // Write packed binary
        let binaryURL = directory.appendingPathComponent("engraving.4bit")
        try packedData.write(to: binaryURL)

        // Write metadata
        let levels = config.dnLevels
        let metadata: [String: Any] = [
            "format": "4bit_packed",
            "version": 1,
            "nx": config.nx,
            "ny": config.ny,
            "nz": config.nz,
            "nLevels": config.nLevels,
            "nBits": config.nBits,
            "totalVoxels": config.totalVoxels,
            "packedBytes": packedData.count,
            "wavelength_m": config.wavelength,
            "nBase": config.nBase,
            "dnMax": config.dnMax,
            "voxelPitch_m": config.voxelPitch,
            "dz_m": config.dz,
            "dnLevels": levels,
            "sparsity": stats.sparsity,
            "histogram": stats.histogram,
            "physicalSize_mm": [
                "width": config.physicalWidth_mm,
                "height": config.physicalHeight_mm,
                "depth": config.physicalDepth_mm
            ]
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]
        )
        let jsonURL = directory.appendingPathComponent("engraving.json")
        try jsonData.write(to: jsonURL)

        print("Saved engraving to \(directory.path)")
        print("  Binary: \(packedData.count) bytes (\(config.totalVoxels) voxels)")
        stats.printSummary()
    }

    // MARK: - Buffer Utilities

    /// Zero all floats in a buffer.
    func zeroBuffer(_ buffer: MTLBuffer, count: Int) {
        memset(buffer.contents(), 0, count * MemoryLayout<Float>.size)
    }

    /// Zero the gradient buffers (call before each backward pass).
    func zeroGradients() {
        zeroBuffer(gradLogits, count: config.totalVoxels * config.nLevels)
        zeroBuffer(gradDeltaN, count: config.totalVoxels)
    }

    /// Copy deltaN values into a Swift array (useful for debugging).
    func getDeltaNArray() -> [Float] {
        let ptr = deltaN.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels
        )
        return Array(UnsafeBufferPointer(start: ptr, count: config.totalVoxels))
    }

    /// Get the softmax probabilities for a specific voxel (for debugging/visualization).
    /// Returns probabilities without Gumbel noise.
    func getVoxelProbabilities(voxelIndex v: Int, temperature: Float) -> [Float] {
        precondition(v >= 0 && v < config.totalVoxels, "Voxel index out of range")

        let logitsPtr = logits.contents().bindMemory(
            to: Float.self, capacity: config.totalVoxels * config.nLevels
        )
        let nL = config.nLevels
        let base = v * nL
        let invTemp = 1.0 / max(temperature, 1e-6)

        var probs = [Float](repeating: 0, count: nL)
        var maxVal: Float = -Float.greatestFiniteMagnitude

        for j in 0..<nL {
            probs[j] = logitsPtr[base + j] * invTemp
            if probs[j] > maxVal { maxVal = probs[j] }
        }

        var sumExp: Float = 0
        for j in 0..<nL {
            probs[j] = expf(probs[j] - maxVal)
            sumExp += probs[j]
        }
        let invSum = 1.0 / sumExp
        for j in 0..<nL {
            probs[j] *= invSum
        }

        return probs
    }

    // MARK: - Info

    func printInfo() {
        let nL = config.nLevels
        let nV = config.totalVoxels
        let fS = config.fieldSize

        let logitsMB  = Float(nV * nL * 4) / 1e6
        let deltaNMB  = Float(nV * 4) / 1e6
        let momentsMB = Float(nV * nL * 4 * 2) / 1e6
        let fieldMB   = Float(fS * 2 * 4 * 2) / 1e6   // field + adjoint
        let cacheMB   = Float((config.nz + 1) * fS * 2 * 4) / 1e6
        let propMB    = Float(fS * 2 * 4) / 1e6
        let totalMB   = logitsMB + deltaNMB + logitsMB + deltaNMB + momentsMB + fieldMB + cacheMB + propMB

        print("============================================================")
        print("VOXEL GRID — 4-bit Quantized (\(nL) levels)")
        print("============================================================")
        print("Grid:          \(config.nx) x \(config.ny) x \(config.nz) voxels (\(nV) total)")
        print(String(format: "Physical:      %.1f x %.1f x %.1f mm",
                      config.physicalWidth_mm, config.physicalHeight_mm, config.physicalDepth_mm))
        print("Quantization:  \(nL) levels (\(config.nBits)-bit), dn 0 .. \(config.dnMax)")
        print(String(format: "Packed size:   %d bytes (%.1f KB)",
                      config.packedGridBytes, Float(config.packedGridBytes) / 1024))
        print("--- GPU Buffers ---")
        print(String(format: "  logits:      %.1f MB (%d x %d floats)", logitsMB, nV, nL))
        print(String(format: "  deltaN:      %.1f MB (%d floats)", deltaNMB, nV))
        print(String(format: "  gradLogits:  %.1f MB", logitsMB))
        print(String(format: "  gradDeltaN:  %.1f MB", deltaNMB))
        print(String(format: "  moments:     %.1f MB (M + V)", momentsMB))
        print(String(format: "  fields:      %.1f MB (field + adjoint)", fieldMB))
        print(String(format: "  fieldCache:  %.1f MB (%d slices)", cacheMB, config.nz + 1))
        print(String(format: "  propagator:  %.1f MB", propMB))
        print(String(format: "  TOTAL:       %.1f MB", totalMB))
        print("============================================================")
    }
}
