// Photonic Neural Compute — I/O Encoder
// SNRGY Studios — February 2026
//
// Handles encoding input vectors as spatial light patterns on the glass
// input face, and reading output intensities from detector regions.
//
// Recurrent architecture:
//   Input face:  Region A (current tokens) + Region B (state vector)
//   Output face: Region A (response tokens) + Region B (updated state)

import Foundation
import Metal

class IOEncoder {
    let config: SimConfig

    // Detector masks: for each output channel, list of pixel indices
    let detectorMasksA: [[Int]]   // Output region A (response)
    let detectorMasksB: [[Int]]   // Output region B (state)
    var allDetectorMasks: [[Int]] { detectorMasksA + detectorMasksB }

    // Input spot centers (cx, cy) for each channel
    let inputCentersA: [(Int, Int)]
    let inputCentersB: [(Int, Int)]

    // Spot size (sigma of Gaussian)
    let spotSigma: Float

    init(config: SimConfig) {
        self.config = config

        let nx = config.nx
        let ny = config.ny

        // Layout: Input A uses top-left quadrant, Input B uses top-right
        // Output A uses bottom-left, Output B uses bottom-right
        // This gives maximum physical separation for the optical paths.

        spotSigma = max(Float(nx) / (6.0 * sqrt(Float(max(config.nInputsA, config.nInputsB)))), 3.0)
        let detRadius = Int(spotSigma * 1.5)

        // Input A: left half of input face
        inputCentersA = Self.layoutChannels(
            n: config.nInputsA,
            xRange: (Int(Float(nx) * 0.1), Int(Float(nx) * 0.45)),
            yRange: (Int(Float(ny) * 0.15), Int(Float(ny) * 0.85))
        )

        // Input B: right half of input face
        inputCentersB = Self.layoutChannels(
            n: config.nInputsB,
            xRange: (Int(Float(nx) * 0.55), Int(Float(nx) * 0.9)),
            yRange: (Int(Float(ny) * 0.15), Int(Float(ny) * 0.85))
        )

        // Output A: left half of output face
        let outputCentersA = Self.layoutChannels(
            n: config.nOutputsA,
            xRange: (Int(Float(nx) * 0.1), Int(Float(nx) * 0.45)),
            yRange: (Int(Float(ny) * 0.15), Int(Float(ny) * 0.85))
        )

        // Output B: right half of output face
        let outputCentersB = Self.layoutChannels(
            n: config.nOutputsB,
            xRange: (Int(Float(nx) * 0.55), Int(Float(nx) * 0.9)),
            yRange: (Int(Float(ny) * 0.15), Int(Float(ny) * 0.85))
        )

        // Build detector masks
        detectorMasksA = Self.buildMasks(centers: outputCentersA, radius: detRadius, nx: nx, ny: ny)
        detectorMasksB = Self.buildMasks(centers: outputCentersB, radius: detRadius, nx: nx, ny: ny)
    }

    // MARK: - Channel Layout

    /// Arrange n channels in a grid pattern within the given x/y ranges.
    static func layoutChannels(
        n: Int, xRange: (Int, Int), yRange: (Int, Int)
    ) -> [(Int, Int)] {
        if n == 0 { return [] }
        if n == 1 {
            return [((xRange.0 + xRange.1) / 2, (yRange.0 + yRange.1) / 2)]
        }

        let nCols = Int(ceil(sqrt(Double(n))))
        let nRows = Int(ceil(Double(n) / Double(nCols)))

        var positions = [(Int, Int)]()
        for idx in 0..<n {
            let row = idx / nCols
            let col = idx % nCols

            let cx: Int
            if nCols == 1 {
                cx = (xRange.0 + xRange.1) / 2
            } else {
                cx = xRange.0 + Int(Float(col) / Float(nCols - 1) * Float(xRange.1 - xRange.0))
            }

            let cy: Int
            if nRows == 1 {
                cy = (yRange.0 + yRange.1) / 2
            } else {
                cy = yRange.0 + Int(Float(row) / Float(nRows - 1) * Float(yRange.1 - yRange.0))
            }

            positions.append((cx, cy))
        }
        return positions
    }

    /// Build circular detector masks for given center positions.
    static func buildMasks(
        centers: [(Int, Int)], radius: Int, nx: Int, ny: Int
    ) -> [[Int]] {
        let radiusSq = radius * radius

        return centers.map { (cx, cy) in
            var mask = [Int]()
            let xMin = max(0, cx - radius)
            let xMax = min(nx - 1, cx + radius)
            let yMin = max(0, cy - radius)
            let yMax = min(ny - 1, cy + radius)

            for x in xMin...xMax {
                for y in yMin...yMax {
                    let dx = x - cx
                    let dy = y - cy
                    if dx * dx + dy * dy < radiusSq {
                        mask.append(x * ny + y)
                    }
                }
            }
            return mask
        }
    }

    // MARK: - Encode Input

    /// Encode input vectors A and B as Gaussian spots on the field buffer.
    /// inputA: current input tokens (nInputsA floats)
    /// inputB: state vector (nInputsB floats) — can be nil for first step
    func encodeInput(
        inputA: [Float], inputB: [Float]?,
        into fieldBuffer: MTLBuffer
    ) {
        let nx = config.nx
        let ny = config.ny
        let ptr = fieldBuffer.contents().bindMemory(to: Float.self, capacity: config.fieldSize * 2)

        // Zero the field
        memset(ptr, 0, config.fieldSize * 8)

        let sigma = spotSigma
        let twoSigmaSq = 2.0 * sigma * sigma

        // Write Input A spots
        for (i, val) in inputA.enumerated() {
            guard i < inputCentersA.count else { break }
            let (cx, cy) = inputCentersA[i]
            addGaussianSpot(ptr: ptr, nx: nx, ny: ny, cx: cx, cy: cy,
                            amplitude: val, twoSigmaSq: twoSigmaSq)
        }

        // Write Input B spots (state vector)
        if let stateVec = inputB {
            for (i, val) in stateVec.enumerated() {
                guard i < inputCentersB.count else { break }
                let (cx, cy) = inputCentersB[i]
                addGaussianSpot(ptr: ptr, nx: nx, ny: ny, cx: cx, cy: cy,
                                amplitude: val, twoSigmaSq: twoSigmaSq)
            }
        }

        // Normalize total power to 1
        var power: Float = 0
        for i in 0..<config.fieldSize {
            let re = ptr[i * 2]
            let im = ptr[i * 2 + 1]
            power += re * re + im * im
        }
        if power > 0 {
            let scale = 1.0 / sqrt(power)
            for i in 0..<config.fieldSize * 2 {
                ptr[i] *= scale
            }
        }
    }

    private func addGaussianSpot(
        ptr: UnsafeMutablePointer<Float>,
        nx: Int, ny: Int, cx: Int, cy: Int,
        amplitude: Float, twoSigmaSq: Float
    ) {
        let radius = Int(sqrt(twoSigmaSq) * 2)  // ~2σ radius
        let xMin = max(0, cx - radius)
        let xMax = min(nx - 1, cx + radius)
        let yMin = max(0, cy - radius)
        let yMax = min(ny - 1, cy + radius)

        for x in xMin...xMax {
            for y in yMin...yMax {
                let dx = Float(x - cx)
                let dy = Float(y - cy)
                let rSq = dx * dx + dy * dy
                let spot = amplitude * exp(-rSq / twoSigmaSq)
                let idx = x * ny + y
                ptr[idx * 2] += spot       // Real part only (coherent input)
            }
        }
    }

    // MARK: - Decode Output

    /// Read detector intensities from the output field.
    /// Returns (outputA, outputB) intensity arrays.
    func decodeOutput(from fieldBuffer: MTLBuffer) -> (outputA: [Float], outputB: [Float]) {
        let ptr = fieldBuffer.contents().bindMemory(to: Float.self, capacity: config.fieldSize * 2)

        func readIntensities(masks: [[Int]]) -> [Float] {
            masks.map { mask in
                var sum: Float = 0
                for idx in mask {
                    let re = ptr[idx * 2]
                    let im = ptr[idx * 2 + 1]
                    sum += re * re + im * im
                }
                return sum
            }
        }

        return (
            outputA: readIntensities(masks: detectorMasksA),
            outputB: readIntensities(masks: detectorMasksB)
        )
    }

    // MARK: - Loss Gradient → Field Gradient

    /// Compute ∂L/∂h* (gradient of loss w.r.t. output field) using softmax CE loss.
    /// This writes into the adjoint buffer.
    func computeOutputGradient(
        field: MTLBuffer,
        adjoint: MTLBuffer,
        intensitiesA: [Float],
        targetClassA: Int
    ) {
        let fieldPtr = field.contents().bindMemory(to: Float.self, capacity: config.fieldSize * 2)
        let adjPtr = adjoint.contents().bindMemory(to: Float.self, capacity: config.fieldSize * 2)

        // Zero adjoint
        memset(adjPtr, 0, config.fieldSize * 8)

        // Softmax CE on output region A
        let maxI = intensitiesA.max() ?? 0
        let expI = intensitiesA.map { exp($0 - maxI) }
        let sumExp = expI.reduce(0, +) + 1e-10
        let softmax = expI.map { $0 / sumExp }

        // ∂L/∂h*[x,y] = sum_i (softmax_i - delta_i) * h[x,y] * mask_i[x,y]
        for (det, mask) in detectorMasksA.enumerated() {
            let dLdI = softmax[det] - (det == targetClassA ? 1.0 : 0.0)
            for idx in mask {
                adjPtr[idx * 2] += dLdI * fieldPtr[idx * 2]          // Real
                adjPtr[idx * 2 + 1] += dLdI * fieldPtr[idx * 2 + 1]  // Imag
            }
        }
    }

    /// Compute softmax CE loss for output region A.
    func computeLoss(intensitiesA: [Float], targetClassA: Int) -> Float {
        let maxI = intensitiesA.max() ?? 0
        let shifted = intensitiesA.map { $0 - maxI }
        let logSumExp = log(shifted.map { exp($0) }.reduce(0, +) + 1e-10) + maxI
        return -intensitiesA[targetClassA] + logSumExp
    }
}
