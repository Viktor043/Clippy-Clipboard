import Foundation
import Accelerate

/// Generates random complex input fields and their exact 2D FFTs using Accelerate.
/// Used for the "optical FFT glass block" regression experiment.
final class FFTDataGenerator {
    let nx: Int
    let ny: Int
    let inputSize: Int  // size of the non-zero region (centered)
    private var rng: UInt64

    init(nx: Int, ny: Int, inputSize: Int, seed: UInt64 = 42) {
        self.nx = nx
        self.ny = ny
        self.inputSize = inputSize
        self.rng = seed
    }

    /// XOR-shift random float in [-1, 1]
    private func nextFloat() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(Int32(bitPattern: UInt32(rng & 0xFFFFFFFF))) / Float(Int32.max)
    }

    /// Generate one batch of random complex inputs and their exact 2D FFTs.
    ///
    /// - Returns: (inputs, targets) each as [B * NX * NY * 2] Float (interleaved real, imag = float2).
    ///   The input is a random complex field in the center `inputSize × inputSize` region, zero-padded.
    ///   The target is the unitary 2D DFT of the full zero-padded 128×128 input.
    func generateBatch(batchSize: Int) -> (inputs: [Float], targets: [Float]) {
        let N = nx * ny
        var inputs = [Float](repeating: 0, count: batchSize * N * 2)
        var targets = [Float](repeating: 0, count: batchSize * N * 2)

        let padX = (nx - inputSize) / 2
        let padY = (ny - inputSize) / 2

        let log2w = vDSP_Length(log2(Double(ny)))  // columns (inner dim)
        let log2h = vDSP_Length(log2(Double(nx)))  // rows (outer dim)
        guard let fftSetup = vDSP_create_fftsetup(max(log2w, log2h), FFTRadix(kFFTRadix2)) else {
            fatalError("[FFTDataGen] Failed to create vDSP FFT setup")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        // Working buffers (reused per sample)
        var realIn = [Float](repeating: 0, count: N)
        var imagIn = [Float](repeating: 0, count: N)
        var realOut = [Float](repeating: 0, count: N)
        var imagOut = [Float](repeating: 0, count: N)

        // Normalization: unitary DFT = 1/sqrt(N)
        let scale = 1.0 / sqrtf(Float(N))

        for b in 0..<batchSize {
            let offset = b * N * 2

            // Zero working buffers
            for i in 0..<N { realIn[i] = 0; imagIn[i] = 0 }

            // Generate random complex field in center inputSize × inputSize
            for iy in 0..<inputSize {
                for ix in 0..<inputSize {
                    let px = padX + ix
                    let py = padY + iy
                    let idx = px * ny + py  // row-major: row * ncols + col
                    realIn[idx] = nextFloat()
                    imagIn[idx] = nextFloat()
                }
            }

            // Store input as interleaved float2
            for p in 0..<N {
                inputs[offset + p * 2]     = realIn[p]
                inputs[offset + p * 2 + 1] = imagIn[p]
            }

            // Compute 2D FFT using vDSP (split complex, row-major)
            realIn.withUnsafeMutableBufferPointer { rIn in
                imagIn.withUnsafeMutableBufferPointer { iIn in
                    realOut.withUnsafeMutableBufferPointer { rOut in
                        imagOut.withUnsafeMutableBufferPointer { iOut in
                            var splitIn = DSPSplitComplex(
                                realp: rIn.baseAddress!, imagp: iIn.baseAddress!)
                            var splitOut = DSPSplitComplex(
                                realp: rOut.baseAddress!, imagp: iOut.baseAddress!)
                            vDSP_fft2d_zop(fftSetup, &splitIn, 1, 0,
                                           &splitOut, 1, 0,
                                           log2w, log2h,
                                           FFTDirection(kFFTDirection_Forward))
                        }
                    }
                }
            }

            // Store target as interleaved float2, with unitary normalization
            for p in 0..<N {
                targets[offset + p * 2]     = realOut[p] * scale
                targets[offset + p * 2 + 1] = imagOut[p] * scale
            }
        }

        return (inputs, targets)
    }
}
