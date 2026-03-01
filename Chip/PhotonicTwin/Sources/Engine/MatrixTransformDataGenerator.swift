import Foundation
import Accelerate

/// Generates random complex input fields and their matrix-transformed outputs.
/// Used to train glass to implement a specific unitary (or near-unitary) matrix transform.
/// The matrix operates on the full NX*NY complex field (same as FFT regression).
///
/// Supported matrix types:
/// - `.randomUnitary`: Random orthogonal matrix via QR decomposition
/// - `.hadamard`: Walsh-Hadamard matrix (normalized)
/// - `.identity`: Identity (baseline test — should trivially converge)
/// - `.custom`: User-provided matrix data
final class MatrixTransformDataGenerator {
    let nx: Int
    let ny: Int
    let matrixDim: Int  // N = nx * ny (square matrix dimension)

    /// Row-major N×N real matrix. For complex fields, applies independently to real and imag parts.
    /// This means it implements a real-linear transform: output = M @ input (for each component).
    private let matrix: [Float]
    let matrixType: String
    private var rng: UInt64

    enum MatrixType {
        case randomUnitary(seed: UInt64)
        case hadamard
        case identity
        case custom(data: [Float], name: String)
    }

    init(nx: Int, ny: Int, matrixType: MatrixType, seed: UInt64 = 42) {
        self.nx = nx
        self.ny = ny
        self.matrixDim = nx * ny
        self.rng = seed

        let N = self.matrixDim

        switch matrixType {
        case .randomUnitary(let matSeed):
            self.matrixType = "randomUnitary"
            self.matrix = MatrixTransformDataGenerator.generateRandomUnitary(N: N, seed: matSeed)

        case .hadamard:
            self.matrixType = "hadamard"
            self.matrix = MatrixTransformDataGenerator.generateHadamard(N: N)

        case .identity:
            self.matrixType = "identity"
            var mat = [Float](repeating: 0, count: N * N)
            for i in 0..<N { mat[i * N + i] = 1.0 }
            self.matrix = mat

        case .custom(let data, let name):
            self.matrixType = name
            precondition(data.count == N * N, "Custom matrix must be \(N)×\(N) = \(N*N) elements")
            self.matrix = data
        }
    }

    // MARK: - Matrix Generation

    /// Generate a random orthogonal matrix using QR decomposition of a random Gaussian matrix.
    /// Result: Q from QR(randn(N,N)), which is uniformly distributed over O(N).
    private static func generateRandomUnitary(N: Int, seed: UInt64) -> [Float] {
        var rng = seed
        func nextGaussian() -> Float {
            // Box-Muller transform
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            let u1 = max(Float(rng & 0xFFFFFFFF) / Float(UInt32.max), 1e-10)
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            let u2 = Float(rng & 0xFFFFFFFF) / Float(UInt32.max)
            return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * Float.pi * u2)
        }

        // Generate random Gaussian matrix
        var A = [Float](repeating: 0, count: N * N)
        for i in 0..<(N * N) { A[i] = nextGaussian() }

        // QR decomposition using LAPACK (via Accelerate)
        var m1 = Int32(N)
        var m2 = Int32(N)
        var lda = Int32(N)
        var tau = [Float](repeating: 0, count: N)
        var work = [Float](repeating: 0, count: N * N)
        var lwork = Int32(N * N)
        var info = Int32(0)

        // Column-major for LAPACK, but we just need an orthogonal matrix
        sgeqrf_(&m1, &m2, &A, &lda, &tau, &work, &lwork, &info)
        if info != 0 {
            print("[MatrixGen] WARNING: sgeqrf failed with info=\(info), using identity")
            var mat = [Float](repeating: 0, count: N * N)
            for i in 0..<N { mat[i * N + i] = 1.0 }
            return mat
        }

        // Extract Q from the QR factored form
        var m3 = Int32(N)
        var m4 = Int32(N)
        var m5 = Int32(N)
        var lda2 = Int32(N)
        var lwork2 = Int32(N * N)
        var info2 = Int32(0)
        sorgqr_(&m3, &m4, &m5, &A, &lda2, &tau, &work, &lwork2, &info2)
        if info2 != 0 {
            print("[MatrixGen] WARNING: sorgqr failed with info=\(info2), using identity")
            var mat = [Float](repeating: 0, count: N * N)
            for i in 0..<N { mat[i * N + i] = 1.0 }
            return mat
        }

        // A now contains Q in column-major format. Convert to row-major.
        var result = [Float](repeating: 0, count: N * N)
        for i in 0..<N {
            for j in 0..<N {
                result[i * N + j] = A[j * N + i]  // transpose col-major → row-major
            }
        }

        return result
    }

    /// Generate normalized Hadamard matrix of size N.
    /// N must be a power of 2. Uses Sylvester construction.
    private static func generateHadamard(N: Int) -> [Float] {
        precondition(N > 0 && (N & (N - 1)) == 0, "Hadamard size must be power of 2")

        var mat = [Float](repeating: 0, count: N * N)
        mat[0] = 1.0  // H_1 = [1]

        var size = 1
        while size < N {
            // H_{2n} = [[H_n, H_n], [H_n, -H_n]]
            for i in 0..<size {
                for j in 0..<size {
                    let val = mat[i * N + j]
                    // Top-right
                    mat[i * N + (j + size)] = val
                    // Bottom-left
                    mat[(i + size) * N + j] = val
                    // Bottom-right
                    mat[(i + size) * N + (j + size)] = -val
                }
            }
            size *= 2
        }

        // Normalize: 1/sqrt(N) to make it unitary
        let scale = 1.0 / sqrtf(Float(N))
        for i in 0..<(N * N) { mat[i] *= scale }
        return mat
    }

    // MARK: - Batch Generation

    /// XOR-shift random float in [-1, 1]
    private func nextFloat() -> Float {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return Float(Int32(bitPattern: UInt32(rng & 0xFFFFFFFF))) / Float(Int32.max)
    }

    /// Generate one batch of random complex inputs and their matrix-transformed outputs.
    ///
    /// - Returns: (inputs, targets) each as [B * NX * NY * 2] Float (interleaved real, imag).
    ///   Input: random complex field filling the full NX×NY grid.
    ///   Target: matrix @ input (applied independently to real and imag components).
    func generateBatch(batchSize: Int) -> (inputs: [Float], targets: [Float]) {
        let N = matrixDim  // nx * ny
        var inputs = [Float](repeating: 0, count: batchSize * N * 2)
        var targets = [Float](repeating: 0, count: batchSize * N * 2)

        // Working buffers
        var vecReal = [Float](repeating: 0, count: N)
        var vecImag = [Float](repeating: 0, count: N)
        var outReal = [Float](repeating: 0, count: N)
        var outImag = [Float](repeating: 0, count: N)

        for b in 0..<batchSize {
            let offset = b * N * 2

            // Generate random complex input
            for i in 0..<N {
                vecReal[i] = nextFloat()
                vecImag[i] = nextFloat()
            }

            // Store input as interleaved float2
            for p in 0..<N {
                inputs[offset + p * 2]     = vecReal[p]
                inputs[offset + p * 2 + 1] = vecImag[p]
            }

            // Matrix-vector multiply using vDSP for speed:
            // outReal = matrix @ vecReal
            // outImag = matrix @ vecImag
            matrix.withUnsafeBufferPointer { matPtr in
                vecReal.withUnsafeBufferPointer { rIn in
                    outReal.withUnsafeMutableBufferPointer { rOut in
                        // y = alpha * A * x + beta * y  (BLAS sgemv)
                        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                    Int32(N), Int32(N),
                                    1.0, matPtr.baseAddress!, Int32(N),
                                    rIn.baseAddress!, 1,
                                    0.0, rOut.baseAddress!, 1)
                    }
                }
                vecImag.withUnsafeBufferPointer { iIn in
                    outImag.withUnsafeMutableBufferPointer { iOut in
                        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                    Int32(N), Int32(N),
                                    1.0, matPtr.baseAddress!, Int32(N),
                                    iIn.baseAddress!, 1,
                                    0.0, iOut.baseAddress!, 1)
                    }
                }
            }

            // Store target as interleaved float2
            for p in 0..<N {
                targets[offset + p * 2]     = outReal[p]
                targets[offset + p * 2 + 1] = outImag[p]
            }
        }

        return (inputs, targets)
    }

    /// Verify the matrix is approximately orthogonal (Q^T Q ≈ I).
    /// Returns the maximum absolute deviation from identity.
    func verifyOrthogonality() -> Float {
        let N = matrixDim
        var maxDev: Float = 0
        for i in 0..<N {
            for j in 0..<N {
                var dot: Float = 0
                for k in 0..<N {
                    dot += matrix[k * N + i] * matrix[k * N + j]  // Q^T Q
                }
                let expected: Float = (i == j) ? 1.0 : 0.0
                maxDev = max(maxDev, abs(dot - expected))
            }
        }
        return maxDev
    }
}
