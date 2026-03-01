import Foundation

/// Simple teacher model. The photonic glass block learns to replicate its behavior.
/// Runs on CPU — generates training data, not compute-bound.
final class ReferenceModel {
    let nInputs: Int
    let nOutputs: Int
    let hiddenSize: Int
    let useLinear: Bool

    private let w1: [Float]  // [nInputs × hiddenSize]
    private let b1: [Float]  // [hiddenSize]
    private let w2: [Float]  // [hiddenSize × nOutputs]
    private let b2: [Float]  // [nOutputs]
    // Linear mode: W_linear [nInputs × nOutputs]
    private let wLinear: [Float]

    private var rng: UInt64

    init(nInputs: Int, nOutputs: Int, hiddenSize: Int = 32, seed: UInt64 = 42, linear: Bool = false) {
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.hiddenSize = hiddenSize
        self.useLinear = linear
        self.rng = seed

        var localRng = seed
        func nextRandom() -> Float {
            localRng ^= localRng << 13
            localRng ^= localRng >> 7
            localRng ^= localRng << 17
            return Float(localRng & 0xFFFFFF) / Float(0xFFFFFF) * 2.0 - 1.0
        }

        if linear {
            // Linear model: output = W @ input (no hidden layer, no ReLU)
            let scale = sqrt(2.0 / Float(nInputs))
            var wl = [Float](repeating: 0, count: nInputs * nOutputs)
            for i in 0..<wl.count { wl[i] = nextRandom() * scale }
            self.wLinear = wl
            self.w1 = []; self.b1 = []; self.w2 = []; self.b2 = []
        } else {
            self.wLinear = []
            // Xavier initialization
            let scale1 = sqrt(2.0 / Float(nInputs))
            let scale2 = sqrt(2.0 / Float(hiddenSize))

            var w1 = [Float](repeating: 0, count: nInputs * hiddenSize)
            var w2 = [Float](repeating: 0, count: hiddenSize * nOutputs)

            for i in 0..<w1.count { w1[i] = nextRandom() * scale1 }
            for i in 0..<w2.count { w2[i] = nextRandom() * scale2 }

            self.w1 = w1
            self.b1 = [Float](repeating: 0, count: hiddenSize)
            self.w2 = w2
            self.b2 = [Float](repeating: 0, count: nOutputs)
        }
        self.rng = localRng
    }

    /// Forward pass. Returns predicted class index.
    func predict(_ input: [Float]) -> Int {
        var output: [Float]
        if useLinear {
            output = [Float](repeating: 0, count: nOutputs)
            for j in 0..<nOutputs {
                for i in 0..<nInputs {
                    output[j] += input[i] * wLinear[i * nOutputs + j]
                }
            }
        } else {
            // Hidden layer: ReLU(input @ W1 + b1)
            var hidden = b1
            for j in 0..<hiddenSize {
                var sum = hidden[j]
                for i in 0..<nInputs {
                    sum += input[i] * w1[i * hiddenSize + j]
                }
                hidden[j] = max(0, sum)
            }

            // Output layer: hidden @ W2 + b2
            output = b2
            for j in 0..<nOutputs {
                var sum = output[j]
                for i in 0..<hiddenSize {
                    sum += hidden[i] * w2[i * nOutputs + j]
                }
                output[j] = sum
            }
        }

        // Argmax
        var maxIdx = 0
        var maxVal = output[0]
        for j in 1..<nOutputs {
            if output[j] > maxVal {
                maxVal = output[j]
                maxIdx = j
            }
        }
        return maxIdx
    }

    /// Generate a class-balanced batch of (input_vector, target_class) pairs.
    /// Uses rejection sampling to ensure equal representation of each class.
    func generateBatch(size: Int) -> (inputs: [[Float]], targets: [Int]) {
        let perClass = size / nOutputs
        var buckets = [[([Float], Int)]](repeating: [], count: nOutputs)
        var attempts = 0
        let maxAttempts = size * 50  // safety limit

        // Fill buckets with rejection sampling
        while attempts < maxAttempts {
            var allFull = true
            for c in 0..<nOutputs {
                if buckets[c].count < perClass { allFull = false; break }
            }
            if allFull { break }

            var x = [Float](repeating: 0, count: nInputs)
            for i in 0..<nInputs {
                rng ^= rng << 13
                rng ^= rng >> 7
                rng ^= rng << 17
                x[i] = abs(Float(rng & 0xFFFFFF) / Float(0xFFFFFF))
            }
            let sum = x.reduce(0, +) + 1e-8
            for i in 0..<nInputs { x[i] /= sum }

            let target = predict(x)
            if buckets[target].count < perClass {
                buckets[target].append((x, target))
            }
            attempts += 1
        }

        // Collect all samples and shuffle
        var all: [([Float], Int)] = []
        for c in 0..<nOutputs {
            all.append(contentsOf: buckets[c].prefix(perClass))
        }

        // Fill remaining if needed (rare class couldn't fill)
        while all.count < size {
            var x = [Float](repeating: 0, count: nInputs)
            for i in 0..<nInputs {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
                x[i] = abs(Float(rng & 0xFFFFFF) / Float(0xFFFFFF))
            }
            let sum = x.reduce(0, +) + 1e-8
            for i in 0..<nInputs { x[i] /= sum }
            all.append((x, predict(x)))
        }

        // Fisher-Yates shuffle
        for i in stride(from: all.count - 1, through: 1, by: -1) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17
            let j = Int(rng % UInt64(i + 1))
            all.swapAt(i, j)
        }

        return (all.map { $0.0 }, all.map { $0.1 })
    }
}
