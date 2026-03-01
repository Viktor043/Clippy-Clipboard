// Photonic Neural Compute — Reference Model
// SNRGY Studios — February 2026
//
// Simple MLP reference model for generating training data.
// The glass block learns to replicate this model's classifications.
//
// For the recurrent architecture, the reference model only classifies
// the "A" inputs (current tokens). The "B" inputs (state vector)
// are handled by the recurrent state loop in the photonic system.

import Foundation

class ReferenceModel {
    let nInputsA: Int       // Number of input token channels
    let nOutputsA: Int      // Number of output classes
    let hiddenSize: Int

    // Weights (stored row-major)
    let w1: [Float]   // nInputsA x hiddenSize
    let b1: [Float]   // hiddenSize
    let w2: [Float]   // hiddenSize x nOutputsA
    let b2: [Float]   // nOutputsA

    init(config: SimConfig, hiddenSize: Int = 32) {
        self.nInputsA = config.nInputsA
        self.nOutputsA = config.nOutputsA
        self.hiddenSize = hiddenSize

        // Seeded RNG for reproducibility (SeededRNG is defined in VoxelGrid.swift)
        var rng = SeededRNG(seed: config.seed &+ 1000)  // Different seed from voxel grid

        let scale1 = sqrt(2.0 / Float(nInputsA))
        let scale2 = sqrt(2.0 / Float(hiddenSize))

        w1 = (0..<nInputsA * hiddenSize).map { _ in Float.random(in: -1...1, using: &rng) * scale1 }
        b1 = [Float](repeating: 0, count: hiddenSize)
        w2 = (0..<hiddenSize * nOutputsA).map { _ in Float.random(in: -1...1, using: &rng) * scale2 }
        b2 = [Float](repeating: 0, count: nOutputsA)
    }

    /// Forward pass through MLP. Returns raw logits (for softmax computation).
    func forward(_ input: [Float]) -> [Float] {
        // Layer 1: ReLU(W1 * x + b1)
        var hidden = [Float](repeating: 0, count: hiddenSize)
        for j in 0..<hiddenSize {
            var sum: Float = b1[j]
            for i in 0..<nInputsA {
                sum += input[i] * w1[i * hiddenSize + j]
            }
            hidden[j] = max(0, sum)  // ReLU
        }

        // Layer 2: W2 * h + b2
        var output = [Float](repeating: 0, count: nOutputsA)
        for j in 0..<nOutputsA {
            var sum: Float = b2[j]
            for i in 0..<hiddenSize {
                sum += hidden[i] * w2[i * nOutputsA + j]
            }
            output[j] = sum
        }

        return output
    }

    /// Predict class index (argmax of logits).
    func predict(_ input: [Float]) -> Int {
        let output = forward(input)

        var maxIdx = 0
        var maxVal = output[0]
        for i in 1..<nOutputsA {
            if output[i] > maxVal {
                maxVal = output[i]
                maxIdx = i
            }
        }
        return maxIdx
    }

    /// Generate a batch of (input, target_class) pairs.
    ///
    /// Each input is a normalized non-negative vector of length nInputsA
    /// that sums to ~1 (mimicking a probability distribution / token embedding).
    ///
    /// - Parameters:
    ///   - size: Number of samples in the batch
    ///   - rng: Seeded RNG (inout for reproducibility)
    /// - Returns: Tuple of (inputs, targets)
    func generateBatch(
        size: Int, rng: inout SeededRNG
    ) -> (inputs: [[Float]], targets: [Int]) {
        var inputs = [[Float]]()
        var targets = [Int]()

        for _ in 0..<size {
            // Random non-negative input vector, normalized to sum = 1
            var x = (0..<nInputsA).map { _ in abs(Float.random(in: -1...1, using: &rng)) }
            let sum = x.reduce(0, +) + 1e-8
            x = x.map { $0 / sum }

            let y = predict(x)
            inputs.append(x)
            targets.append(y)
        }

        return (inputs, targets)
    }

    /// Generate a batch for recurrent training (multiple time steps per sequence).
    ///
    /// Each sequence consists of `seqLen` input vectors. The target is the
    /// classification at each time step (since the reference model is memoryless,
    /// each step's target is independent — the recurrent state handling is the
    /// photonic system's job to learn).
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences
    ///   - seqLen: Number of time steps per sequence
    ///   - rng: Seeded RNG
    /// - Returns: (inputs: [batch][time][features], targets: [batch][time])
    func generateSequenceBatch(
        batchSize: Int, seqLen: Int, rng: inout SeededRNG
    ) -> (inputs: [[[Float]]], targets: [[Int]]) {
        var inputs = [[[Float]]]()
        var targets = [[Int]]()

        for _ in 0..<batchSize {
            var seqInputs = [[Float]]()
            var seqTargets = [Int]()

            for _ in 0..<seqLen {
                var x = (0..<nInputsA).map { _ in abs(Float.random(in: -1...1, using: &rng)) }
                let sum = x.reduce(0, +) + 1e-8
                x = x.map { $0 / sum }

                seqTargets.append(predict(x))
                seqInputs.append(x)
            }

            inputs.append(seqInputs)
            targets.append(seqTargets)
        }

        return (inputs, targets)
    }

    // MARK: - Info

    func printInfo() {
        print("--- Reference Model ---")
        print("Architecture:  MLP (\(nInputsA) → \(hiddenSize) → \(nOutputsA))")
        print("Parameters:    \(nInputsA * hiddenSize + hiddenSize + hiddenSize * nOutputsA + nOutputsA)")
        print("Activation:    ReLU")

        // Quick class balance check
        var rng = SeededRNG(seed: 12345)
        let (_, targets) = generateBatch(size: 1000, rng: &rng)
        var counts = [Int](repeating: 0, count: nOutputsA)
        for t in targets { counts[t] += 1 }
        let balance = counts.map { String(format: "%.0f%%", Float($0) / 10.0) }.joined(separator: "/")
        print("Class balance: \(balance) (1000 samples)")
    }
}
