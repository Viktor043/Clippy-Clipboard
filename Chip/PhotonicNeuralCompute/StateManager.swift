// Photonic Neural Compute — State Manager
// SNRGY Studios — February 2026
//
// Manages the recurrent state vector for BPTT (Backpropagation Through Time).
//
// Recurrent architecture:
//   Input face:  Region A (current tokens) + Region B (state vector)
//   Output face: Region A (response tokens) + Region B (updated state)
//
// The state vector feeds from Output B of one time step to Input B
// of the next step. For BPTT, we store state and output histories
// so the backward pass can reconstruct the full computation graph.

import Foundation

class StateManager {
    let config: SimConfig

    // --- State Vectors ---

    /// Current state vector (nOutputsB floats).
    /// Feeds into Input B on the next forward step.
    private(set) var currentState: [Float]

    /// State history for BPTT: stores the state vector BEFORE each time step.
    /// Length = number of forward steps taken since last resetSequence().
    /// stateHistory[t] = state that was fed as Input B at time step t.
    private(set) var stateHistory: [[Float]]

    /// Output B history for BPTT gradient computation.
    /// outputBHistory[t] = raw Output B intensities at time step t.
    private(set) var outputBHistory: [[Float]]

    /// Output A history (for loss computation over multiple time steps).
    /// outputAHistory[t] = Output A intensities at time step t.
    private(set) var outputAHistory: [[Float]]

    /// Number of completed forward steps in current sequence.
    var timeSteps: Int { stateHistory.count }

    // --- Configuration ---

    /// Maximum norm for state vector (prevents runaway growth in recurrence).
    let maxStateNorm: Float = 2.0

    // MARK: - Initialization

    init(config: SimConfig) {
        self.config = config
        currentState = [Float](repeating: 0, count: config.nOutputsB)
        stateHistory = []
        outputBHistory = []
        outputAHistory = []
    }

    // MARK: - Sequence Management

    /// Reset state for start of a new sequence.
    /// Call this before beginning a new input sequence.
    func resetSequence() {
        currentState = [Float](repeating: 0, count: config.nOutputsB)
        stateHistory = []
        outputBHistory = []
        outputAHistory = []
    }

    /// Save the current state to history BEFORE a forward step.
    /// This is the state that will be used as Input B for time step t.
    func saveStateBeforeStep() {
        stateHistory.append(currentState)
    }

    /// Update state from Output B after a forward step.
    /// Applies normalization to prevent unbounded growth in the recurrent loop.
    ///
    /// - Parameters:
    ///   - outputA: Output region A intensities (response tokens)
    ///   - outputB: Output region B intensities (new state)
    func recordOutputAndUpdateState(outputA: [Float], outputB: [Float]) {
        outputAHistory.append(outputA)
        outputBHistory.append(outputB)

        // Update state: normalize Output B intensities to create the new state vector.
        // The intensities are non-negative (|E|^2), so we shift and scale to [-1, 1]
        // for a more useful state representation.
        currentState = normalizeState(outputB)
    }

    /// Get the current state vector for feeding into Input B.
    func getStateForInputB() -> [Float] {
        return currentState
    }

    /// Get the state that was used as Input B at a specific time step (for BPTT).
    func stateAt(step: Int) -> [Float]? {
        guard step >= 0 && step < stateHistory.count else { return nil }
        return stateHistory[step]
    }

    /// Get Output A at a specific time step.
    func outputAAt(step: Int) -> [Float]? {
        guard step >= 0 && step < outputAHistory.count else { return nil }
        return outputAHistory[step]
    }

    /// Get Output B at a specific time step.
    func outputBAt(step: Int) -> [Float]? {
        guard step >= 0 && step < outputBHistory.count else { return nil }
        return outputBHistory[step]
    }

    // MARK: - State Normalization

    /// Normalize state vector to prevent runaway growth in recurrence.
    ///
    /// Transforms raw output intensities (non-negative) into a bounded state:
    ///   1. Subtract mean to center around 0
    ///   2. Scale by 1/max(norm, maxStateNorm) to keep bounded
    ///
    /// This ensures the state vector doesn't grow unboundedly over many
    /// recurrent steps, which would cause numerical instability.
    private func normalizeState(_ rawOutput: [Float]) -> [Float] {
        let n = rawOutput.count
        guard n > 0 else { return rawOutput }

        // Center: subtract mean
        let mean = rawOutput.reduce(0, +) / Float(n)
        var centered = rawOutput.map { $0 - mean }

        // Clip norm
        let norm = sqrt(centered.map { $0 * $0 }.reduce(0, +) + 1e-8)
        if norm > maxStateNorm {
            let scale = maxStateNorm / norm
            centered = centered.map { $0 * scale }
        }

        return centered
    }

    // MARK: - BPTT Gradient Helpers

    /// Compute the gradient contribution from the state feedback path.
    ///
    /// When doing BPTT, the gradient flows backward through:
    ///   Output B (time t) → [normalize] → Input B (time t+1) → [BPM] → loss(t+1)
    ///
    /// This method computes dL/d(OutputB_t) given dL/d(InputB_{t+1}).
    /// The normalization is a simple scaling, so the gradient passes through
    /// with the same scaling factor.
    ///
    /// - Parameters:
    ///   - gradInputB: Gradient of loss w.r.t. the Input B that was computed from this Output B
    ///   - step: The time step index (which Output B produced this Input B)
    /// - Returns: Gradient of loss w.r.t. Output B at the given time step
    func gradientThroughStateNormalization(
        gradInputB: [Float], step: Int
    ) -> [Float] {
        guard let rawOutput = outputBAt(step: step) else {
            return [Float](repeating: 0, count: config.nOutputsB)
        }

        let n = rawOutput.count
        guard n > 0 else { return gradInputB }

        // Recompute normalization parameters
        let mean = rawOutput.reduce(0, +) / Float(n)
        let centered = rawOutput.map { $0 - mean }
        let norm = sqrt(centered.map { $0 * $0 }.reduce(0, +) + 1e-8)

        if norm > maxStateNorm {
            // Gradient through: out = (raw - mean) * maxStateNorm / norm
            // This is approximately: d(out)/d(raw) ≈ maxStateNorm / norm
            // (ignoring the Jacobian of the mean subtraction for simplicity)
            let scale = maxStateNorm / norm
            return gradInputB.map { $0 * scale }
        } else {
            // Gradient through: out = raw - mean
            // d(out_i)/d(raw_j) = delta_ij - 1/n
            // For simplicity, just pass through (the mean subtraction gradient
            // would subtract the mean of the gradient, which is a small correction)
            let gradMean = gradInputB.reduce(0, +) / Float(n)
            return gradInputB.map { $0 - gradMean }
        }
    }

    // MARK: - Info

    func printInfo() {
        print("--- State Manager ---")
        print("State dim (nOutputsB): \(config.nOutputsB)")
        print("BPTT steps:            \(config.bpttSteps)")
        print("Max state norm:        \(maxStateNorm)")
        print("Current history depth: \(timeSteps)")
    }
}
