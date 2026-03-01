import Foundation
import SwiftUI

/// Preset metadata for the UI picker.
struct PresetInfo: Identifiable {
    let id: Int             // preset number (1-13)
    let name: String        // display name
    let description: String // short description
    let config: SimConfig

    static let all: [PresetInfo] = [
        PresetInfo(id: 1, name: "quick", description: "64×64×20 Gumbel co-located", config: .quick),
        PresetInfo(id: 2, name: "medium", description: "128×128×40 Gumbel co-located", config: .medium),
        PresetInfo(id: 8, name: "debug", description: "32×32×10 direct co-located", config: .debug),
        PresetInfo(id: 9, name: "d2nnDebug", description: "32×32×10 separated I/O", config: .d2nnDebug),
        PresetInfo(id: 10, name: "d2nn", description: "64×64×20 separated I/O", config: .d2nn),
        PresetInfo(id: 11, name: "d2nnDiff", description: "64×64×20 sep I/O + differential", config: .d2nnDifferential),
        PresetInfo(id: 12, name: "d2nnDeep", description: "64×64×40 sep I/O + differential", config: .d2nnDeep),
        PresetInfo(id: 13, name: "d2nn128", description: "128×128×40 sep I/O + differential", config: .d2nn128),
    ]
}

/// Observable bridge between the training engine and SwiftUI.
/// Updated from background training thread, read on main thread.
@MainActor
final class TrainingMonitor: ObservableObject {
    // MARK: - Training progress
    @Published var currentStep: Int = 0
    @Published var totalSteps: Int = 0
    @Published var isRunning: Bool = false
    @Published var isPaused: Bool = false

    // MARK: - Metrics history (ring buffer, last 10K points)
    @Published var lossHistory: [Float] = []
    @Published var accuracyHistory: [Float] = []
    @Published var tempHistory: [Float] = []

    // MARK: - Current values
    @Published var currentLoss: Float = 0
    @Published var currentAccuracy: Float = 0
    @Published var currentTemperature: Float = 0
    @Published var stepsPerSecond: Double = 0
    @Published var estimatedTimeRemaining: TimeInterval = 0
    @Published var memoryUsedMB: Double = 0

    // MARK: - Smoothed values (EMA)
    @Published var smoothedLoss: Float = 0
    @Published var smoothedAccuracy: Float = 0
    @Published var smoothedLossHistory: [Float] = []
    @Published var smoothedAccuracyHistory: [Float] = []
    /// EMA alpha: 0.05 ≈ 40-point window (~2000 training steps at logEvery=50)
    var emaAlpha: Float = 0.05
    private var emaInitialized = false

    // MARK: - Peak tracking
    @Published var peakAccuracy: Float = 0
    @Published var peakAccuracyStep: Int = 0
    @Published var peakSmoothedAccuracy: Float = 0
    @Published var peakSmoothedAccuracyStep: Int = 0
    @Published var bestLoss: Float = Float.infinity
    @Published var bestLossStep: Int = 0
    @Published var bestSmoothedLoss: Float = Float.infinity
    @Published var bestSmoothedLossStep: Int = 0

    // MARK: - EMA tracking
    @Published var emaEnabled: Bool = false
    @Published var emaDecay: Float = 0.999

    // MARK: - Visualization data
    @Published var outputFieldMagnitude: [Float]? = nil   // NX × NY
    @Published var voxelSlice: [Float]? = nil             // NX × NY
    @Published var selectedZSlice: Int = 0

    // MARK: - Config info
    @Published var configSummary: String = ""
    @Published var lanAddress: String = ""
    @Published var currentPresetName: String = ""
    @Published var isRegressionMode: Bool = false

    // MARK: - Experiment info (displayed in UI info panel)
    @Published var experimentTitle: String = "Photonic Neural Network"
    @Published var experimentDescription: String = ""
    @Published var experimentMethod: String = ""
    @Published var experimentFindings: String = ""

    // MARK: - Run history
    @Published var runHistory: [RunRecord] = []
    @Published var showRunHistory: Bool = false

    // MARK: - Log messages
    @Published var logMessages: [String] = []

    static let maxHistory = 5_000

    let historyManager = RunHistoryManager()
    var currentRunId: UUID?

    init() {
        historyManager.cleanupStale()
        runHistory = historyManager.records
    }

    func addMetrics(step: Int, loss: Float, accuracy: Float, temperature: Float, stepsPerSec: Double) {
        currentStep = step
        currentLoss = loss
        currentAccuracy = accuracy
        currentTemperature = temperature
        stepsPerSecond = stepsPerSec

        // Update EMA
        if !emaInitialized {
            smoothedLoss = loss
            smoothedAccuracy = accuracy
            emaInitialized = true
        } else {
            smoothedLoss = emaAlpha * loss + (1 - emaAlpha) * smoothedLoss
            smoothedAccuracy = emaAlpha * accuracy + (1 - emaAlpha) * smoothedAccuracy
        }

        // Track peaks (raw)
        if accuracy > peakAccuracy {
            peakAccuracy = accuracy
            peakAccuracyStep = step
        }
        if loss < bestLoss && loss > 0 {
            bestLoss = loss
            bestLossStep = step
        }

        // Track peaks (smoothed)
        if smoothedAccuracy > peakSmoothedAccuracy {
            peakSmoothedAccuracy = smoothedAccuracy
            peakSmoothedAccuracyStep = step
        }
        if smoothedLoss < bestSmoothedLoss && smoothedLoss > 0 {
            bestSmoothedLoss = smoothedLoss
            bestSmoothedLossStep = step
        }

        lossHistory.append(loss)
        accuracyHistory.append(accuracy)
        smoothedLossHistory.append(smoothedLoss)
        smoothedAccuracyHistory.append(smoothedAccuracy)
        tempHistory.append(temperature)

        // Efficient trim: drop first half when exceeding max
        if lossHistory.count > Self.maxHistory {
            let half = Self.maxHistory / 2
            lossHistory = Array(lossHistory.suffix(half))
            accuracyHistory = Array(accuracyHistory.suffix(half))
            smoothedLossHistory = Array(smoothedLossHistory.suffix(half))
            smoothedAccuracyHistory = Array(smoothedAccuracyHistory.suffix(half))
            tempHistory = Array(tempHistory.suffix(half))
        }

        let remaining = totalSteps - step
        estimatedTimeRemaining = stepsPerSec > 0 ? Double(remaining) / stepsPerSec : 0

        // Update run history
        if let runId = currentRunId {
            historyManager.updateRun(id: runId, step: step, loss: loss, accuracy: accuracy)
        }
    }

    func log(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        logMessages.append("[\(timestamp)] \(message)")
        if logMessages.count > 200 {
            logMessages.removeFirst(logMessages.count - 200)
        }
    }

    /// Reset metrics for a new run
    func resetForNewRun() {
        currentStep = 0
        currentLoss = 0
        currentAccuracy = 0
        currentTemperature = 0
        stepsPerSecond = 0
        estimatedTimeRemaining = 0
        peakAccuracy = 0
        peakAccuracyStep = 0
        peakSmoothedAccuracy = 0
        peakSmoothedAccuracyStep = 0
        bestLoss = Float.infinity
        bestLossStep = 0
        bestSmoothedLoss = Float.infinity
        bestSmoothedLossStep = 0
        smoothedLoss = 0
        smoothedAccuracy = 0
        emaInitialized = false
        lossHistory = []
        accuracyHistory = []
        smoothedLossHistory = []
        smoothedAccuracyHistory = []
        tempHistory = []
        outputFieldMagnitude = nil
        voxelSlice = nil
        selectedZSlice = 0
        logMessages = []
    }

    func startRun(presetName: String, config: SimConfig) {
        // Finish previous run if any
        if let runId = currentRunId {
            historyManager.finishRun(id: runId, status: .stopped)
        }
        currentRunId = historyManager.startRun(presetName: presetName, config: config)
        currentPresetName = presetName
        isRegressionMode = config.useRegressionMode
        runHistory = historyManager.records

        // Auto-generate experiment info from config
        updateExperimentInfo(presetName: presetName, config: config)
    }

    private func updateExperimentInfo(presetName: String, config: SimConfig) {
        experimentTitle = "Photonic D2NN Classifier"

        var desc = "Training a diffractive neural network to classify 4-class patterns using light propagation through engineered glass.\n"
        desc += "Grid: \(config.nx)×\(config.ny)×\(config.nz) voxels"
        desc += " | I/O: in=\(String(format: "%.0f%%", config.inputXFraction * 100)) out=\(String(format: "%.0f%%", config.outputXFraction * 100))"
        if config.useDifferentialDetection { desc += " | Differential detection (8 detectors)" }
        if config.usePhaseEncoding { desc += " | Phase-encoded inputs" }
        experimentDescription = desc

        var method = ""
        method += config.useDirect ? "Direct optimization" : "Gumbel-softmax (16 levels)"
        method += " | Adam(β₁=\(config.adamBeta1), β₂=\(config.adamBeta2))"
        method += " | lr=\(config.lr) (\(config.lrSchedule == .constant ? "const" : "cosine"))"
        method += " | Batch=\(config.batchSize)"
        method += " | Score normalization [0,10]"
        if config.sparsityLambda > 0 { method += " | L1 sparsity=\(config.sparsityLambda)" }
        experimentMethod = method

        experimentFindings = """
        Best: 87.5% peak (score norm, 128×128×40, sep I/O + diff) | Previous best: 84.4%
        Score normalization (Mengu 2020): +3% improvement, faster convergence
        Baseline: 25% (random) | Co-located ceiling: ~70% | Sep I/O: 75% | Sep+diff: 84.4%
        """
    }

    func finishCurrentRun(status: RunRecord.RunStatus = .completed) {
        if let runId = currentRunId {
            historyManager.finishRun(id: runId, status: status)
            currentRunId = nil
            runHistory = historyManager.records
        }
    }
}
