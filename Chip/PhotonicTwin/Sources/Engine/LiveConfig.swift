import Foundation

/// Thread-safe wrapper for SimConfig, allowing live parameter tuning while training runs.
/// Training loop reads params each step; UI/HTTP can update between steps.
final class LiveConfig: @unchecked Sendable {
    private var _config: SimConfig
    private let lock = NSLock()

    /// Track which params have been changed live (for UI feedback)
    @Published var changeLog: [String] = []

    init(_ config: SimConfig) {
        _config = config
    }

    /// Read the current config (thread-safe snapshot)
    var config: SimConfig {
        lock.lock()
        defer { lock.unlock() }
        return _config
    }

    /// Atomically update config fields. Returns the updated config.
    @discardableResult
    func update(_ body: (inout SimConfig) -> Void) -> SimConfig {
        lock.lock()
        body(&_config)
        let snapshot = _config
        lock.unlock()
        return snapshot
    }

    // MARK: - Dirty flags (trigger buffer recomputation)

    /// I/O geometry changed → need to recompute spot positions + detector masks
    private var _ioGeometryDirty = false
    /// Physics changed (dnMax, dz, wavelength) → need to recompute propagator + dn_levels
    private var _physicsDirty = false

    var ioGeometryDirty: Bool {
        lock.lock(); defer { lock.unlock() }; return _ioGeometryDirty
    }
    func clearIOGeometryDirty() {
        lock.lock(); _ioGeometryDirty = false; lock.unlock()
    }

    var physicsDirty: Bool {
        lock.lock(); defer { lock.unlock() }; return _physicsDirty
    }
    func clearPhysicsDirty() {
        lock.lock(); _physicsDirty = false; lock.unlock()
    }

    // MARK: - JSON interface (for HTTP API)

    /// Returns ALL parameters as JSON dictionary
    func tunableParamsJSON() -> [String: Any] {
        let c = config
        return [
            // Training parameters (live-tunable)
            "lr": Double(c.lr),
            "minLR": Double(c.minLR),
            "lrSchedule": c.lrSchedule == .cosine ? "cosine" : "constant",
            "sparsityLambda": Double(c.sparsityLambda),
            "gradClipMax": Double(c.gradClipMax),
            "nSteps": c.nSteps,
            "warmupSteps": c.warmupSteps,
            // Adam parameters (live-tunable)
            "adamBeta1": Double(c.adamBeta1),
            "adamBeta2": Double(c.adamBeta2),
            "adamEps": Double(c.adamEps),
            // Gumbel-softmax temperature (live-tunable)
            "initialTemp": Double(c.initialTemp),
            "minTemp": Double(c.minTemp),
            "annealRate": Double(c.annealRate),
            // I/O geometry (live-tunable, triggers recompute)
            "inputXFraction": Double(c.inputXFraction),
            "outputXFraction": Double(c.outputXFraction),
            "spotSigmaOverride": Double(c.spotSigmaOverride),
            "detectorRadiusOverride": c.detectorRadiusOverride,
            "normScale": Double(c.normScale),
            // Input encoding (live-tunable)
            "usePhaseEncoding": c.usePhaseEncoding,
            // Physics parameters (live-tunable, triggers recompute)
            "dnMax": c.dnMax,
            "dz": c.dz,
            "kappaMax": Double(c.kappaMax),
            "saturationIntensity": Double(c.saturationIntensity),
            "wavelength": c.wavelength,
            "nBase": c.nBase,
            "voxelPitch": c.voxelPitch,
            // Recurrent parameters (live-tunable)
            "nPulses": c.nPulses,
            "thresholdInit": Double(c.thresholdInit),
            "feedbackRatioInit": Double(c.feedbackRatioInit),
            // EMA weight averaging (live-tunable)
            "useEMA": c.useEMA,
            "emaDecay": Double(c.emaDecay),
            "emaSwapInterval": c.emaSwapInterval,
            "emaEvalInterval": c.emaEvalInterval,
            // Runtime (live-tunable)
            "logEvery": c.logEvery,
            "checkpointInterval": c.checkpointInterval,
            // Read-only info (require restart to change)
            "nx": c.nx,
            "ny": c.ny,
            "nz": c.nz,
            "batchSize": c.batchSize,
            "useDirect": c.useDirect,
            "useDifferentialDetection": c.useDifferentialDetection,
            "nInputsA": c.nInputsA,
            "nOutputsA": c.nOutputsA,
            "nDetectors": c.nDetectors,
            "useLinearTeacher": c.useLinearTeacher,
            // Derived (read-only)
            "k0": c.k0,
            "totalVoxels": c.totalVoxels,
            "fieldSize": c.fieldSize,
        ]
    }

    /// Apply parameter changes from a JSON dictionary. Returns list of changed params.
    func applyJSON(_ dict: [String: Any]) -> [String] {
        var changed: [String] = []
        var ioChanged = false
        var physicsChanged = false

        update { config in
            // Training
            if let v = dict["lr"] as? Double { config.lr = Float(v); changed.append("lr=\(v)") }
            if let v = dict["minLR"] as? Double { config.minLR = Float(v); changed.append("minLR=\(v)") }
            if let v = dict["lrSchedule"] as? String {
                config.lrSchedule = v == "constant" ? .constant : .cosine
                changed.append("lrSchedule=\(v)")
            }
            if let v = dict["sparsityLambda"] as? Double { config.sparsityLambda = Float(v); changed.append("sparsityLambda=\(v)") }
            if let v = dict["nSteps"] as? Int { config.nSteps = v; changed.append("nSteps=\(v)") }
            if let v = dict["warmupSteps"] as? Int { config.warmupSteps = v; changed.append("warmupSteps=\(v)") }

            // Adam
            if let v = dict["adamBeta1"] as? Double { config.adamBeta1 = Float(v); changed.append("adamBeta1=\(v)") }
            if let v = dict["adamBeta2"] as? Double { config.adamBeta2 = Float(v); changed.append("adamBeta2=\(v)") }
            if let v = dict["adamEps"] as? Double { config.adamEps = Float(v); changed.append("adamEps=\(v)") }

            // Temperature
            if let v = dict["initialTemp"] as? Double { config.initialTemp = Float(v); changed.append("initialTemp=\(v)") }
            if let v = dict["minTemp"] as? Double { config.minTemp = Float(v); changed.append("minTemp=\(v)") }
            if let v = dict["annealRate"] as? Double { config.annealRate = Float(v); changed.append("annealRate=\(v)") }

            // I/O geometry
            if let v = dict["inputXFraction"] as? Double {
                config.inputXFraction = Float(max(0, min(1, v)))
                changed.append("inputXFraction=\(v)")
                ioChanged = true
            }
            if let v = dict["outputXFraction"] as? Double {
                config.outputXFraction = Float(max(0, min(1, v)))
                changed.append("outputXFraction=\(v)")
                ioChanged = true
            }

            // Spot size overrides (trigger I/O recompute)
            if let v = dict["spotSigmaOverride"] as? Double {
                config.spotSigmaOverride = Float(max(0, v))
                changed.append("spotSigmaOverride=\(v)")
                ioChanged = true
            }
            if let v = dict["detectorRadiusOverride"] as? Int {
                config.detectorRadiusOverride = max(0, v)
                changed.append("detectorRadiusOverride=\(v)")
                ioChanged = true
            }

            // Input encoding
            if let v = dict["usePhaseEncoding"] as? Bool {
                config.usePhaseEncoding = v; changed.append("usePhaseEncoding=\(v)")
            }
            // Also accept 0/1 integer for curl convenience
            if let v = dict["usePhaseEncoding"] as? Int {
                config.usePhaseEncoding = v != 0; changed.append("usePhaseEncoding=\(v != 0)")
            }

            // Physics (trigger recompute)
            if let v = dict["dnMax"] as? Double {
                config.dnMax = v; changed.append("dnMax=\(v)")
                physicsChanged = true
            }
            if let v = dict["dz"] as? Double {
                config.dz = v; changed.append("dz=\(v)")
                physicsChanged = true
            }
            if let v = dict["kappaMax"] as? Double {
                config.kappaMax = v; changed.append("kappaMax=\(v)")
            }
            if let v = dict["saturationIntensity"] as? Double {
                config.saturationIntensity = Float(v); changed.append("saturationIntensity=\(v)")
            }
            if let v = dict["wavelength"] as? Double {
                config.wavelength = v; changed.append("wavelength=\(v)")
                physicsChanged = true  // changes k0, propagator
            }

            // Recurrent
            if let v = dict["nPulses"] as? Int { config.nPulses = v; changed.append("nPulses=\(v)") }
            if let v = dict["thresholdInit"] as? Double { config.thresholdInit = Float(v); changed.append("thresholdInit=\(v)") }
            if let v = dict["feedbackRatioInit"] as? Double { config.feedbackRatioInit = Float(v); changed.append("feedbackRatioInit=\(v)") }

            // EMA
            if let v = dict["useEMA"] as? Bool { config.useEMA = v; changed.append("useEMA=\(v)") }
            if let v = dict["useEMA"] as? Int { config.useEMA = v != 0; changed.append("useEMA=\(v != 0)") }
            if let v = dict["emaDecay"] as? Double { config.emaDecay = Float(v); changed.append("emaDecay=\(v)") }
            if let v = dict["emaSwapInterval"] as? Int { config.emaSwapInterval = v; changed.append("emaSwapInterval=\(v)") }
            if let v = dict["emaEvalInterval"] as? Int { config.emaEvalInterval = v; changed.append("emaEvalInterval=\(v)") }

            // Gradient clipping
            if let v = dict["gradClipMax"] as? Double { config.gradClipMax = Float(v); changed.append("gradClipMax=\(v)") }

            // Loss normalization scale
            if let v = dict["normScale"] as? Double { config.normScale = Float(max(0.1, v)); changed.append("normScale=\(v)") }

            // Runtime
            if let v = dict["logEvery"] as? Int { config.logEvery = v; changed.append("logEvery=\(v)") }
            if let v = dict["checkpointInterval"] as? Int { config.checkpointInterval = v; changed.append("checkpointInterval=\(v)") }
        }

        if ioChanged {
            lock.lock(); _ioGeometryDirty = true; lock.unlock()
        }
        if physicsChanged {
            lock.lock(); _physicsDirty = true; lock.unlock()
        }
        for c in changed { logChange(c) }
        return changed
    }

    // MARK: - Convenience setters

    func setLR(_ value: Float) {
        update { $0.lr = value }
        logChange("lr → \(value)")
    }

    func setMinLR(_ value: Float) {
        update { $0.minLR = value }
        logChange("minLR → \(value)")
    }

    func setLRSchedule(_ schedule: SimConfig.LRSchedule) {
        update { $0.lrSchedule = schedule }
        logChange("lrSchedule → \(schedule)")
    }

    func setSparsityLambda(_ value: Float) {
        update { $0.sparsityLambda = value }
        logChange("sparsityLambda → \(value)")
    }

    func setInitialTemp(_ value: Float) {
        update { $0.initialTemp = value }
        logChange("initialTemp → \(value)")
    }

    func setMinTemp(_ value: Float) {
        update { $0.minTemp = value }
        logChange("minTemp → \(value)")
    }

    func setAnnealRate(_ value: Float) {
        update { $0.annealRate = value }
        logChange("annealRate → \(value)")
    }

    func setNSteps(_ value: Int) {
        update { $0.nSteps = value }
        logChange("nSteps → \(value)")
    }

    func setAdamBeta1(_ value: Float) {
        update { $0.adamBeta1 = value }
        logChange("adamBeta1 → \(value)")
    }

    func setAdamBeta2(_ value: Float) {
        update { $0.adamBeta2 = value }
        logChange("adamBeta2 → \(value)")
    }

    func setAdamEps(_ value: Float) {
        update { $0.adamEps = value }
        logChange("adamEps → \(value)")
    }

    func setInputXFraction(_ value: Float) {
        update { $0.inputXFraction = max(0, min(1, value)) }
        lock.lock(); _ioGeometryDirty = true; lock.unlock()
        logChange("inputXFraction → \(value)")
    }

    func setOutputXFraction(_ value: Float) {
        update { $0.outputXFraction = max(0, min(1, value)) }
        lock.lock(); _ioGeometryDirty = true; lock.unlock()
        logChange("outputXFraction → \(value)")
    }

    // MARK: - Logging

    private func logChange(_ msg: String) {
        DispatchQueue.main.async { [weak self] in
            self?.changeLog.append("[\(Self.timestamp)] \(msg)")
            if let count = self?.changeLog.count, count > 100 {
                self?.changeLog.removeFirst(count - 100)
            }
        }
    }

    private static var timestamp: String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f.string(from: Date())
    }
}
