import Foundation
import Metal

/// Manages checkpoint save/load for training state persistence.
final class CheckpointManager {
    let config: SimConfig
    let baseDir: URL

    init(config: SimConfig, baseDir: URL? = nil) {
        self.config = config
        self.baseDir = baseDir ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/photonic-twin/checkpoints")
    }

    func checkpointDir(step: Int) -> URL {
        baseDir.appendingPathComponent("step_\(String(format: "%06d", step))")
    }

    struct MetricsSnapshot: Sendable {
        var loss: Float = 0
        var accuracy: Float = 0
        var stepsPerSecond: Double = 0
    }

    func save(
        step: Int,
        logits: MTLBuffer,
        momentM: MTLBuffer,
        momentV: MTLBuffer,
        kappaLogits: MTLBuffer? = nil,
        kappaMomentM: MTLBuffer? = nil,
        kappaMomentV: MTLBuffer? = nil,
        config: SimConfig,
        metrics: MetricsSnapshot = MetricsSnapshot()
    ) throws {
        let dir = checkpointDir(step: step)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let logitsCount = config.totalVoxels * config.nLevels

        // Save binary buffers — refractive index
        try saveBuffer(logits, count: logitsCount, to: dir.appendingPathComponent("logits.bin"))
        try saveBuffer(momentM, count: logitsCount, to: dir.appendingPathComponent("moment_m.bin"))
        try saveBuffer(momentV, count: logitsCount, to: dir.appendingPathComponent("moment_v.bin"))

        // Save binary buffers — absorption
        if let kl = kappaLogits {
            try saveBuffer(kl, count: logitsCount, to: dir.appendingPathComponent("kappa_logits.bin"))
        }
        if let km = kappaMomentM {
            try saveBuffer(km, count: logitsCount, to: dir.appendingPathComponent("kappa_moment_m.bin"))
        }
        if let kv = kappaMomentV {
            try saveBuffer(kv, count: logitsCount, to: dir.appendingPathComponent("kappa_moment_v.bin"))
        }

        // Save manifest
        let manifest: [String: Any] = [
            "version": 2,
            "step": step,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "config": [
                "nx": config.nx, "ny": config.ny, "nz": config.nz,
                "nLevels": config.nLevels,
                "batchSize": config.batchSize,
                "lr": Double(config.lr),
                "nSteps": config.nSteps,
                "seed": Int(config.seed),
                "kappaMax": config.kappaMax
            ],
            "training_state": [
                "temperature": Double(config.temperature(at: step)),
                "steps_completed": step
            ]
        ]
        let json = try JSONSerialization.data(withJSONObject: manifest, options: .prettyPrinted)
        try json.write(to: dir.appendingPathComponent("manifest.json"))

        // Save metrics snapshot (guard against NaN which crashes JSON serialization)
        let safeLoss = metrics.loss.isFinite ? metrics.loss : Float(0)
        let safeAcc = metrics.accuracy.isFinite ? metrics.accuracy : Float(0)
        let safeSpeed = metrics.stepsPerSecond.isFinite ? metrics.stepsPerSecond : 0.0
        let metricsDict: [String: Any] = [
            "step": step,
            "loss": safeLoss,
            "accuracy": safeAcc,
            "steps_per_second": safeSpeed
        ]
        let metricsJson = try JSONSerialization.data(withJSONObject: metricsDict, options: .prettyPrinted)
        try metricsJson.write(to: dir.appendingPathComponent("metrics.json"))

        // Update latest symlink
        let latestLink = baseDir.appendingPathComponent("latest")
        try? FileManager.default.removeItem(at: latestLink)
        try FileManager.default.createSymbolicLink(at: latestLink, withDestinationURL: dir)

        // Cleanup old checkpoints
        cleanupOldCheckpoints()
    }

    func load(step: Int, logits: MTLBuffer, momentM: MTLBuffer, momentV: MTLBuffer) throws -> Int {
        let dir = checkpointDir(step: step)
        let logitsCount = config.totalVoxels * config.nLevels

        try loadBuffer(logits, count: logitsCount, from: dir.appendingPathComponent("logits.bin"))
        try loadBuffer(momentM, count: logitsCount, from: dir.appendingPathComponent("moment_m.bin"))
        try loadBuffer(momentV, count: logitsCount, from: dir.appendingPathComponent("moment_v.bin"))

        // Read step from manifest
        let manifestData = try Data(contentsOf: dir.appendingPathComponent("manifest.json"))
        if let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
           let savedStep = manifest["step"] as? Int {
            return savedStep
        }
        return step
    }

    // MARK: - Private helpers

    private func saveBuffer(_ buffer: MTLBuffer, count: Int, to url: URL) throws {
        let data = Data(bytes: buffer.contents(), count: count * MemoryLayout<Float>.size)
        try data.write(to: url)
    }

    private func loadBuffer(_ buffer: MTLBuffer, count: Int, from url: URL) throws {
        let data = try Data(contentsOf: url)
        _ = data.withUnsafeBytes { ptr in
            memcpy(buffer.contents(), ptr.baseAddress!, min(data.count, count * MemoryLayout<Float>.size))
        }
    }

    private func cleanupOldCheckpoints() {
        guard let entries = try? FileManager.default.contentsOfDirectory(at: baseDir, includingPropertiesForKeys: [.isDirectoryKey]) else { return }

        let dirs = entries.filter { url in
            url.lastPathComponent.hasPrefix("step_") &&
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }.sorted { $0.lastPathComponent > $1.lastPathComponent }

        // Keep only maxCheckpoints newest
        if dirs.count > config.maxCheckpoints {
            for dir in dirs.suffix(from: config.maxCheckpoints) {
                try? FileManager.default.removeItem(at: dir)
            }
        }
    }
}
