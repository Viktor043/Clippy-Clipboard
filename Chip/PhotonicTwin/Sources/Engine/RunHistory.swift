import Foundation

/// A single training run record.
struct RunRecord: Codable, Identifiable {
    let id: UUID
    let presetName: String
    let configSummary: String
    let gridDescription: String   // e.g., "64×64×20"
    let startTime: Date
    var endTime: Date?
    var peakAccuracy: Float
    var bestLoss: Float
    var peakAccuracyStep: Int
    var bestLossStep: Int
    var finalStep: Int
    var totalSteps: Int
    var status: RunStatus

    enum RunStatus: String, Codable {
        case running, completed, stopped
    }

    init(presetName: String, config: SimConfig) {
        self.id = UUID()
        self.presetName = presetName
        self.startTime = Date()
        self.peakAccuracy = 0
        self.bestLoss = Float.infinity
        self.peakAccuracyStep = 0
        self.bestLossStep = 0
        self.finalStep = 0
        self.totalSteps = config.nSteps
        self.status = .running

        var summary = "\(config.nx)×\(config.ny)×\(config.nz) B=\(config.batchSize) lr=\(config.lr)"
        if config.inputXFraction != 0.5 { summary += " IO=\(config.inputXFraction)->\(config.outputXFraction)" }
        if config.useDifferentialDetection { summary += " DIFF" }
        if config.lrSchedule == .constant { summary += " const" }
        if config.useLinearTeacher { summary += " LIN" }
        self.configSummary = summary
        self.gridDescription = "\(config.nx)×\(config.ny)×\(config.nz)"
    }

    mutating func updateMetrics(step: Int, loss: Float, accuracy: Float) {
        finalStep = step
        if accuracy > peakAccuracy {
            peakAccuracy = accuracy
            peakAccuracyStep = step
        }
        if loss < bestLoss {
            bestLoss = loss
            bestLossStep = step
        }
    }

    mutating func finish(status: RunStatus = .completed) {
        self.endTime = Date()
        self.status = status
    }

    var durationString: String {
        let end = endTime ?? Date()
        let elapsed = end.timeIntervalSince(startTime)
        let h = Int(elapsed) / 3600
        let m = Int(elapsed) % 3600 / 60
        if h > 0 { return "\(h)h \(m)m" }
        return "\(m)m"
    }

    var dateString: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "MMM d, HH:mm"
        return fmt.string(from: startTime)
    }
}

/// Manages persistent run history stored as JSON.
final class RunHistoryManager {
    private static let historyPath = NSHomeDirectory() + "/Documents/photonic-twin/run_history.json"

    private(set) var records: [RunRecord] = []

    init() {
        load()
    }

    func load() {
        guard let data = FileManager.default.contents(atPath: Self.historyPath) else { return }
        do {
            records = try JSONDecoder().decode([RunRecord].self, from: data)
        } catch {
            print("[RunHistory] Failed to load: \(error)")
        }
    }

    func save() {
        do {
            let dir = (Self.historyPath as NSString).deletingLastPathComponent
            try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(records)
            try data.write(to: URL(fileURLWithPath: Self.historyPath))
        } catch {
            print("[RunHistory] Failed to save: \(error)")
        }
    }

    func startRun(presetName: String, config: SimConfig) -> UUID {
        let record = RunRecord(presetName: presetName, config: config)
        records.append(record)
        save()
        return record.id
    }

    func updateRun(id: UUID, step: Int, loss: Float, accuracy: Float) {
        guard let idx = records.firstIndex(where: { $0.id == id }) else { return }
        records[idx].updateMetrics(step: step, loss: loss, accuracy: accuracy)
        // Save periodically (every ~50 updates to avoid thrashing)
        if step % 500 == 0 { save() }
    }

    func finishRun(id: UUID, status: RunRecord.RunStatus = .completed) {
        guard let idx = records.firstIndex(where: { $0.id == id }) else { return }
        records[idx].finish(status: status)
        save()
    }

    /// Mark any "running" records as stopped (from previous crash/restart)
    func cleanupStale() {
        for i in records.indices {
            if records[i].status == .running {
                records[i].finish(status: .stopped)
            }
        }
        save()
    }
}
