import SwiftUI
import Charts

struct DashboardView: View {
    @ObservedObject var monitor: TrainingMonitor
    let onStart: () -> Void
    let onPause: () -> Void
    let onNewRun: (PresetInfo) -> Void

    @State private var showPresetPicker = false

    var body: some View {
        VStack(spacing: 0) {
            // Top status bar
            statusBar
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.3))

            Divider().background(Color.gray.opacity(0.3))

            // Main content
            HSplitView {
                // Left: Charts + Log
                VStack(spacing: 12) {
                    LossChartView(
                        lossHistory: monitor.lossHistory,
                        accuracyHistory: monitor.accuracyHistory,
                        smoothedLossHistory: monitor.smoothedLossHistory,
                        smoothedAccuracyHistory: monitor.smoothedAccuracyHistory
                    )
                    .frame(minHeight: 200)

                    // Peak metrics bar
                    peakMetricsBar
                        .padding(.horizontal, 4)

                    // Experiment info panel
                    experimentInfoPanel
                        .padding(.horizontal, 4)

                    // Log messages
                    logView
                        .frame(minHeight: 100)
                }
                .padding(12)
                .frame(minWidth: 400)

                // Right: Visualizations + Run History
                VStack(spacing: 12) {
                    if let field = monitor.outputFieldMagnitude {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Output Field |E|\u{00B2}").font(.caption).foregroundColor(.secondary)
                            FieldView(data: field, nx: gridNX, ny: gridNY)
                                .aspectRatio(1, contentMode: .fit)
                        }
                    }

                    if let slice = monitor.voxelSlice {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Voxel Slice z=\(monitor.selectedZSlice)").font(.caption).foregroundColor(.secondary)
                                Slider(value: Binding(
                                    get: { Double(monitor.selectedZSlice) },
                                    set: { monitor.selectedZSlice = Int($0) }
                                ), in: 0...Double(max(gridNZ - 1, 1)), step: 1)
                                .frame(maxWidth: 150)
                            }
                            FieldView(data: slice, nx: gridNX, ny: gridNY)
                                .aspectRatio(1, contentMode: .fit)
                        }
                    }

                    // Run History
                    runHistoryView
                        .frame(minHeight: 120)

                    Spacer()
                }
                .padding(12)
                .frame(minWidth: 300)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .sheet(isPresented: $showPresetPicker) {
            presetPickerSheet
        }
    }

    private var gridNX: Int {
        let parts = monitor.configSummary.split(separator: "\u{00D7}")
        return Int(parts.first ?? "128") ?? 128
    }

    private var gridNY: Int {
        let parts = monitor.configSummary.split(separator: "\u{00D7}")
        return parts.count > 1 ? (Int(parts[1]) ?? 128) : 128
    }

    private var gridNZ: Int {
        let parts = monitor.configSummary.split(separator: "\u{00D7}")
        if parts.count > 2 {
            let zPart = parts[2].split(separator: " ").first ?? "40"
            return Int(zPart) ?? 40
        }
        return 40
    }

    // MARK: - Status bar

    private var statusBar: some View {
        HStack(spacing: 12) {
            // Preset name badge
            if !monitor.currentPresetName.isEmpty {
                Text(monitor.currentPresetName)
                    .font(.system(.caption, design: .monospaced).bold())
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.blue.opacity(0.2))
                    .cornerRadius(4)
            }

            metricPill("Step", "\(monitor.currentStep)/\(monitor.totalSteps)")
            metricPillDual(
                "Loss",
                raw: String(format: "%.4f", monitor.currentLoss),
                smoothed: String(format: "%.4f", monitor.smoothedLoss)
            )
            metricPillDual(
                "Accuracy",
                raw: String(format: "%.1f%%", monitor.currentAccuracy * 100),
                smoothed: String(format: "%.1f%%", monitor.smoothedAccuracy * 100)
            )
            metricPill("Speed", String(format: "%.1f/s", monitor.stepsPerSecond))
            metricPill("ETA", formatETA(monitor.estimatedTimeRemaining))

            Spacer()

            // LAN address
            if !monitor.lanAddress.isEmpty {
                Text(monitor.lanAddress)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.blue)
                    .textSelection(.enabled)
            }

            // New Run button
            Button("New Run") {
                showPresetPicker = true
            }
            .buttonStyle(.bordered)
            .tint(.blue)

            // Pause/Resume
            Button(monitor.isRunning ? "Pause" : "Resume") {
                if monitor.isRunning { onPause() } else { onStart() }
            }
            .buttonStyle(.bordered)
            .tint(monitor.isRunning ? .orange : .green)
        }
    }

    // MARK: - Peak metrics bar

    private var peakMetricsBar: some View {
        HStack(spacing: 16) {
            // Raw peak accuracy
            HStack(spacing: 4) {
                Image(systemName: "arrow.up.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
                Text("Peak:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(String(format: "%.1f%%", monitor.peakAccuracy * 100))
                    .font(.system(.caption, design: .monospaced).bold())
                    .foregroundColor(.green)
                Text("@ \(monitor.peakAccuracyStep)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            // Smoothed peak accuracy
            HStack(spacing: 4) {
                Image(systemName: "chart.line.uptrend.xyaxis")
                    .foregroundColor(.cyan)
                    .font(.caption)
                Text("Trend Peak:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(String(format: "%.1f%%", monitor.peakSmoothedAccuracy * 100))
                    .font(.system(.caption, design: .monospaced).bold())
                    .foregroundColor(.cyan)
                Text("@ \(monitor.peakSmoothedAccuracyStep)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            // Best smoothed loss
            HStack(spacing: 4) {
                Image(systemName: "arrow.down.circle.fill")
                    .foregroundColor(.orange)
                    .font(.caption)
                Text("Best Loss:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(monitor.bestSmoothedLoss < Float.infinity ? String(format: "%.4f", monitor.bestSmoothedLoss) : "--")
                    .font(.system(.caption, design: .monospaced).bold())
                    .foregroundColor(.orange)
                Text("(trend)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(Color.black.opacity(0.15))
        .cornerRadius(6)
    }

    // MARK: - Preset picker sheet

    private var presetPickerSheet: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Select Preset")
                .font(.title2.bold())

            ScrollView {
                VStack(spacing: 8) {
                    ForEach(PresetInfo.all) { preset in
                        Button(action: {
                            showPresetPicker = false
                            onNewRun(preset)
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(preset.name)
                                        .font(.system(.body, design: .monospaced).bold())
                                    Text(preset.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                                Image(systemName: "play.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(.blue)
                            }
                            .padding(12)
                            .background(Color.black.opacity(0.1))
                            .cornerRadius(8)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .frame(maxHeight: 400)

            HStack {
                Spacer()
                Button("Cancel") { showPresetPicker = false }
                    .buttonStyle(.bordered)
            }
        }
        .padding(20)
        .frame(width: 400)
    }

    // MARK: - Run History

    private var runHistoryView: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Run History")
                    .font(.caption.bold())
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(monitor.runHistory.count) runs")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(monitor.runHistory.reversed()) { record in
                        runRecordRow(record)
                    }
                }
            }
            .background(Color.black.opacity(0.15))
            .cornerRadius(6)
        }
    }

    private func runRecordRow(_ record: RunRecord) -> some View {
        HStack(spacing: 8) {
            // Status indicator
            Circle()
                .fill(statusColor(record.status))
                .frame(width: 6, height: 6)

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text(record.presetName)
                        .font(.system(size: 10, design: .monospaced).bold())
                    Text(record.gridDescription)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                Text(record.dateString)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Peak accuracy
            VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.1f%%", record.peakAccuracy * 100))
                    .font(.system(size: 11, design: .monospaced).bold())
                    .foregroundColor(record.peakAccuracy > 0.7 ? .green : .primary)
                Text(record.durationString)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    private func statusColor(_ status: RunRecord.RunStatus) -> Color {
        switch status {
        case .running: return .green
        case .completed: return .blue
        case .stopped: return .gray
        }
    }

    private func metricPill(_ label: String, _ value: String) -> some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.body, design: .monospaced).bold())
                .foregroundColor(.primary)
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.secondary)
        }
    }

    /// Metric pill showing both raw and smoothed (trend) values
    private func metricPillDual(_ label: String, raw: String, smoothed: String) -> some View {
        VStack(spacing: 1) {
            HStack(spacing: 4) {
                Text(smoothed)
                    .font(.system(.body, design: .monospaced).bold())
                    .foregroundColor(.primary)
                Text(raw)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            HStack(spacing: 2) {
                Text(label)
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                Text("(trend | raw)")
                    .font(.system(size: 7))
                    .foregroundColor(.secondary.opacity(0.6))
            }
        }
    }

    private func formatETA(_ seconds: TimeInterval) -> String {
        if seconds <= 0 { return "--" }
        let h = Int(seconds) / 3600
        let m = Int(seconds) % 3600 / 60
        if h > 0 { return "\(h)h \(m)m" }
        return "\(m)m"
    }

    // MARK: - Experiment info panel

    private var experimentInfoPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "flask.fill")
                    .foregroundColor(.purple)
                    .font(.caption)
                Text(monitor.experimentTitle)
                    .font(.caption.bold())
                    .foregroundColor(.purple)
                Spacer()
            }

            if !monitor.experimentDescription.isEmpty {
                Text(monitor.experimentDescription)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.primary.opacity(0.8))
                    .lineLimit(3)
            }

            if !monitor.experimentMethod.isEmpty {
                HStack(spacing: 4) {
                    Text("Method:")
                        .font(.system(size: 9).bold())
                        .foregroundColor(.secondary)
                    Text(monitor.experimentMethod)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }

            if !monitor.experimentFindings.isEmpty {
                HStack(alignment: .top, spacing: 4) {
                    Text("Findings:")
                        .font(.system(size: 9).bold())
                        .foregroundColor(.green.opacity(0.8))
                    Text(monitor.experimentFindings)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.green.opacity(0.8))
                        .lineLimit(4)
                }
            }
        }
        .padding(8)
        .background(Color.purple.opacity(0.08))
        .cornerRadius(6)
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.purple.opacity(0.2), lineWidth: 1)
        )
    }

    // MARK: - Log view

    private var logView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Log").font(.caption).foregroundColor(.secondary)
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(monitor.logMessages.enumerated()), id: \.offset) { idx, msg in
                            Text(msg)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.secondary)
                                .id(idx)
                        }
                    }
                }
                .onChange(of: monitor.logMessages.count) {
                    if let last = monitor.logMessages.indices.last {
                        proxy.scrollTo(last, anchor: .bottom)
                    }
                }
            }
            .background(Color.black.opacity(0.2))
            .cornerRadius(6)
        }
    }
}
