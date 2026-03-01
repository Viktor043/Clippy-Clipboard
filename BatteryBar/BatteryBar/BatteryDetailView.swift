import SwiftUI

/// The dropdown panel that shows detailed battery information.
struct BatteryDetailView: View {
    @ObservedObject var batteryService: BatteryService

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header
            HStack {
                Image(systemName: headerIcon)
                    .font(.system(size: 22))
                    .foregroundColor(headerColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(info.percentage)%")
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                    Text(statusText)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                Spacer()
            }

            Divider()

            // Details grid
            DetailRow(label: "Status", value: statusText)

            if let time = info.timeRemaining {
                DetailRow(label: "Time Remaining", value: formatMinutes(time))
            }
            if let time = info.timeToFullCharge {
                DetailRow(label: "Time to Full", value: formatMinutes(time))
            }
            if let health = info.health {
                DetailRow(label: "Battery Health", value: "\(health)%")
            }
            if let cycles = info.cycleCount {
                DetailRow(label: "Cycle Count", value: "\(cycles)")
            }
            if let temp = info.temperature {
                DetailRow(label: "Temperature", value: String(format: "%.1f °C", temp))
            }
            if let watts = info.wattage {
                DetailRow(label: "Power", value: String(format: "%.1f W", watts))
            }
            if let voltage = info.voltage {
                DetailRow(label: "Voltage", value: String(format: "%.2f V", voltage))
            }
            if let current = info.maxCapacity, let design = info.designCapacity {
                DetailRow(label: "Capacity", value: "\(current) / \(design) mAh")
            }

            Divider()

            HStack {
                Text("BatteryBar")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                Spacer()
                Button("Quit") {
                    NSApplication.shared.terminate(nil)
                }
                .buttonStyle(.plain)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
            }
        }
        .padding(14)
        .frame(width: 260)
    }

    // MARK: - Helpers

    private var info: BatteryInfo { batteryService.batteryInfo }

    private var statusText: String {
        if info.isFullyCharged { return "Fully Charged" }
        if info.isCharging { return "Charging" }
        if info.isPluggedIn { return "Plugged In, Not Charging" }
        return "On Battery"
    }

    private var headerIcon: String {
        if info.isCharging || info.isFullyCharged {
            return "battery.100.bolt"
        }
        switch info.percentage {
        case 0...10:  return "battery.0"
        case 11...35: return "battery.25"
        case 36...65: return "battery.50"
        case 66...90: return "battery.75"
        default:      return "battery.100"
        }
    }

    private var headerColor: Color {
        if info.isCharging { return .green }
        if info.percentage <= 10 { return .red }
        if info.percentage <= 20 { return .orange }
        return .primary
    }

    private func formatMinutes(_ minutes: Int) -> String {
        let h = minutes / 60
        let m = minutes % 60
        if h > 0 {
            return "\(h)h \(m)m"
        }
        return "\(m) min"
    }
}

/// A single label-value row in the detail panel.
private struct DetailRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
        }
    }
}
