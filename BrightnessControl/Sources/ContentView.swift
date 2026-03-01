import SwiftUI

struct ContentView: View {
    @ObservedObject var screenManager: ScreenBrightnessManager
    @ObservedObject var keyboardManager: KeyboardBrightnessManager

    var body: some View {
        VStack(spacing: 14) {
            // Header
            HStack(spacing: 6) {
                Image(systemName: "sun.min.fill")
                    .foregroundColor(.orange)
                Text("Brightness Control")
                    .font(.headline)
            }
            .padding(.top, 4)

            Divider()

            // MARK: - Display Section
            VStack(alignment: .leading, spacing: 10) {
                Label("Display", systemImage: "display")
                    .font(.subheadline.bold())

                // Hardware backlight
                VStack(alignment: .leading, spacing: 4) {
                    Text("Backlight")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(spacing: 8) {
                        Image(systemName: "sun.max")
                            .frame(width: 14)
                            .foregroundColor(.secondary)
                            .font(.caption)
                        Slider(value: $screenManager.hardwareBrightness, in: 0...1)
                            .onChange(of: screenManager.hardwareBrightness) { _ in
                                screenManager.applyHardwareBrightness()
                            }
                        Text("\(Int(screenManager.hardwareBrightness * 100))%")
                            .frame(width: 34, alignment: .trailing)
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                    if !screenManager.hardwareControlAvailable {
                        HStack(spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.caption2)
                            Text("Hardware control unavailable — use software dimming")
                                .font(.caption2)
                        }
                        .foregroundColor(.orange)
                    }
                }

                // Software dimming (gamma)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Software Dimming")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(spacing: 8) {
                        Image(systemName: "circle.lefthalf.filled")
                            .frame(width: 14)
                            .foregroundColor(.secondary)
                            .font(.caption)
                        Slider(value: $screenManager.softwareDimming, in: 0.01...1)
                            .onChange(of: screenManager.softwareDimming) { _ in
                                screenManager.applySoftwareDimming()
                            }
                        Text("\(Int(screenManager.softwareDimming * 100))%")
                            .frame(width: 34, alignment: .trailing)
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                    Text("Goes below hardware minimum via gamma overlay")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            Divider()

            // MARK: - Keyboard Section
            VStack(alignment: .leading, spacing: 10) {
                Label("Keyboard Backlight", systemImage: "keyboard")
                    .font(.subheadline.bold())

                // Base brightness
                VStack(alignment: .leading, spacing: 4) {
                    Text("Backlight")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(spacing: 8) {
                        Image(systemName: "light.max")
                            .frame(width: 14)
                            .foregroundColor(.secondary)
                            .font(.caption)
                        Slider(value: $keyboardManager.brightness, in: 0...1)
                            .disabled(!keyboardManager.isAvailable)
                            .onChange(of: keyboardManager.brightness) { _ in
                                keyboardManager.applyBrightness()
                            }
                        Text("\(Int(keyboardManager.brightness * 100))%")
                            .frame(width: 34, alignment: .trailing)
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                }

                // Software reduction
                VStack(alignment: .leading, spacing: 4) {
                    Text("Software Reduction")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(spacing: 8) {
                        Image(systemName: "light.min")
                            .frame(width: 14)
                            .foregroundColor(.secondary)
                            .font(.caption)
                        Slider(value: $keyboardManager.reduction, in: 0.01...1)
                            .disabled(!keyboardManager.isAvailable)
                            .onChange(of: keyboardManager.reduction) { _ in
                                keyboardManager.applyBrightness()
                            }
                        Text("\(Int(keyboardManager.reduction * 100))%")
                            .frame(width: 34, alignment: .trailing)
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                    let eff = keyboardManager.effectiveBrightness
                    Text("Effective: \(String(format: eff < 0.01 ? "%.2f" : "%.1f", eff * 100))% — multiplies backlight for ultra-low levels")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                if !keyboardManager.isAvailable {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.caption2)
                        Text("Keyboard backlight not detected")
                            .font(.caption2)
                    }
                    .foregroundColor(.orange)
                }
            }

            Divider()

            // MARK: - Actions
            HStack {
                Button("Reset All") {
                    screenManager.reset()
                    keyboardManager.reset()
                }

                Spacer()

                Button("Quit") {
                    screenManager.resetGamma()
                    NSApp.terminate(nil)
                }
            }
            .font(.caption)
        }
        .padding(16)
        .frame(width: 300)
    }
}
