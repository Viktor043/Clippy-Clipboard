import SwiftUI
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("[PhotonicTwin] applicationDidFinishLaunching")
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        // Force main window to appear even in background mode
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            if let window = NSApp.windows.first {
                window.makeKeyAndOrderFront(nil)
                print("[PhotonicTwin] Forced window to front")
            }
        }
    }
}

@main
struct PhotonicTwinApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var monitor = TrainingMonitor()
    @State private var engine: MetalEngine?
    @State private var trainingLoop: TrainingLoop?
    @State private var statusServer: StatusServer?
    @State private var currentLiveConfig: LiveConfig?

    var body: some Scene {
        WindowGroup("Photonic Twin") {
            DashboardView(
                monitor: monitor,
                onStart: startTraining,
                onPause: pauseTraining,
                onNewRun: startNewRun
            )
            .frame(minWidth: 900, minHeight: 600)
            .onAppear { setupAndStart() }
        }
    }

    private func setupAndStart() {
        print("[PhotonicTwin] setupAndStart called")

        // Start LAN status server (persists across runs)
        // Port configurable via PHOTONIC_PORT env var (default 8090)
        // This allows running parallel experiments on different ports
        if statusServer == nil {
            let port: UInt16
            if let envPort = ProcessInfo.processInfo.environment["PHOTONIC_PORT"],
               let p = UInt16(envPort) {
                port = p
            } else {
                port = 8090
            }
            let server = StatusServer(port: port, monitor: monitor)
            server.start()
            self.statusServer = server
            print("[PhotonicTwin] Status server started on port \(port)")

            let ips = StatusServer.localIPAddresses()
            let addr = ips.first.map { "http://\($0):\(port)" } ?? "http://localhost:\(port)"
            Task { @MainActor in
                monitor.lanAddress = addr
                monitor.log("LAN dashboard: \(addr)")
            }
        }

        // Select config: env var > command line arg > ~/.photonic_preset file > default
        // Priority: PHOTONIC_PRESET env var wins (enables parallel experiments with different presets)
        // 1=quick, 2=medium, 3=large, 4=max, 5=saturable, 6=recurrent3, 7=recurrentSaturable, 8=debug
        // 9=d2nnDebug, 10=d2nn, 11=d2nnDifferential, 12=d2nnDeep, 13=d2nn128
        // 14=uclaReplication (128×128 continuous), 15=uclaReplicationSmall (64×64 continuous)
        // Set via: PHOTONIC_PRESET=10 PHOTONIC_PORT=8091 .build/debug/PhotonicTwin &
        let preset: Int
        if let envVal = ProcessInfo.processInfo.environment["PHOTONIC_PRESET"],
           let p = Int(envVal) {
            preset = p
        } else {
            let args = CommandLine.arguments
            if args.count > 1, let p = Int(args[1]) {
                preset = p
            } else {
                let presetFile = NSHomeDirectory() + "/.photonic_preset"
                if let fileContent = try? String(contentsOfFile: presetFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
                   let p = Int(fileContent) {
                    preset = p
                } else {
                    preset = 2
                }
            }
        }

        let config: SimConfig
        let presetName: String
        switch preset {
        case 1: config = .quick; presetName = "quick"
        case 2: config = .medium; presetName = "medium"
        case 3: config = .large; presetName = "large"
        case 4: config = .maxQuality; presetName = "maxQuality"
        case 5: config = .saturableAbsorption; presetName = "saturableAbsorption"
        case 6: config = .recurrent3Pulse; presetName = "recurrent3Pulse"
        case 7: config = .recurrentSaturable; presetName = "recurrentSaturable"
        case 8: config = .debug; presetName = "debug"
        case 9: config = .d2nnDebug; presetName = "d2nnDebug"
        case 10: config = .d2nn; presetName = "d2nn"
        case 11: config = .d2nnDifferential; presetName = "d2nnDifferential"
        case 12: config = .d2nnDeep; presetName = "d2nnDeep"
        case 13: config = .d2nn128; presetName = "d2nn128"
        case 14: config = .uclaReplication; presetName = "uclaReplication"
        case 15: config = .uclaReplicationSmall; presetName = "uclaReplicationSmall"
        case 16: config = .discreteD2NN; presetName = "discreteD2NN"
        case 17: config = .discreteD2NNSmall; presetName = "discreteD2NNSmall"
        case 18: config = .discreteD2NNLarge; presetName = "discreteD2NNLarge"
        case 19: config = .mnistD2NN; presetName = "mnistD2NN"
        case 20: config = .mnistD2NNSmall; presetName = "mnistD2NNSmall"
        case 21: config = .mnistD2NNLarge; presetName = "mnistD2NNLarge"
        case 22: config = .mnist10Class; presetName = "mnist10Class"
        case 23: config = .mnist2ClassDebug; presetName = "mnist2ClassDebug"
        case 24: config = .mnist2ClassRandom; presetName = "mnist2ClassRandom"
        case 25: config = .mnist2Class1Layer; presetName = "mnist2Class1Layer"
        case 26: config = .mnist10ClassSmall; presetName = "mnist10ClassSmall"
        case 27: config = .mnist10ClassDiff; presetName = "mnist10ClassDiff"
        case 28: config = .mnist10ClassBatch128; presetName = "mnist10ClassBatch128"
        case 29: config = .mnist10ClassPhase; presetName = "mnist10ClassPhase"
        case 30: config = .mnist10ClassLargeDet; presetName = "mnist10ClassLargeDet"
        case 31: config = .mnist10ClassSmallDiff; presetName = "mnist10ClassSmallDiff"
        case 32: config = .opticalFFT; presetName = "opticalFFT"
        case 33: config = .opticalFFTSmall; presetName = "opticalFFTSmall"
        case 34: config = .fashion10ClassPhase; presetName = "fashion10ClassPhase"
        case 35: config = .fashion10Class; presetName = "fashion10Class"
        case 36: config = .opticalFFTTiny; presetName = "opticalFFTTiny"
        case 37: config = .opticalFFTMedium; presetName = "opticalFFTMedium"
        case 38: config = .opticalFFT4x4; presetName = "opticalFFT4x4"
        case 39: config = .opticalFFT8x8; presetName = "opticalFFT8x8"
        case 40: config = .opticalFFTTinyDeep; presetName = "opticalFFTTinyDeep"
        case 41: config = .opticalFFT32x32Deep; presetName = "opticalFFT32x32Deep"
        case 42: config = .opticalFFTTinyUltra; presetName = "opticalFFTTinyUltra"
        case 43: config = .matrixUnitary8x8; presetName = "matrixUnitary8x8"
        case 44: config = .matrixHadamard8x8; presetName = "matrixHadamard8x8"
        case 45: config = .matrixIdentity8x8; presetName = "matrixIdentity8x8"
        case 46: config = .matrixUnitary16x16; presetName = "matrixUnitary16x16"
        case 47: config = .matrixHadamard16x16; presetName = "matrixHadamard16x16"
        case 48: config = .tinyStoriesWV; presetName = "tinyStoriesWV"
        case 49: config = .tinyStoriesWQ; presetName = "tinyStoriesWQ"
        case 50: config = .tinyStoriesWK; presetName = "tinyStoriesWK"
        case 51: config = .tinyStoriesWO; presetName = "tinyStoriesWO"
        case 52: config = .tinyStoriesWVRaw; presetName = "tinyStoriesWVRaw"
        case 53: config = .matrixUnitary64x64; presetName = "matrixUnitary64x64"
        case 54: config = .matrixTinyStories64x64; presetName = "matrixTinyStories64x64"
        case 55: config = .matrixUnitary64x64Shallow; presetName = "matrixUnitary64x64Shallow"
        case 56: config = .matrixUnitary64x64Tiny; presetName = "matrixUnitary64x64Tiny"
        default: config = .medium; presetName = "medium"
        }
        print("[PhotonicTwin] Preset: \(presetName)")

        launchRun(config: config, presetName: presetName)
    }

    /// Launch a new training run with the given config.
    /// Can be called multiple times to switch presets.
    private func launchRun(config: SimConfig, presetName: String) {
        // Stop current training if running
        trainingLoop?.pause()

        // Brief delay to let the training loop stop
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.3) {
            DispatchQueue.main.async {
                self.monitor.finishCurrentRun(status: .stopped)
                self.monitor.resetForNewRun()

                do {
                    print("[PhotonicTwin] Creating MetalEngine for \(presetName)...")
                    let eng = try MetalEngine(config: config)
                    self.engine = eng
                    print("[PhotonicTwin] MetalEngine created: \(eng.device.name)")

                    self.monitor.log("Device: \(eng.device.name)")
                    self.monitor.startRun(presetName: presetName, config: config)

                    let lc = LiveConfig(config)
                    self.currentLiveConfig = lc
                    // Wire LiveConfig + engine into StatusServer for HTTP API access
                    self.statusServer?.liveConfig = lc
                    self.statusServer?.engine = eng

                    let loop = TrainingLoop(engine: eng, config: config, monitor: self.monitor, liveConfig: lc)
                    self.trainingLoop = loop
                    loop.start()
                    print("[PhotonicTwin] Training started: \(presetName)")
                } catch {
                    print("[PhotonicTwin] Setup failed: \(error)")
                    self.monitor.log("Setup failed: \(error)")
                }
            }
        }
    }

    private func startTraining() {
        trainingLoop?.start(fromStep: monitor.currentStep)
    }

    private func pauseTraining() {
        trainingLoop?.pause()
    }

    /// Called from the UI when user selects a new preset
    private func startNewRun(presetInfo: PresetInfo) {
        print("[PhotonicTwin] Switching to preset: \(presetInfo.name)")
        launchRun(config: presetInfo.config, presetName: presetInfo.name)
    }
}
