import SwiftUI

struct SettingsView: View {
    @AppStorage("maxHistoryItems") var maxHistoryItems: Int = 100
    @AppStorage("maxCharactersPerItem") var maxCharactersPerItem: Int = 20_000
    @AppStorage("showClippyCharacter") var showClippyCharacter: Bool = true
    @AppStorage("pollInterval") var pollInterval: Double = 0.5
    @AppStorage("playSounds") var playSounds: Bool = true
    @AppStorage("captureText") var captureText: Bool = true
    @AppStorage("captureRichText") var captureRichText: Bool = true
    @AppStorage("captureImages") var captureImages: Bool = true
    @AppStorage("captureFiles") var captureFiles: Bool = true
    @AppStorage("showMenuBarIcon") var showMenuBarIcon: Bool = true
    @AppStorage("clippyScale") var clippyScale: Double = 1.0
    @AppStorage(AppConstants.appFontNameKey) var appFontName: String = AppConstants.defaultFontName
    @AppStorage(AppConstants.appFontSizeKey) var appFontSize: Double = AppConstants.defaultFontSize

    @State private var excludedApps: [String] = UserDefaults.standard.stringArray(forKey: "excludedApps") ?? []
    @State private var newAppBundleID: String = ""

    var body: some View {
        Form {
            Section("General") {
                Toggle("Show menu bar icon", isOn: $showMenuBarIcon)
            }

            Section("Clipboard") {
                HStack {
                    Text("History limit:")
                    Spacer()
                    Picker("", selection: $maxHistoryItems) {
                        Text("25").tag(25)
                        Text("50").tag(50)
                        Text("100").tag(100)
                        Text("200").tag(200)
                        Text("500").tag(500)
                    }
                    .frame(width: 100)
                }

                HStack {
                    Text("Max chars per item:")
                    Spacer()
                    Picker("", selection: $maxCharactersPerItem) {
                        Text("5,000").tag(5_000)
                        Text("10,000").tag(10_000)
                        Text("20,000").tag(20_000)
                        Text("50,000").tag(50_000)
                    }
                    .frame(width: 100)
                }

                HStack {
                    Text("Check interval:")
                    Spacer()
                    Picker("", selection: $pollInterval) {
                        Text("0.25s").tag(0.25)
                        Text("0.5s").tag(0.5)
                        Text("1.0s").tag(1.0)
                        Text("2.0s").tag(2.0)
                    }
                    .frame(width: 100)
                }
            }

            Section("Appearance") {
                HStack {
                    Text("Font:")
                    Spacer()
                    Picker("", selection: $appFontName) {
                        Text("System").tag("System")
                        ForEach(NSFontManager.shared.availableFontFamilies.sorted(), id: \.self) { name in
                            Text(name).tag(name)
                        }
                    }
                    .frame(width: 160)
                }

                HStack {
                    Text("Font size:")
                    Spacer()
                    Picker("", selection: $appFontSize) {
                        Text("10").tag(10.0)
                        Text("11").tag(11.0)
                        Text("12").tag(12.0)
                        Text("13").tag(13.0)
                        Text("14").tag(14.0)
                        Text("16").tag(16.0)
                    }
                    .frame(width: 100)
                }

            }

            Section("Clippy Character") {
                Toggle("Show Clippy", isOn: $showClippyCharacter)
                Toggle("Play sounds", isOn: $playSounds)
                HStack {
                    Text("Character size:")
                    Spacer()
                    Picker("", selection: $clippyScale) {
                        Text("XS").tag(0.5)
                        Text("S").tag(0.75)
                        Text("M").tag(1.0)
                        Text("L").tag(1.25)
                        Text("XL").tag(1.5)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 180)
                }
            }

            Section("Advanced") {
                DisclosureGroup("Capture Types") {
                    Toggle("Plain text", isOn: $captureText)
                    Toggle("Rich text", isOn: $captureRichText)
                    Toggle("Images & screenshots", isOn: $captureImages)
                    Toggle("Files", isOn: $captureFiles)
                }

                DisclosureGroup("Excluded Apps") {
                    if excludedApps.isEmpty {
                        Text("No excluded apps")
                            .foregroundColor(.secondary)
                            .italic()
                    } else {
                        ForEach(excludedApps, id: \.self) { app in
                            HStack {
                                Text(displayName(for: app))
                                    .lineLimit(1)
                                Spacer()
                                Button(action: { removeExcludedApp(app) }) {
                                    Image(systemName: "minus.circle.fill")
                                        .foregroundColor(.red.opacity(0.7))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    HStack(spacing: 4) {
                        TextField("com.example.app", text: $newAppBundleID)
                            .textFieldStyle(.roundedBorder)
                        Button(action: addExcludedApp) {
                            Image(systemName: "plus.circle.fill")
                                .foregroundColor(.accentColor)
                        }
                        .buttonStyle(.plain)
                        .disabled(newAppBundleID.trimmingCharacters(in: .whitespaces).isEmpty)
                    }

                    Menu {
                        ForEach(runningAppBundleIDs(), id: \.self) { bundleID in
                            Button(displayName(for: bundleID)) {
                                if !excludedApps.contains(bundleID) {
                                    excludedApps.append(bundleID)
                                    saveExcludedApps()
                                }
                            }
                        }
                    } label: {
                        Label("Add from running apps", systemImage: "app.badge.checkmark")
                    }

                    Menu {
                        ForEach(installedAppBundleIDs(), id: \.self) { bundleID in
                            Button(displayName(for: bundleID)) {
                                if !excludedApps.contains(bundleID) {
                                    excludedApps.append(bundleID)
                                    saveExcludedApps()
                                }
                            }
                        }
                    } label: {
                        Label("Add from installed apps", systemImage: "app.badge")
                    }
                }
            }

            Section {
                HStack {
                    Spacer()
                    Button("Quit Clippy") {
                        NSApplication.shared.terminate(nil)
                    }
                    Spacer()
                }
            }

            Section {
                VStack(spacing: 2) {
                    Text("Clippy v1.0.0")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                    Text("\u{00A9} 2026 SNRGY Studios AB")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary.opacity(0.6))
                    Text("Made by Viktor Pantazis")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary.opacity(0.6))
                }
                .frame(maxWidth: .infinity)
            }
        }
        .formStyle(.grouped)
        .frame(width: 400, height: 680)
    }

    private func addExcludedApp() {
        let bundleID = newAppBundleID.trimmingCharacters(in: .whitespaces)
        guard !bundleID.isEmpty, !excludedApps.contains(bundleID) else { return }
        excludedApps.append(bundleID)
        newAppBundleID = ""
        saveExcludedApps()
    }

    private func removeExcludedApp(_ app: String) {
        excludedApps.removeAll { $0 == app }
        saveExcludedApps()
    }

    private func saveExcludedApps() {
        UserDefaults.standard.set(excludedApps, forKey: "excludedApps")
    }

    private func runningAppBundleIDs() -> [String] {
        NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular }
            .compactMap { $0.bundleIdentifier }
            .filter { !excludedApps.contains($0) && $0 != Bundle.main.bundleIdentifier }
            .sorted()
    }

    private func installedAppBundleIDs() -> [String] {
        let appDirs = ["/Applications", "/System/Applications"]
        var bundleIDs: [String] = []
        let fm = FileManager.default
        for dir in appDirs {
            guard let contents = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for item in contents where item.hasSuffix(".app") {
                let path = "\(dir)/\(item)"
                if let bundle = Bundle(path: path), let id = bundle.bundleIdentifier {
                    if !excludedApps.contains(id) && id != Bundle.main.bundleIdentifier {
                        bundleIDs.append(id)
                    }
                }
            }
        }
        return bundleIDs.sorted()
    }

    private func displayName(for bundleID: String) -> String {
        if let app = NSWorkspace.shared.runningApplications.first(where: { $0.bundleIdentifier == bundleID }) {
            return app.localizedName ?? bundleID
        }
        let appDirs = ["/Applications", "/System/Applications"]
        for dir in appDirs {
            guard let contents = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
            for item in contents where item.hasSuffix(".app") {
                let path = "\(dir)/\(item)"
                if let bundle = Bundle(path: path), bundle.bundleIdentifier == bundleID {
                    return item.replacingOccurrences(of: ".app", with: "")
                }
            }
        }
        let components = bundleID.split(separator: ".")
        if let last = components.last {
            return String(last)
        }
        return bundleID
    }
}
