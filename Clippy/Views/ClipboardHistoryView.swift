import SwiftUI
import AppKit

struct ClipboardHistoryView: View {
    @ObservedObject var store: ClipboardStore
    @State private var searchText = ""
    @State private var showSettings = false
    @State private var showClearConfirm = false
    @State private var hostWindow: NSWindow?
    @AppStorage(AppConstants.appFontNameKey) private var appFontName: String = AppConstants.defaultFontName
    @AppStorage(AppConstants.appFontSizeKey) private var appFontSize: Double = AppConstants.defaultFontSize
    @AppStorage(AppConstants.popoverWidthKey) private var popoverWidth: Double = Double(AppConstants.defaultPopoverWidth)
    @AppStorage(AppConstants.popoverHeightKey) private var popoverHeight: Double = Double(AppConstants.defaultPopoverHeight)
    var onPaste: (() -> Void)?

    private var appFont: Font {
        if appFontName == "System" || appFontName.isEmpty {
            return .system(size: appFontSize)
        } else {
            return .custom(appFontName, size: appFontSize)
        }
    }

    var filteredItems: [ClipboardItem] {
        if searchText.isEmpty {
            return store.items
        }
        return store.items.filter { item in
            if let text = item.textContent {
                return text.localizedCaseInsensitiveContains(searchText)
            }
            return false
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 8) {
                if showSettings {
                    Button(action: {
                        // Resign any focused text field first so this click always triggers
                        NSApp.keyWindow?.makeFirstResponder(nil)
                        withAnimation(.easeInOut(duration: 0.2)) { showSettings = false }
                    }) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundColor(.accentColor)
                            .frame(width: 24, height: 24)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .focusEffectDisabled()
                }

                Text(showSettings ? "Settings" : "Clippy")
                    .font(.headline)

                Spacer()

                if !showSettings {
                    Button(action: { showClearConfirm = true }) {
                        Image(systemName: "trash")
                            .font(.system(size: 14))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .focusEffectDisabled()
                    .help("Clear all history")
                    .alert("Clear clipboard history?", isPresented: $showClearConfirm) {
                        Button("Clear All", role: .destructive) { store.clearAll() }
                        Button("Cancel", role: .cancel) { }
                    } message: {
                        Text("This will remove all saved clipboard items.")
                    }

                    Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showSettings = true } }) {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.primary.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                    .help("Settings")
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            // Content area with smooth slide transitions
            ZStack {
                if showSettings {
                    PopoverSettingsView()
                        .transition(.asymmetric(
                            insertion: .move(edge: .trailing).combined(with: .opacity),
                            removal: .move(edge: .trailing).combined(with: .opacity)
                        ))
                } else {
                    clipboardContent
                        .transition(.asymmetric(
                            insertion: .move(edge: .leading).combined(with: .opacity),
                            removal: .move(edge: .leading).combined(with: .opacity)
                        ))
                }
            }
            .clipped()
        }
        .frame(
            width: CGFloat(popoverWidth),
            height: CGFloat(popoverHeight)
        )
        .background(WindowAccessor(window: $hostWindow))
        .onDisappear {
            DetailPanelController.shared.hide()
        }
    }

    @ViewBuilder
    private var clipboardContent: some View {
        VStack(spacing: 0) {
            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Search clipboard...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(appFont)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)

            Divider()

            // List
            if filteredItems.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "clipboard")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary)
                    Text(searchText.isEmpty ? "No clipboard history" : "No matching items")
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 1) {
                        ForEach(filteredItems) { item in
                            ClipboardItemRow(item: item) {
                                pasteItem(item)
                            } onDelete: {
                                store.deleteItem(id: item.id)
                            } onExpand: {
                                showDetailPanel(for: item)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private func showDetailPanel(for item: ClipboardItem) {
        // Use the reliably-captured host window reference
        guard let window = hostWindow else { return }
        DetailPanelController.shared.show(item: item, relativeTo: window)
    }

    private func pasteItem(_ item: ClipboardItem) {
        // Close detail panel
        DetailPanelController.shared.hide()

        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()

        // Set self-copy marker to prevent re-recording
        pasteboard.setData(Data(), forType: NSPasteboard.PasteboardType(AppConstants.selfCopyPasteboardType))

        switch item.type {
        case .text:
            if let text = item.textContent {
                pasteboard.setString(text, forType: .string)
            }
        case .richText:
            if let data = item.rawData {
                pasteboard.setData(data, forType: .rtf)
            }
            if let text = item.textContent {
                pasteboard.setString(text, forType: .string)
            }
        case .image:
            if let data = item.imageData {
                pasteboard.setData(data, forType: .png)
            }
        case .file:
            if let path = item.textContent {
                let url = URL(fileURLWithPath: path)
                pasteboard.writeObjects([url as NSURL])
            }
        }

        onPaste?()
    }
}

// MARK: - Inline Settings View for Popover

struct PopoverSettingsView: View {
    @AppStorage("maxHistoryItems") var maxHistoryItems: Int = 100
    @AppStorage("maxCharactersPerItem") var maxCharactersPerItem: Int = 20_000
    @AppStorage("showClippyCharacter") var showClippyCharacter: Bool = true
    @AppStorage("playSounds") var playSounds: Bool = true
    @AppStorage("captureText") var captureText: Bool = true
    @AppStorage("captureRichText") var captureRichText: Bool = true
    @AppStorage("captureImages") var captureImages: Bool = true
    @AppStorage("captureFiles") var captureFiles: Bool = true
    @AppStorage("showMenuBarIcon") var showMenuBarIcon: Bool = true
    @AppStorage("clippyScale") var clippyScale: Double = 1.0
    @AppStorage(AppConstants.appFontNameKey) var appFontName: String = AppConstants.defaultFontName
    @AppStorage(AppConstants.appFontSizeKey) var appFontSize: Double = AppConstants.defaultFontSize
    @AppStorage(AppConstants.popoverWidthKey) var popoverWidth: Double = Double(AppConstants.defaultPopoverWidth)
    @AppStorage(AppConstants.popoverHeightKey) var popoverHeight: Double = Double(AppConstants.defaultPopoverHeight)
    @AppStorage(AppConstants.panelOpacityKey) var panelOpacity: Double = AppConstants.defaultPanelOpacity
    @AppStorage(AppConstants.panelBlurKey) var panelBlurIntensity: Double = AppConstants.defaultPanelBlur

    @State private var excludedApps: [String] = UserDefaults.standard.stringArray(forKey: "excludedApps") ?? []
    @State private var newAppBundleID: String = ""

    private static let availableFonts: [String] = {
        var fonts = ["System"]
        fonts.append(contentsOf: NSFontManager.shared.availableFontFamilies.sorted())
        return fonts
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // General
                VStack(alignment: .leading, spacing: 8) {
                    Label("General", systemImage: "gearshape")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)

                    Toggle("Show menu bar icon", isOn: $showMenuBarIcon)
                        .font(.system(size: 12))
                }
                .toggleStyle(.checkbox)
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(8)

                // Menus
                VStack(alignment: .leading, spacing: 8) {
                    Label("Menus", systemImage: "rectangle.on.rectangle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)

                    HStack {
                        Text("Width")
                            .font(.system(size: 12))
                        Spacer()
                        Text("\(Int(popoverWidth)) px")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(width: 45, alignment: .trailing)
                    }
                    Slider(value: $popoverWidth,
                           in: Double(AppConstants.minPopoverWidth)...Double(AppConstants.maxPopoverWidth),
                           step: 10)

                    HStack {
                        Text("Height")
                            .font(.system(size: 12))
                        Spacer()
                        Text("\(Int(popoverHeight)) px")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(width: 45, alignment: .trailing)
                    }
                    Slider(value: $popoverHeight,
                           in: Double(AppConstants.minPopoverHeight)...Double(AppConstants.maxPopoverHeight),
                           step: 10)

                    Divider()

                    HStack {
                        Text("Transparency")
                            .font(.system(size: 12))
                        Spacer()
                        Text("\(Int(panelOpacity * 100))%")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(width: 35, alignment: .trailing)
                    }
                    Slider(value: $panelOpacity,
                           in: AppConstants.minPanelOpacity...AppConstants.maxPanelOpacity,
                           step: 0.05)

                    HStack {
                        Text("Background blur")
                            .font(.system(size: 12))
                        Spacer()
                        Text("\(Int(panelBlurIntensity * 100))%")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .frame(width: 35, alignment: .trailing)
                    }
                    Slider(value: $panelBlurIntensity,
                           in: AppConstants.minPanelBlur...AppConstants.maxPanelBlur,
                           step: 0.05)
                }
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(8)

                // Appearance
                VStack(alignment: .leading, spacing: 8) {
                    Label("Appearance", systemImage: "textformat")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)

                    HStack {
                        Text("Font")
                            .font(.system(size: 12))
                        Spacer()
                        Picker("", selection: $appFontName) {
                            ForEach(Self.availableFonts, id: \.self) { name in
                                Text(name).tag(name)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 140)
                    }

                    HStack {
                        Text("Font size")
                            .font(.system(size: 12))
                        Spacer()
                        Picker("", selection: $appFontSize) {
                            Text("10").tag(10.0)
                            Text("11").tag(11.0)
                            Text("12").tag(12.0)
                            Text("13").tag(13.0)
                            Text("14").tag(14.0)
                            Text("16").tag(16.0)
                        }
                        .labelsHidden()
                        .frame(width: 80)
                    }
                }
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(8)

                // Clippy Character
                VStack(alignment: .leading, spacing: 8) {
                    Label("Clippy Character", systemImage: "figure.wave")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)

                    Toggle("Show Clippy character", isOn: $showClippyCharacter)
                        .font(.system(size: 12))
                    Toggle("Play sounds", isOn: $playSounds)
                        .font(.system(size: 12))

                    HStack {
                        Text("Size")
                            .font(.system(size: 12))
                        Spacer()
                        Picker("", selection: $clippyScale) {
                            Text("XS").tag(0.5)
                            Text("S").tag(0.75)
                            Text("M").tag(1.0)
                            Text("L").tag(1.25)
                            Text("XL").tag(1.5)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(width: 180)
                    }
                }
                .toggleStyle(.checkbox)
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(8)

                // Advanced
                VStack(alignment: .leading, spacing: 8) {
                    Label("Advanced", systemImage: "wrench.and.screwdriver")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)

                    // History limit
                    HStack {
                        Text("Remember up to")
                            .font(.system(size: 12))
                        Spacer()
                        Picker("", selection: $maxHistoryItems) {
                            Text("25").tag(25)
                            Text("50").tag(50)
                            Text("100").tag(100)
                            Text("200").tag(200)
                            Text("500").tag(500)
                        }
                        .labelsHidden()
                        .frame(width: 80)
                        Text("items")
                            .font(.system(size: 12))
                    }

                    // Character limit per item
                    HStack {
                        Text("Max chars per item")
                            .font(.system(size: 12))
                        Spacer()
                        Picker("", selection: $maxCharactersPerItem) {
                            Text("5K").tag(5_000)
                            Text("10K").tag(10_000)
                            Text("20K").tag(20_000)
                            Text("50K").tag(50_000)
                        }
                        .labelsHidden()
                        .frame(width: 80)
                    }

                    Divider()

                    // Capture types
                    Text("Capture types:")
                        .font(.system(size: 12))

                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Plain text", isOn: $captureText)
                            .font(.system(size: 12))
                        Toggle("Rich text", isOn: $captureRichText)
                            .font(.system(size: 12))
                        Toggle("Images & screenshots", isOn: $captureImages)
                            .font(.system(size: 12))
                        Toggle("Files", isOn: $captureFiles)
                            .font(.system(size: 12))
                    }
                    .toggleStyle(.checkbox)

                    Divider()

                    // Excluded apps
                    Text("Excluded apps (no clipboard capture):")
                        .font(.system(size: 12))

                    if excludedApps.isEmpty {
                        Text("No excluded apps")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)
                            .italic()
                    } else {
                        ForEach(excludedApps, id: \.self) { app in
                            HStack {
                                Text(displayName(for: app))
                                    .font(.system(size: 11))
                                    .lineLimit(1)
                                Spacer()
                                Button(action: { removeExcludedApp(app) }) {
                                    Image(systemName: "minus.circle.fill")
                                        .font(.system(size: 12))
                                        .foregroundColor(.red.opacity(0.7))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }

                    HStack(spacing: 4) {
                        TextField("com.example.app", text: $newAppBundleID)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(size: 11))
                        Button(action: addExcludedApp) {
                            Image(systemName: "plus.circle.fill")
                                .font(.system(size: 14))
                                .foregroundColor(.accentColor)
                        }
                        .buttonStyle(.plain)
                        .disabled(newAppBundleID.trimmingCharacters(in: .whitespaces).isEmpty)
                    }

                    // Quick-add running apps
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
                            .font(.system(size: 11))
                    }
                    .menuStyle(.borderlessButton)

                    // Add from installed apps
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
                            .font(.system(size: 11))
                    }
                    .menuStyle(.borderlessButton)
                }
                .padding(10)
                .background(Color.primary.opacity(0.03))
                .cornerRadius(8)

                // How to Use
                TutorialSection()

                // Quit
                HStack {
                    Spacer()
                    Button("Quit Clippy") {
                        NSApplication.shared.terminate(nil)
                    }
                    .font(.system(size: 12))
                    Spacer()
                }

                // Copyright
                VStack(spacing: 2) {
                    Text("Clippy v1.0.0")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundColor(.secondary)
                    Text("\u{00A9} 2026 SNRGY Studios AB")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary.opacity(0.6))
                    Text("Made by Viktor Pantazis")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary.opacity(0.6))
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 4)
            }
            .padding(12)
        }
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

// MARK: - Tutorial Section

private struct TutorialSection: View {
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: { withAnimation(.easeInOut(duration: 0.2)) { isExpanded.toggle() } }) {
                HStack {
                    Label("How to Use", systemImage: "questionmark.circle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.secondary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                }
            }
            .buttonStyle(.plain)

            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    tutorialItem(
                        icon: "paperclip",
                        title: "Open Clipboard History",
                        text: "Click the paperclip in your menu bar or click the Clippy character."
                    )

                    tutorialItem(
                        icon: "doc.on.doc",
                        title: "Paste from History",
                        text: "Click any item to copy it back to your clipboard, then Cmd+V to paste."
                    )

                    tutorialItem(
                        icon: "magnifyingglass",
                        title: "Search",
                        text: "Type in the search bar to filter your clipboard history."
                    )

                    tutorialItem(
                        icon: "chevron.left",
                        title: "Expand & Preview",
                        text: "Hover over an item and click Expand to see the full content in a side panel."
                    )

                    tutorialItem(
                        icon: "cursorarrow.click.2",
                        title: "Multi-Select Text",
                        text: "In the preview panel, Cmd+click to select multiple text sections. Double-click to clear. Cmd+C to copy."
                    )

                    tutorialItem(
                        icon: "hand.draw",
                        title: "Move Clippy",
                        text: "Drag the character anywhere on screen. Right-click it to resize, lock position, or mute sounds."
                    )

                    tutorialItem(
                        icon: "lock.shield",
                        title: "Sensitive Apps",
                        text: "Add password managers to the excluded apps list above so Clippy never saves their clipboard."
                    )
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(10)
        .background(Color.primary.opacity(0.03))
        .cornerRadius(8)
    }

    private func tutorialItem(icon: String, title: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundColor(.accentColor)
                .frame(width: 16, alignment: .center)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 11, weight: .semibold))
                Text(text)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Window Accessor

/// Captures the hosting NSWindow for a SwiftUI view so we can pass it to DetailPanelController.
/// Using this eliminates fragile window-lookup heuristics.
private struct WindowAccessor: NSViewRepresentable {
    @Binding var window: NSWindow?

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async {
            self.window = view.window
        }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        DispatchQueue.main.async {
            if self.window !== nsView.window {
                self.window = nsView.window
            }
        }
    }
}
