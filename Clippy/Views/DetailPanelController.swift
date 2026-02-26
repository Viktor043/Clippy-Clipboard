import AppKit
import SwiftUI

/// Manages a floating NSPanel that shows a full-text preview to the left of a popover.
final class DetailPanelController {
    static let shared = DetailPanelController()

    private var panel: NSPanel?
    private var currentItemID: UUID?
    private var popoverWindowObserver: NSObjectProtocol?
    private weak var trackedPopoverWindow: NSWindow?

    /// When true, the detail panel is in "standalone" mode — the popover has closed
    /// but the user is interacting with the detail panel (selecting text, etc.).
    private var isStandalone = false
    private var localClickMonitor: Any?
    private var globalClickMonitor: Any?

    private init() {
        // Observe display changes — dismiss stale panels when screens rearrange
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screenConfigChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        removeClickMonitors()
        popoverWindowObserver = nil
    }

    func show(item: ClipboardItem, relativeTo popoverWindow: NSWindow?) {
        guard let popoverWindow = popoverWindow else { return }

        // If already showing this item, do nothing
        if currentItemID == item.id, let panel = panel, panel.isVisible { return }

        currentItemID = item.id
        isStandalone = false
        removeClickMonitors()

        // Track this popover window so we can dismiss when it closes
        trackPopoverWindow(popoverWindow)

        let fontSize = UserDefaults.standard.object(forKey: AppConstants.appFontSizeKey) as? Double ?? AppConstants.defaultFontSize
        let fontName = UserDefaults.standard.string(forKey: AppConstants.appFontNameKey) ?? AppConstants.defaultFontName

        let detailView = DetailPanelView(item: item, fontSize: fontSize, fontName: fontName) { [weak self] in
            self?.hide()
        }

        let hostingController = NSHostingController(rootView: detailView)

        // Wrap in NSVisualEffectView with .popover material for consistent appearance
        // across all panels (matches NSPopover chrome and Clippy panel)
        let effectView = NSVisualEffectView()
        effectView.material = .popover
        effectView.state = .active
        effectView.blendingMode = .behindWindow
        effectView.wantsLayer = true
        effectView.layer?.cornerRadius = AppConstants.panelCornerRadius
        effectView.layer?.masksToBounds = true

        let hostingView = hostingController.view
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        effectView.addSubview(hostingView)
        NSLayoutConstraint.activate([
            hostingView.leadingAnchor.constraint(equalTo: effectView.leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: effectView.trailingAnchor),
            hostingView.topAnchor.constraint(equalTo: effectView.topAnchor),
            hostingView.bottomAnchor.constraint(equalTo: effectView.bottomAnchor),
        ])

        let panelSize = NSSize(
            width: AppConstants.popoverWidth,
            height: AppConstants.popoverHeight
        )

        // Position to the left of the popover, clamped to the popover's screen
        let panelOrigin = clampedOrigin(
            popoverFrame: popoverWindow.frame,
            panelSize: panelSize,
            screen: popoverWindow.screen
        )

        if let existingPanel = panel {
            existingPanel.contentView = effectView
            existingPanel.setFrame(NSRect(origin: panelOrigin, size: panelSize), display: true, animate: true)
            existingPanel.orderFront(nil)
        } else {
            let newPanel = KeyablePanel(
                contentRect: NSRect(origin: panelOrigin, size: panelSize),
                styleMask: [.borderless, .nonactivatingPanel],
                backing: .buffered,
                defer: false
            )
            newPanel.level = .floating
            newPanel.backgroundColor = .clear
            newPanel.isOpaque = false
            newPanel.hasShadow = true
            // Keep visible even when app isn't active (menu bar app is rarely "active")
            newPanel.hidesOnDeactivate = false
            newPanel.isReleasedWhenClosed = false
            // Enable mouse tracking for SwiftUI hover to work reliably
            newPanel.acceptsMouseMovedEvents = true
            // Stay on the same Space as the popover, don't follow to other spaces
            newPanel.collectionBehavior = [.moveToActiveSpace]
            newPanel.contentView = effectView
            newPanel.orderFront(nil)
            panel = newPanel
        }
    }

    /// Called by popoverDidClose delegates. If the mouse is over the detail panel,
    /// keep it alive for text selection; otherwise dismiss immediately.
    func popoverDidDismiss() {
        stopTrackingPopoverWindow()

        guard let panel = panel, panel.isVisible else {
            hide()
            return
        }

        // Check if the mouse is currently over the detail panel
        let mouseLocation = NSEvent.mouseLocation
        if panel.frame.contains(mouseLocation) {
            // User is interacting with the panel — enter standalone mode
            isStandalone = true
            installClickMonitors()
        } else {
            // Mouse is elsewhere — dismiss
            hide()
        }
    }

    func hide() {
        panel?.orderOut(nil)
        currentItemID = nil
        isStandalone = false
        stopTrackingPopoverWindow()
        removeClickMonitors()
    }

    func isShowing(itemID: UUID) -> Bool {
        return currentItemID == itemID && (panel?.isVisible ?? false)
    }

    var isVisible: Bool {
        return panel?.isVisible ?? false
    }

    /// Expose the panel window so click-outside handlers can identify it
    var panelWindow: NSWindow? {
        return panel
    }

    // MARK: - Click-Outside Monitors (standalone mode)

    /// Install event monitors that dismiss the detail panel when the user clicks outside it.
    private func installClickMonitors() {
        removeClickMonitors()

        // Local monitor: catches clicks within our app (e.g. clicking the menu bar icon again)
        localClickMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] event in
            self?.handleClickEvent()
            return event
        }

        // Global monitor: catches clicks outside our app (e.g. clicking the desktop or another app)
        globalClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            self?.handleClickEvent()
        }
    }

    private func handleClickEvent() {
        guard let panel = panel, panel.isVisible, isStandalone else { return }

        let mouseLocation = NSEvent.mouseLocation
        if !panel.frame.contains(mouseLocation) {
            // Click was outside the detail panel — dismiss
            hide()
        }
        // If the click was inside the panel, keep it alive (user is selecting text, etc.)
    }

    private func removeClickMonitors() {
        if let monitor = localClickMonitor {
            NSEvent.removeMonitor(monitor)
            localClickMonitor = nil
        }
        if let monitor = globalClickMonitor {
            NSEvent.removeMonitor(monitor)
            globalClickMonitor = nil
        }
    }

    // MARK: - Popover Window Tracking

    /// Watch the popover window — when it becomes invisible or closes, check whether
    /// to keep the detail panel alive (if user is interacting) or dismiss it.
    private func trackPopoverWindow(_ window: NSWindow) {
        stopTrackingPopoverWindow()
        trackedPopoverWindow = window

        popoverWindowObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self else { return }
            if let closedWindow = notification.object as? NSWindow,
               closedWindow === self.trackedPopoverWindow {
                self.popoverDidDismiss()
            }
        }

        // Also poll visibility: transient popovers don't always post willClose.
        schedulePopoverVisibilityCheck()
    }

    private func stopTrackingPopoverWindow() {
        if let observer = popoverWindowObserver {
            NotificationCenter.default.removeObserver(observer)
            popoverWindowObserver = nil
        }
        trackedPopoverWindow = nil
    }

    /// Periodically check if the popover window is still visible.
    private func schedulePopoverVisibilityCheck() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self, self.isVisible else { return }

            // In standalone mode, we don't need to poll the popover anymore
            if self.isStandalone { return }

            if let trackedWindow = self.trackedPopoverWindow {
                if !trackedWindow.isVisible {
                    self.popoverDidDismiss()
                    return
                }
            } else {
                // Tracked window was deallocated
                self.popoverDidDismiss()
                return
            }

            // Continue checking
            self.schedulePopoverVisibilityCheck()
        }
    }

    // MARK: - Position Clamping

    /// Position to the left of the popover, but clamped so it stays on-screen.
    private func clampedOrigin(popoverFrame: NSRect, panelSize: NSSize, screen: NSScreen?) -> NSPoint {
        let idealX = popoverFrame.minX - panelSize.width - 2
        // Center vertically relative to the popover (accounts for popover arrow offset)
        let idealY = popoverFrame.midY - panelSize.height / 2

        guard let screen = screen else {
            return NSPoint(x: idealX, y: idealY)
        }

        let visibleFrame = screen.visibleFrame

        // Clamp X: if the panel would go off the left edge, put it to the right of the popover instead
        var x = idealX
        if x < visibleFrame.minX {
            x = popoverFrame.maxX + 2
        }
        // If still off-screen (right edge), just place at left edge of screen
        if x + panelSize.width > visibleFrame.maxX {
            x = visibleFrame.minX
        }

        // Clamp Y
        var y = idealY
        if y < visibleFrame.minY {
            y = visibleFrame.minY
        }
        if y + panelSize.height > visibleFrame.maxY {
            y = visibleFrame.maxY - panelSize.height
        }

        return NSPoint(x: x, y: y)
    }

    // MARK: - Notification Handlers

    @objc private func screenConfigChanged() {
        // Display arrangement changed — dismiss stale panels
        hide()
    }
}

// MARK: - Detail Panel SwiftUI View

struct DetailPanelView: View {
    let item: ClipboardItem
    let fontSize: Double
    let fontName: String
    var onClose: () -> Void

    @State private var copyConfirmation = false

    private var appFont: Font {
        if fontName == "System" || fontName.isEmpty {
            return .system(size: fontSize)
        } else {
            return .custom(fontName, size: fontSize)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header with Copy All + close button
            HStack(spacing: 8) {
                Text("Preview")
                    .font(.headline)

                Spacer()

                // Copy All button (pill style — matches Expand button)
                Button(action: copyAll) {
                    HStack(spacing: 2) {
                        Image(systemName: copyConfirmation ? "checkmark" : "chevron.left")
                            .font(.system(size: 8, weight: .semibold))
                        Text(copyConfirmation ? "Copied" : "Copy All")
                            .font(.system(size: 9, weight: .medium))
                    }
                    .foregroundColor(copyConfirmation ? .green : .accentColor)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(Color.accentColor.opacity(0.1))
                    .cornerRadius(4)
                }
                .buttonStyle(.plain)
                .help("Copy full content to clipboard")

                Button(action: onClose) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(.accentColor)
                }
                .buttonStyle(.plain)
                .help("Close preview")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            // Content area
            VStack(alignment: .leading, spacing: 0) {
                if item.type == .image, let thumbnail = item.thumbnail {
                    Image(nsImage: thumbnail)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: 160)
                        .cornerRadius(4)
                        .padding(.horizontal, 12)
                        .padding(.top, 8)
                }

                if let text = item.textContent, !text.isEmpty {
                    MultiSelectTextView(text: text, fontName: fontName, fontSize: fontSize)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(.horizontal, 4)
                }

                // Metadata
                VStack(alignment: .leading, spacing: 2) {
                    Text("Type: \(item.type.rawValue)")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text("Copied: \(item.timestamp.formatted(date: .abbreviated, time: .shortened))")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                    Text("\u{2318}+click to add selections \u{2022} double-click to clear")
                        .font(.system(size: 9))
                        .foregroundColor(.secondary.opacity(0.6))
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
        }
        .frame(width: AppConstants.popoverWidth, height: AppConstants.popoverHeight)
    }

    private func copyAll() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
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

        // Show confirmation, then close panel
        withAnimation(.easeInOut(duration: 0.2)) {
            copyConfirmation = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            onClose()
        }
    }
}

// MARK: - Multi-Select Text View (NSViewRepresentable)

/// Wraps a custom NSTextView that supports:
/// - Cmd+click to add multiple non-contiguous selections
/// - Double-click to clear all selections
/// - Normal click+drag for single selection
struct MultiSelectTextView: NSViewRepresentable {
    let text: String
    let fontName: String
    let fontSize: Double

    func makeNSView(context: Context) -> NSScrollView {
        let textView = ClippyTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.isRichText = false
        textView.textColor = .labelColor
        textView.font = nsFont()
        textView.string = text
        textView.textContainerInset = NSSize(width: 8, height: 4)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(
            width: 0,
            height: CGFloat.greatestFiniteMagnitude
        )

        let scrollView = NSScrollView()
        scrollView.documentView = textView
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.drawsBackground = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder

        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? ClippyTextView else { return }
        if textView.string != text {
            textView.string = text
        }
        textView.font = nsFont()
    }

    private func nsFont() -> NSFont {
        if fontName == "System" || fontName.isEmpty {
            return NSFont.systemFont(ofSize: fontSize)
        } else {
            return NSFont(name: fontName, size: fontSize) ?? NSFont.systemFont(ofSize: fontSize)
        }
    }
}

// MARK: - Custom NSTextView with multi-select support

/// NSTextView subclass that adds:
/// - Cmd+click+drag: accumulates multiple non-contiguous selections
/// - Double-click: clears all selections
class ClippyTextView: NSTextView {
    /// Ranges saved from before a Cmd+click, to merge with the new drag selection.
    private var savedRanges: [NSValue] = []
    private var isCommandSelecting = false

    override func mouseDown(with event: NSEvent) {
        // Make the panel key so keyboard shortcuts (Cmd+C) work
        window?.makeKey()

        // Double-click: clear all selections
        if event.clickCount == 2 {
            savedRanges = []
            isCommandSelecting = false
            setSelectedRange(NSRange(location: 0, length: 0))
            return
        }

        if event.modifierFlags.contains(.command) {
            // Save the current selections before the new drag starts
            savedRanges = selectedRanges
            isCommandSelecting = true
        } else {
            savedRanges = []
            isCommandSelecting = false
        }

        // super.mouseDown runs the full mouse-tracking loop (blocks until mouseUp)
        super.mouseDown(with: event)

        // After mouseUp, stop merging
        isCommandSelecting = false
    }

    /// Intercept selection updates — during Cmd+drag, merge the new selection
    /// with previously saved selections so the user sees all highlights live.
    override func setSelectedRanges(_ ranges: [NSValue], affinity: NSSelectionAffinity, stillSelecting: Bool) {
        if isCommandSelecting && !savedRanges.isEmpty {
            var merged = savedRanges
            for r in ranges {
                // Avoid adding empty/duplicate ranges
                let range = r.rangeValue
                if range.length > 0 || stillSelecting {
                    merged.append(r)
                }
            }
            // NSTextView requires at least one range
            if merged.isEmpty {
                merged = [NSValue(range: NSRange(location: 0, length: 0))]
            }
            super.setSelectedRanges(merged, affinity: affinity, stillSelecting: stillSelecting)
        } else {
            super.setSelectedRanges(ranges, affinity: affinity, stillSelecting: stillSelecting)
        }
    }

    // Allow the text view to become first responder for keyboard shortcuts (Cmd+C)
    override var acceptsFirstResponder: Bool { true }

    override func becomeFirstResponder() -> Bool { true }
}
