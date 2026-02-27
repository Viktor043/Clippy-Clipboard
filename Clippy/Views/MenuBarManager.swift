import AppKit
import SwiftUI

final class MenuBarManager: NSObject, ObservableObject, NSPopoverDelegate {
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private let store: ClipboardStore
    private var localClickMonitor: Any?
    private var globalClickMonitor: Any?
    private var widthObserver: NSKeyValueObservation?
    private var heightObserver: NSKeyValueObservation?
    private var opacityObserver: NSKeyValueObservation?

    init(store: ClipboardStore) {
        self.store = store
        super.init()
        observePanelSettings()
    }

    func setupStatusBar() {
        guard statusItem == nil else { return }

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            button.image = NSImage(systemSymbolName: "paperclip", accessibilityDescription: "Clippy")
            button.action = #selector(togglePopover)
            button.target = self
        }

        if popover == nil {
            popover = NSPopover()
            popover?.contentSize = currentPanelSize()
            // Use applicationDefined so we control dismissal via click monitors.
            // This prevents the popover from auto-closing when the user clicks
            // inside the detail panel (e.g. the close button).
            popover?.behavior = .applicationDefined
            popover?.animates = true
            popover?.delegate = self
        }
    }

    func showStatusBar() {
        setupStatusBar()
    }

    func hideStatusBar() {
        guard let item = statusItem else { return }
        NSStatusBar.system.removeStatusItem(item)
        statusItem = nil
    }

    @objc private func togglePopover() {
        guard let popover = popover, let button = statusItem?.button else { return }

        if popover.isShown {
            closePopover()
        } else {
            // Update size to latest settings before showing
            popover.contentSize = currentPanelSize()

            let historyView = ClipboardHistoryView(store: store) { [weak self] in
                self?.closePopover()
            }
            popover.contentViewController = NSHostingController(rootView: historyView)
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)

            // Ensure popover becomes key window for keyboard input
            popover.contentViewController?.view.window?.makeKey()

            // Apply panel transparency
            applyPopoverOpacity()

            // Install click-outside monitors to dismiss
            installClickMonitors()
        }
    }

    private func closePopover() {
        removeClickMonitors()
        popover?.performClose(nil)
    }

    // MARK: - Click-Outside Monitors

    private func installClickMonitors() {
        removeClickMonitors()

        // Defer by one run loop cycle so the opening click is fully processed first
        DispatchQueue.main.async { [weak self] in
            guard let self = self, self.popover?.isShown == true else { return }

            self.localClickMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] event in
                self?.handleClickEvent()
                return event
            }

            self.globalClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
                self?.handleClickEvent()
            }
        }
    }

    private func handleClickEvent() {
        guard let popover = popover, popover.isShown else { return }

        let mouseLocation = NSEvent.mouseLocation

        // Click inside the popover — keep it alive
        if let popoverWindow = popover.contentViewController?.view.window,
           popoverWindow.frame.contains(mouseLocation) {
            return
        }

        // Click inside the detail panel — keep popover alive
        if DetailPanelController.shared.isVisible,
           let detailWindow = DetailPanelController.shared.panelWindow,
           detailWindow.frame.contains(mouseLocation) {
            return
        }

        // Click on the status bar button — let togglePopover handle it
        if let button = statusItem?.button, let buttonWindow = button.window,
           buttonWindow.frame.contains(mouseLocation) {
            return
        }

        // Click outside everything — dismiss
        DetailPanelController.shared.hide()
        closePopover()
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

    // MARK: - Dynamic Panel Size & Appearance

    private func currentPanelSize() -> NSSize {
        let w = UserDefaults.standard.object(forKey: AppConstants.popoverWidthKey) as? Double ?? Double(AppConstants.defaultPopoverWidth)
        let h = UserDefaults.standard.object(forKey: AppConstants.popoverHeightKey) as? Double ?? Double(AppConstants.defaultPopoverHeight)
        return NSSize(width: w, height: h)
    }

    private func applyPopoverOpacity() {
        guard let popover = popover, popover.isShown else { return }
        let opacity = UserDefaults.standard.object(forKey: AppConstants.panelOpacityKey) as? Double ?? AppConstants.defaultPanelOpacity
        popover.contentViewController?.view.window?.alphaValue = CGFloat(opacity)
    }

    private func observePanelSettings() {
        widthObserver = UserDefaults.standard.observe(\.popoverWidth, options: [.new]) { [weak self] _, _ in
            DispatchQueue.main.async {
                guard let self = self, let popover = self.popover, popover.isShown else { return }
                popover.contentSize = self.currentPanelSize()
            }
        }
        heightObserver = UserDefaults.standard.observe(\.popoverHeight, options: [.new]) { [weak self] _, _ in
            DispatchQueue.main.async {
                guard let self = self, let popover = self.popover, popover.isShown else { return }
                popover.contentSize = self.currentPanelSize()
            }
        }
        opacityObserver = UserDefaults.standard.observe(\.panelOpacity, options: [.new]) { [weak self] _, _ in
            DispatchQueue.main.async { self?.applyPopoverOpacity() }
        }
    }

    // MARK: - NSPopoverDelegate

    func popoverDidClose(_ notification: Notification) {
        removeClickMonitors()
        // When the popover closes, let the detail panel decide whether to stay
        // (if user is interacting with it) or dismiss
        DetailPanelController.shared.popoverDidDismiss()
    }
}
