import Cocoa
import SwiftUI

class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private let screenManager = ScreenBrightnessManager()
    private let keyboardManager = KeyboardBrightnessManager()
    private var eventMonitor: Any?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        let contentView = ContentView(
            screenManager: screenManager,
            keyboardManager: keyboardManager
        )

        popover = NSPopover()
        popover.contentSize = NSSize(width: 320, height: 0) // Height determined by content
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(rootView: contentView)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "sun.min.fill", accessibilityDescription: "Brightness Control")
            button.action = #selector(togglePopover)
            button.target = self
        }

        // Close popover when clicking outside
        eventMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            if let popover = self?.popover, popover.isShown {
                popover.performClose(nil)
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Always restore gamma on quit so the user isn't stuck with a dark screen
        screenManager.resetGamma()

        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
        }
    }

    @objc private func togglePopover() {
        guard let button = statusItem.button else { return }
        if popover.isShown {
            popover.performClose(nil)
        } else {
            // Refresh current values before showing
            screenManager.readCurrentHardwareBrightness()
            keyboardManager.readCurrentBrightness()

            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
            NSApp.activate(ignoringOtherApps: true)
        }
    }
}
