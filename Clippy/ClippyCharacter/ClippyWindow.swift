import AppKit

final class ClippyWindow: NSPanel {
    private var initialMouseLocation: NSPoint?

    init() {
        // Apply stored scale to initial window size
        let storedScale = UserDefaults.standard.object(forKey: AppConstants.clippyScaleKey) as? Double ?? AppConstants.clippyDefaultScale
        let baseSize = AppConstants.clippyWindowSize
        let size = NSSize(
            width: baseSize.width * storedScale,
            height: baseSize.height * storedScale
        )

        // Restore saved position or default to bottom-right
        var origin: NSPoint
        if UserDefaults.standard.bool(forKey: AppConstants.hasStoredPositionKey) {
            let x = UserDefaults.standard.double(forKey: AppConstants.clippyPositionXKey)
            let y = UserDefaults.standard.double(forKey: AppConstants.clippyPositionYKey)
            origin = NSPoint(x: x, y: y)

            // Validate the stored position is on any connected screen
            if !ClippyWindow.isOriginOnScreen(origin, size: size) {
                // Stored position is off-screen (display was disconnected, rearranged, etc.)
                // Fall back to default position on the main screen
                let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
                origin = NSPoint(
                    x: screenFrame.maxX - size.width - AppConstants.clippyEdgeInset,
                    y: screenFrame.minY + AppConstants.clippyEdgeInset
                )
            }
        } else {
            let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            origin = NSPoint(
                x: screenFrame.maxX - size.width - AppConstants.clippyEdgeInset,
                y: screenFrame.minY + AppConstants.clippyEdgeInset
            )
        }

        super.init(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )

        level = .floating
        // Don't use isMovableByWindowBackground — we handle drag manually
        // so clicks can pass through to the view controller
        isMovableByWindowBackground = false
        backgroundColor = .clear
        hasShadow = false
        isOpaque = false
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        hidesOnDeactivate = false
        isReleasedWhenClosed = false
        acceptsMouseMovedEvents = true

        // Listen for display configuration changes to revalidate position
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screenConfigChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Screen Validation

    /// Check whether at least a portion of the window would be visible on any connected screen.
    private static func isOriginOnScreen(_ origin: NSPoint, size: NSSize) -> Bool {
        let windowRect = NSRect(origin: origin, size: size)
        for screen in NSScreen.screens {
            // If at least 20px of the window overlaps with the screen's visible area, it's fine
            if screen.visibleFrame.intersects(windowRect) {
                let intersection = screen.visibleFrame.intersection(windowRect)
                if intersection.width >= 20 && intersection.height >= 20 {
                    return true
                }
            }
        }
        return false
    }

    @objc private func screenConfigChanged() {
        // If Clippy is now off-screen after a display change, move to main screen
        if !ClippyWindow.isOriginOnScreen(frame.origin, size: frame.size) {
            let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let newOrigin = NSPoint(
                x: screenFrame.maxX - frame.width - AppConstants.clippyEdgeInset,
                y: screenFrame.minY + AppConstants.clippyEdgeInset
            )
            setFrameOrigin(newOrigin)
            savePosition()
        }
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    // Manual drag handling: if mouse moves enough, drag the window
    override func mouseDragged(with event: NSEvent) {
        // Check if position is locked
        let locked = UserDefaults.standard.bool(forKey: AppConstants.positionLockedKey)
        if locked { return }

        let currentLocation = NSEvent.mouseLocation
        if initialMouseLocation == nil {
            initialMouseLocation = currentLocation
        }

        guard let startLoc = initialMouseLocation else { return }
        let dx = currentLocation.x - startLoc.x
        let dy = currentLocation.y - startLoc.y

        // Only start dragging after 5px movement threshold
        if abs(dx) > 5 || abs(dy) > 5 {
            var origin = self.frame.origin
            origin.x += event.deltaX
            origin.y -= event.deltaY
            self.setFrameOrigin(origin)
        }
    }

    override func mouseUp(with event: NSEvent) {
        // Save position after drag
        if initialMouseLocation != nil {
            savePosition()
        }
        initialMouseLocation = nil
        super.mouseUp(with: event)
    }

    private func savePosition() {
        let origin = frame.origin
        UserDefaults.standard.set(origin.x, forKey: AppConstants.clippyPositionXKey)
        UserDefaults.standard.set(origin.y, forKey: AppConstants.clippyPositionYKey)
        UserDefaults.standard.set(true, forKey: AppConstants.hasStoredPositionKey)
    }
}
