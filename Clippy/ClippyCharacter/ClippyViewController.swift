import AppKit
import SpriteKit
import SwiftUI

final class ClippyViewController: NSViewController {
    private var skView: SKView!
    private var scene: ClippyScene!
    private var animationController: ClippyAnimationController?
    private var balloonView: ClippyBalloonView?
    private var clipboardPanel: NSPanel?
    private var panelLocalClickMonitor: Any?
    private var panelGlobalClickMonitor: Any?
    private var agent: Agent?
    private var clickOverlay: ClickOverlayView?
    private var overlayHeightConstraint: NSLayoutConstraint?

    // Reference to the shared store so we can build clipboard menus
    var store: ClipboardStore?

    override func loadView() {
        // Use a ClickThroughView container so clicks in transparent areas above
        // the character pass through to the desktop instead of being swallowed.
        let container = ClickThroughView()
        container.wantsLayer = true

        // RenderOnlySKView never claims mouse hits — all interaction goes
        // through the ClickOverlayView, which is sized to the character sprite.
        skView = RenderOnlySKView()
        skView.allowsTransparency = true
        skView.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(skView)

        NSLayoutConstraint.activate([
            skView.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            skView.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            skView.topAnchor.constraint(equalTo: container.topAnchor),
            skView.bottomAnchor.constraint(equalTo: container.bottomAnchor),
        ])

        self.view = container
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        scene = ClippyScene(size: AppConstants.clippyWindowSize)
        scene.backgroundColor = .clear
        skView.presentScene(scene)
        skView.allowsTransparency = true

        // Add transparent overlay as a SIBLING of the SKView (both children of container).
        // Since it's added after SKView, it's on top in the z-order and gets hit-tested first.
        let overlay = ClickOverlayView()
        overlay.wantsLayer = true
        overlay.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(overlay)
        self.clickOverlay = overlay

        // The overlay height will be updated in loadAgent / applyScale to match
        // the actual character sprite size. Start with a sensible default.
        let heightConstraint = overlay.heightAnchor.constraint(equalToConstant: AppConstants.clippyWindowSize.height)
        self.overlayHeightConstraint = heightConstraint
        NSLayoutConstraint.activate([
            overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            heightConstraint,
            overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        overlay.onClick = { [weak self] event in
            self?.handleClippyClick(event: event)
        }
        overlay.onRightClick = { [weak self] event in
            self?.handleRightClick(event: event)
        }
    }

    func loadAgent(fromURL url: URL) {
        guard let loadedAgent = Agent(agentURL: url) else {
            print("Failed to load agent from: \(url.path)")
            return
        }

        self.agent = loadedAgent
        let charSize = loadedAgent.character.size
        scene.updateSpriteSize(charSize)

        if let window = view.window {
            let newSize = NSSize(
                width: max(charSize.width, AppConstants.clippyWindowSize.width),
                height: max(charSize.height, AppConstants.clippyWindowSize.height) + 40
            )
            window.setContentSize(newSize)
            scene.size = newSize
            scene.agentSprite.position = CGPoint(x: newSize.width / 2, y: charSize.height / 2)

            // Size the click overlay to match the character sprite height
            overlayHeightConstraint?.constant = charSize.height
        }

        animationController = ClippyAnimationController(
            agent: loadedAgent,
            sprite: scene.agentSprite
        )
        animationController?.showInitialFrame()
        animationController?.startIdleTimer()
    }

    // MARK: - Scale

    func applyScale(_ scale: Double) {
        guard let window = view.window, let agent = self.agent else { return }

        let charSize = agent.character.size
        let baseWidth = max(charSize.width, AppConstants.clippyWindowSize.width)
        let baseHeight = max(charSize.height, AppConstants.clippyWindowSize.height) + 40

        let newWidth = baseWidth * scale
        let newHeight = baseHeight * scale
        let newSize = NSSize(width: newWidth, height: newHeight)

        // Keep centered on current position
        let oldFrame = window.frame
        let centerX = oldFrame.midX
        let centerY = oldFrame.midY
        let newOrigin = NSPoint(
            x: centerX - newWidth / 2,
            y: centerY - newHeight / 2
        )

        window.setFrame(NSRect(origin: newOrigin, size: newSize), display: true, animate: true)

        // Update SpriteKit scene
        scene.size = newSize
        let scaledCharHeight = charSize.height * scale
        scene.agentSprite.size = CGSize(
            width: charSize.width * scale,
            height: scaledCharHeight
        )
        scene.agentSprite.position = CGPoint(x: newSize.width / 2, y: scaledCharHeight / 2)

        // Resize click overlay to match the scaled character
        overlayHeightConstraint?.constant = scaledCharHeight
    }

    // MARK: - Click Handling

    private func handleClippyClick(event: NSEvent) {
        animationController?.handleClick()
        showClipboardPopover()
    }

    private func handleRightClick(event: NSEvent) {
        let menu = NSMenu()

        // Play Animation
        let animateItem = NSMenuItem(title: "Play Animation", action: #selector(playRandomAnimation), keyEquivalent: "")
        animateItem.target = self
        menu.addItem(animateItem)

        menu.addItem(NSMenuItem.separator())

        // Size submenu
        let sizeMenu = NSMenu()
        let labels = ["Extra Small", "Small", "Medium", "Large", "Extra Large"]
        let scales = AppConstants.clippyScaleOptions
        let currentScale = UserDefaults.standard.object(forKey: AppConstants.clippyScaleKey) as? Double ?? AppConstants.clippyDefaultScale

        for (i, scale) in scales.enumerated() {
            let item = NSMenuItem(title: labels[i], action: #selector(setClippyScale(_:)), keyEquivalent: "")
            item.target = self
            item.tag = Int(scale * 100)
            if abs(scale - currentScale) < 0.01 {
                item.state = .on
            }
            sizeMenu.addItem(item)
        }

        let sizeItem = NSMenuItem(title: "Size", action: nil, keyEquivalent: "")
        sizeItem.submenu = sizeMenu
        menu.addItem(sizeItem)

        // Lock/Unlock position
        let isLocked = UserDefaults.standard.bool(forKey: AppConstants.positionLockedKey)
        let lockTitle = isLocked ? "Unlock Position" : "Lock Position"
        let lockItem = NSMenuItem(title: lockTitle, action: #selector(togglePositionLock), keyEquivalent: "")
        lockItem.target = self
        menu.addItem(lockItem)

        // Mute/Unmute toggle
        let isMuted = !(UserDefaults.standard.object(forKey: "playSounds") as? Bool ?? true)
        let muteTitle = isMuted ? "Unmute Sounds" : "Mute Sounds"
        let muteItem = NSMenuItem(title: muteTitle, action: #selector(toggleMute), keyEquivalent: "")
        muteItem.target = self
        menu.addItem(muteItem)

        menu.addItem(NSMenuItem.separator())

        // Hide Clippy
        let hideItem = NSMenuItem(title: "Hide Clippy", action: #selector(hideClippy), keyEquivalent: "")
        hideItem.target = self
        menu.addItem(hideItem)

        let location = NSPoint(x: view.bounds.midX, y: view.bounds.maxY)
        menu.popUp(positioning: nil, at: location, in: view)
    }

    // MARK: - Clipboard Panel (standalone panel instead of NSPopover for proper hover/tracking)

    private func showClipboardPopover() {
        // Toggle: if already showing, close it
        if let panel = clipboardPanel, panel.isVisible {
            hideClipboardPanel()
            return
        }

        guard let store = store, let clippyWindow = view.window else { return }

        let historyView = ClipboardHistoryView(store: store) { [weak self] in
            self?.hideClipboardPanel()
        }
        let hostingController = NSHostingController(rootView: historyView)

        let panelSize = NSSize(
            width: AppConstants.popoverWidth,
            height: AppConstants.popoverHeight
        )

        // Position above the Clippy character, centered horizontally
        let clippyFrame = clippyWindow.frame
        let panelOrigin = NSPoint(
            x: clippyFrame.midX - panelSize.width / 2,
            y: clippyFrame.maxY + 4
        )

        // Clamp to screen bounds
        let origin = clampPanelOrigin(panelOrigin, panelSize: panelSize, screen: clippyWindow.screen)

        // Wrap the hosting view in an NSVisualEffectView with .popover material
        // This gives the exact same background as NSPopover uses internally
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

        if let existingPanel = clipboardPanel {
            existingPanel.contentView = effectView
            existingPanel.setFrame(NSRect(origin: origin, size: panelSize), display: true, animate: true)
            existingPanel.orderFront(nil)
        } else {
            let panel = KeyablePanel(
                contentRect: NSRect(origin: origin, size: panelSize),
                styleMask: [.borderless, .nonactivatingPanel],
                backing: .buffered,
                defer: false
            )
            panel.level = .floating
            panel.backgroundColor = .clear
            panel.isOpaque = false
            panel.hasShadow = true
            panel.hidesOnDeactivate = false
            panel.isReleasedWhenClosed = false
            // Enable mouse tracking for SwiftUI hover to work reliably
            panel.acceptsMouseMovedEvents = true
            panel.collectionBehavior = [.moveToActiveSpace]
            panel.contentView = effectView
            panel.orderFront(nil)
            clipboardPanel = panel
        }

        // Make key so keyboard input works in search field
        clipboardPanel?.makeKey()

        // Install click-outside monitors to dismiss
        installPanelClickMonitors()
    }

    private func hideClipboardPanel() {
        removePanelClickMonitors()
        clipboardPanel?.orderOut(nil)
        // Let the detail panel decide whether to stay or dismiss
        DetailPanelController.shared.popoverDidDismiss()
    }

    private func installPanelClickMonitors() {
        removePanelClickMonitors()

        // Defer installation by one run loop cycle so the current click event
        // (the one that opened the panel) is fully processed before monitors start.
        DispatchQueue.main.async { [weak self] in
            guard let self = self, self.clipboardPanel?.isVisible == true else { return }

            self.panelLocalClickMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] event in
                self?.handlePanelClickEvent()
                return event
            }

            self.panelGlobalClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
                self?.handlePanelClickEvent()
            }
        }
    }

    private func handlePanelClickEvent() {
        guard let panel = clipboardPanel, panel.isVisible else { return }

        let mouseLocation = NSEvent.mouseLocation

        // Click inside the clipboard panel — keep it alive
        if panel.frame.contains(mouseLocation) { return }

        // Click inside the detail panel — keep both alive
        if DetailPanelController.shared.isVisible,
           let detailWindow = DetailPanelController.shared.panelWindow,
           detailWindow.frame.contains(mouseLocation) {
            return
        }

        // Click on the Clippy character — let the normal ClickOverlayView toggle handle it.
        // Don't dismiss here; showClipboardPopover() will toggle the panel off.
        if let clippyWindow = view.window, clippyWindow.frame.contains(mouseLocation) {
            return
        }

        // Click outside everything — dismiss
        DetailPanelController.shared.hide()
        hideClipboardPanel()
    }

    private func removePanelClickMonitors() {
        if let monitor = panelLocalClickMonitor {
            NSEvent.removeMonitor(monitor)
            panelLocalClickMonitor = nil
        }
        if let monitor = panelGlobalClickMonitor {
            NSEvent.removeMonitor(monitor)
            panelGlobalClickMonitor = nil
        }
    }

    /// Clamp the panel origin so it stays on-screen.
    private func clampPanelOrigin(_ origin: NSPoint, panelSize: NSSize, screen: NSScreen?) -> NSPoint {
        guard let screen = screen else { return origin }
        let visibleFrame = screen.visibleFrame

        var x = origin.x
        var y = origin.y

        if x < visibleFrame.minX { x = visibleFrame.minX }
        if x + panelSize.width > visibleFrame.maxX { x = visibleFrame.maxX - panelSize.width }
        if y < visibleFrame.minY { y = visibleFrame.minY }
        if y + panelSize.height > visibleFrame.maxY { y = visibleFrame.maxY - panelSize.height }

        return NSPoint(x: x, y: y)
    }

    // MARK: - Actions

    @objc private func playRandomAnimation() {
        animationController?.playRandom { [weak self] in
            self?.animationController?.showInitialFrame()
            self?.animationController?.startIdleTimer()
        }
    }

    @objc private func hideClippy() {
        view.window?.orderOut(nil)
        UserDefaults.standard.set(false, forKey: "showClippyCharacter")
    }

    @objc private func setClippyScale(_ sender: NSMenuItem) {
        let scale = Double(sender.tag) / 100.0
        UserDefaults.standard.set(scale, forKey: AppConstants.clippyScaleKey)
    }

    @objc private func togglePositionLock() {
        let current = UserDefaults.standard.bool(forKey: AppConstants.positionLockedKey)
        UserDefaults.standard.set(!current, forKey: AppConstants.positionLockedKey)

        // If locking, save current position
        if !current, let window = view.window {
            let origin = window.frame.origin
            UserDefaults.standard.set(origin.x, forKey: AppConstants.clippyPositionXKey)
            UserDefaults.standard.set(origin.y, forKey: AppConstants.clippyPositionYKey)
            UserDefaults.standard.set(true, forKey: AppConstants.hasStoredPositionKey)
        }
    }

    @objc private func toggleMute() {
        let current = UserDefaults.standard.object(forKey: "playSounds") as? Bool ?? true
        UserDefaults.standard.set(!current, forKey: "playSounds")
    }

    // MARK: - Reactions

    func reactToCopy(itemType: ClipboardItemType) {
        let animationName: String
        switch itemType {
        case .image:
            animationName = "Searching"
        case .file:
            animationName = "Writing"
        default:
            animationName = "Greeting"
        }

        animationController?.play(animationNamed: animationName) { [weak self] in
            self?.animationController?.showInitialFrame()
            self?.animationController?.startIdleTimer()
        }

        let message: String
        switch itemType {
        case .text: message = "Copied text!"
        case .richText: message = "Got that rich text!"
        case .image: message = "Nice screenshot!"
        case .file: message = "File copied!"
        }
        showBalloon(message: message)
    }

    // MARK: - Balloon

    private func showBalloon(message: String) {
        balloonView?.removeFromSuperview()

        let balloon = ClippyBalloonView(message: message)
        balloon.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(balloon)

        NSLayoutConstraint.activate([
            balloon.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            balloon.bottomAnchor.constraint(equalTo: view.topAnchor, constant: -4),
            balloon.widthAnchor.constraint(lessThanOrEqualToConstant: 180)
        ])

        balloonView = balloon

        DispatchQueue.main.asyncAfter(deadline: .now() + AppConstants.balloonDismissDelay) { [weak self] in
            NSAnimationContext.runAnimationGroup({ context in
                context.duration = 0.3
                self?.balloonView?.animator().alphaValue = 0
            }, completionHandler: {
                self?.balloonView?.removeFromSuperview()
                self?.balloonView = nil
            })
        }
    }
}

// MARK: - Transparent Click Overlay

/// A transparent NSView that sits as a SIBLING on top of the SKView (both inside a container).
/// SKView routes hit testing to SKScene internally, bypassing any child subviews.
/// By making this a sibling added after SKView, standard NSView hit testing finds it first.
private class ClickOverlayView: NSView {
    var onClick: ((NSEvent) -> Void)?
    var onRightClick: ((NSEvent) -> Void)?

    private var mouseDownLocation: NSPoint?
    private var mouseDownTime: Date?

    override var acceptsFirstResponder: Bool { true }

    // Guarantee this view catches all hits within its frame
    override func hitTest(_ point: NSPoint) -> NSView? {
        if frame.contains(point) {
            return self
        }
        return nil
    }

    override func mouseDown(with event: NSEvent) {
        mouseDownLocation = event.locationInWindow
        mouseDownTime = Date()
    }

    override func mouseDragged(with event: NSEvent) {
        // Don't forward drag if position is locked
        let locked = UserDefaults.standard.bool(forKey: AppConstants.positionLockedKey)
        if locked { return }
        window?.mouseDragged(with: event)
    }

    override func mouseUp(with event: NSEvent) {
        let upLocation = event.locationInWindow
        let elapsed = Date().timeIntervalSince(mouseDownTime ?? Date())

        if let downLoc = mouseDownLocation {
            let dx = abs(upLocation.x - downLoc.x)
            let dy = abs(upLocation.y - downLoc.y)
            if dx < 10 && dy < 10 && elapsed < 0.5 {
                onClick?(event)
            }
        }

        mouseDownLocation = nil
        mouseDownTime = nil

        // Forward mouseUp to window to reset its drag state
        window?.mouseUp(with: event)
    }

    override func rightMouseDown(with event: NSEvent) {
        // Consume to prevent default behavior
    }

    override func rightMouseUp(with event: NSEvent) {
        onRightClick?(event)
    }
}

// MARK: - Click-Through Container

/// Container view that returns nil (instead of self) when no subview claims a hit.
/// This allows clicks in transparent areas above the Clippy character to pass through
/// the borderless window to the desktop or apps underneath.
private class ClickThroughView: NSView {
    override func hitTest(_ point: NSPoint) -> NSView? {
        guard frame.contains(point) else { return nil }
        let localPoint = convert(point, from: superview)
        for subview in subviews.reversed() {
            if let hit = subview.hitTest(localPoint) {
                return hit
            }
        }
        // No subview claimed the hit — pass through instead of returning self
        return nil
    }
}

// MARK: - Rendering-Only SKView

/// SKView subclass that never intercepts mouse events.
/// All user interaction is handled by the ClickOverlayView; this view only renders the sprite.
private class RenderOnlySKView: SKView {
    override func hitTest(_ point: NSPoint) -> NSView? {
        return nil
    }
}
