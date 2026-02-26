import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var menuBarManager: MenuBarManager?
    private var clipboardMonitor: ClipboardMonitor?
    private var clippyWindow: ClippyWindow?
    private var clippyViewController: ClippyViewController?
    private let store = ClipboardStore.shared
    private var clippyObserver: NSKeyValueObservation?
    private var menuBarObserver: NSKeyValueObservation?
    private var scaleObserver: NSKeyValueObservation?

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        setupClipboardMonitor()
        setupClippyCharacter()
        observeClippyToggle()
        observeMenuBarToggle()
        observeScaleChange()
        updateActivationPolicy()
    }

    func applicationWillTerminate(_ notification: Notification) {
        clipboardMonitor?.stopMonitoring()
        clippyObserver?.invalidate()
        menuBarObserver?.invalidate()
        scaleObserver?.invalidate()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        }
        return true
    }

    // MARK: - Setup

    private func setupMenuBar() {
        menuBarManager = MenuBarManager(store: store)
        let showMenuBar = UserDefaults.standard.object(forKey: AppConstants.showMenuBarIconKey) as? Bool ?? true
        if showMenuBar {
            menuBarManager?.setupStatusBar()
        }
    }

    private func setupClipboardMonitor() {
        clipboardMonitor = ClipboardMonitor()
        clipboardMonitor?.onNewClip = { [weak self] item in
            guard let self = self else { return }
            self.store.addItem(item)
            self.clippyViewController?.reactToCopy(itemType: item.type)
        }
        clipboardMonitor?.startMonitoring()
    }

    private func setupClippyCharacter() {
        let showClippy = UserDefaults.standard.object(forKey: "showClippyCharacter") as? Bool ?? true
        guard showClippy else { return }

        createClippyWindow()
    }

    // MARK: - Clippy Window Lifecycle

    private func createClippyWindow() {
        guard clippyWindow == nil else {
            clippyWindow?.orderFront(nil)
            return
        }

        clippyWindow = ClippyWindow()

        let viewController = ClippyViewController()
        viewController.store = store
        clippyWindow?.contentViewController = viewController
        clippyWindow?.orderFront(nil)
        clippyViewController = viewController

        // Load the agent from the app bundle
        if let agentURL = Bundle.main.url(forResource: "clippit", withExtension: "agent") {
            viewController.loadAgent(fromURL: agentURL)
        } else if let resourcesURL = Bundle.main.resourceURL?
            .appendingPathComponent("Resources")
            .appendingPathComponent("clippit.agent") {
            if FileManager.default.fileExists(atPath: resourcesURL.path) {
                viewController.loadAgent(fromURL: resourcesURL)
            } else {
                print("Clippy agent not found at: \(resourcesURL.path)")
            }
        }

        // Apply stored scale
        let storedScale = UserDefaults.standard.object(forKey: AppConstants.clippyScaleKey) as? Double ?? AppConstants.clippyDefaultScale
        if storedScale != 1.0 {
            viewController.applyScale(storedScale)
        }
    }

    private func destroyClippyWindow() {
        clippyWindow?.orderOut(nil)
        clippyWindow?.contentViewController = nil
        clippyWindow = nil
        clippyViewController = nil
    }

    // MARK: - Observers

    private func observeClippyToggle() {
        clippyObserver = UserDefaults.standard.observe(\.showClippyCharacter, options: [.new]) { [weak self] _, change in
            DispatchQueue.main.async {
                let show = change.newValue ?? true
                if show {
                    self?.createClippyWindow()
                } else {
                    self?.destroyClippyWindow()
                }
                self?.updateActivationPolicy()
            }
        }
    }

    private func observeMenuBarToggle() {
        menuBarObserver = UserDefaults.standard.observe(\.showMenuBarIcon, options: [.new]) { [weak self] _, change in
            DispatchQueue.main.async {
                let show = change.newValue ?? true
                if show {
                    self?.menuBarManager?.showStatusBar()
                } else {
                    self?.menuBarManager?.hideStatusBar()
                }
                self?.updateActivationPolicy()
            }
        }
    }

    private func observeScaleChange() {
        scaleObserver = UserDefaults.standard.observe(\.clippyScale, options: [.new]) { [weak self] _, change in
            DispatchQueue.main.async {
                let scale = change.newValue ?? AppConstants.clippyDefaultScale
                self?.clippyViewController?.applyScale(scale)
            }
        }
    }

    // MARK: - Activation Policy

    private func updateActivationPolicy() {
        let showMenuBar = UserDefaults.standard.object(forKey: AppConstants.showMenuBarIconKey) as? Bool ?? true
        if showMenuBar {
            NSApp.setActivationPolicy(.accessory)
        } else {
            // Menu bar hidden — show dock icon so user can still access the app
            NSApp.setActivationPolicy(.regular)
        }
    }
}

// MARK: - KVO-compatible UserDefaults keys

extension UserDefaults {
    @objc dynamic var showClippyCharacter: Bool {
        return object(forKey: "showClippyCharacter") as? Bool ?? true
    }
    @objc dynamic var showMenuBarIcon: Bool {
        return object(forKey: AppConstants.showMenuBarIconKey) as? Bool ?? true
    }
    @objc dynamic var clippyScale: Double {
        return object(forKey: AppConstants.clippyScaleKey) as? Double ?? AppConstants.clippyDefaultScale
    }
}
