import AppKit

/// NSPanel subclass that reliably accepts keyboard input and first responder.
/// Standard NSPanel with [.borderless, .nonactivatingPanel] can fail to route
/// keyboard events (e.g. typing in a search field, Cmd+C) because borderless
/// panels don't always become key properly. This subclass fixes that.
class KeyablePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}
