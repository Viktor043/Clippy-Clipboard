import Cocoa
import ObjectiveC

// objc_msgSend typed signatures for CoreBrightness KeyboardBrightnessClient
private typealias GetBrightnessType = @convention(c) (AnyObject, Selector, UInt64) -> Float
private typealias SetBrightnessType = @convention(c) (AnyObject, Selector, Float, UInt64) -> Bool

class KeyboardBrightnessManager: ObservableObject {
    @Published var brightness: Float = 0.0
    /// Software reduction factor (0.01–1.0) multiplied against brightness for ultra-low levels
    @Published var reduction: Float = 1.0
    private(set) var isAvailable = false

    private var client: NSObject?
    private var keyboardID: UInt64 = 0
    private var getBrightnessFn: GetBrightnessType?
    private var setBrightnessFn: SetBrightnessType?

    private let getSel = Selector(("brightnessForKeyboard:"))
    private let setSel = Selector(("setBrightness:forKeyboard:"))

    init() {
        setupCoreBrightness()
        if isAvailable, let current = readBrightness() {
            brightness = current
        }
    }

    // MARK: - CoreBrightness Setup

    private func setupCoreBrightness() {
        // Load CoreBrightness private framework
        guard dlopen("/System/Library/PrivateFrameworks/CoreBrightness.framework/CoreBrightness", RTLD_NOW) != nil else {
            return
        }

        // Get the KeyboardBrightnessClient class
        guard let cls = NSClassFromString("KeyboardBrightnessClient") as? NSObject.Type else {
            return
        }

        let instance = cls.init()

        // Get available keyboard backlight IDs
        guard let result = instance.perform(Selector(("copyKeyboardBacklightIDs")))?.takeRetainedValue(),
              let ids = result as? [NSNumber],
              let firstID = ids.first else {
            return
        }

        client = instance
        keyboardID = firstID.uint64Value

        // Resolve objc_msgSend for typed calls (float params/returns)
        guard let msgSend = dlsym(UnsafeMutableRawPointer(bitPattern: -2)!, "objc_msgSend") else {
            return
        }

        getBrightnessFn = unsafeBitCast(msgSend, to: GetBrightnessType.self)
        setBrightnessFn = unsafeBitCast(msgSend, to: SetBrightnessType.self)

        isAvailable = true
    }

    // MARK: - Read / Write

    private func readBrightness() -> Float? {
        guard let client = client, let fn = getBrightnessFn else { return nil }
        let val = fn(client, getSel, keyboardID)
        return val >= 0 ? val : nil
    }

    /// Effective brightness sent to hardware = brightness * reduction
    var effectiveBrightness: Float {
        max(0.0, min(1.0, brightness * reduction))
    }

    func applyBrightness() {
        guard let client = client, let fn = setBrightnessFn else { return }
        _ = fn(client, setSel, effectiveBrightness, keyboardID)
    }

    func readCurrentBrightness() {
        if let val = readBrightness() {
            brightness = val
        }
    }

    func reset() {
        brightness = 0.5
        reduction = 1.0
        applyBrightness()
    }
}
