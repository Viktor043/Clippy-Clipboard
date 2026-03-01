import Cocoa
import CoreGraphics
import IOKit

// Private framework function types
private typealias DSSetBrightness = @convention(c) (UInt32, Float) -> Int32
private typealias DSGetBrightness = @convention(c) (UInt32, UnsafeMutablePointer<Float>) -> Int32

class ScreenBrightnessManager: ObservableObject {
    @Published var hardwareBrightness: Float = 1.0
    @Published var softwareDimming: Float = 1.0

    private(set) var hardwareControlAvailable = false

    private var setBrightnessImpl: ((Float) -> Bool)?
    private var getBrightnessImpl: (() -> Float?)?

    init() {
        loadBrightnessAPI()
        if let current = getBrightnessImpl?() {
            hardwareBrightness = current
        }
    }

    // MARK: - API Loading

    private func loadBrightnessAPI() {
        // Try DisplayServices private framework first (most reliable on Apple Silicon)
        if let handle = dlopen("/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices", RTLD_NOW) {
            let displayID = CGMainDisplayID()

            if let setSym = dlsym(handle, "DisplayServicesSetBrightness"),
               let getSym = dlsym(handle, "DisplayServicesGetBrightness") {
                let setFunc = unsafeBitCast(setSym, to: DSSetBrightness.self)
                let getFunc = unsafeBitCast(getSym, to: DSGetBrightness.self)

                setBrightnessImpl = { value in
                    return setFunc(displayID, value) == 0
                }
                getBrightnessImpl = {
                    var val: Float = 0
                    if getFunc(displayID, &val) == 0 {
                        return val
                    }
                    return nil
                }
                hardwareControlAvailable = true
                return
            }
        }

        // Fallback: try IOKit IODisplayConnect
        let mainPort: mach_port_t = 0
        var iterator: io_iterator_t = 0
        let result = IOServiceGetMatchingServices(mainPort, IOServiceMatching("IODisplayConnect"), &iterator)
        if result == kIOReturnSuccess {
            let service = IOIteratorNext(iterator)
            IOObjectRelease(iterator)
            if service != IO_OBJECT_NULL {
                IOObjectRelease(service)
                setBrightnessImpl = { value in
                    var iter: io_iterator_t = 0
                    let r = IOServiceGetMatchingServices(0, IOServiceMatching("IODisplayConnect"), &iter)
                    guard r == kIOReturnSuccess else { return false }
                    var success = false
                    var svc = IOIteratorNext(iter)
                    while svc != IO_OBJECT_NULL {
                        let res = IODisplaySetFloatParameter(svc, 0, kIODisplayBrightnessKey as CFString, value)
                        if res == kIOReturnSuccess { success = true }
                        IOObjectRelease(svc)
                        svc = IOIteratorNext(iter)
                    }
                    IOObjectRelease(iter)
                    return success
                }
                getBrightnessImpl = {
                    var iter: io_iterator_t = 0
                    let r = IOServiceGetMatchingServices(0, IOServiceMatching("IODisplayConnect"), &iter)
                    guard r == kIOReturnSuccess else { return nil }
                    var brightness: Float?
                    var svc = IOIteratorNext(iter)
                    while svc != IO_OBJECT_NULL {
                        var val: Float = 0
                        let res = IODisplayGetFloatParameter(svc, 0, kIODisplayBrightnessKey as CFString, &val)
                        if res == kIOReturnSuccess { brightness = val }
                        IOObjectRelease(svc)
                        svc = IOIteratorNext(iter)
                    }
                    IOObjectRelease(iter)
                    return brightness
                }
                hardwareControlAvailable = true
                return
            }
        }
    }

    // MARK: - Hardware Brightness

    func applyHardwareBrightness() {
        let clamped = max(0.0, min(1.0, hardwareBrightness))
        _ = setBrightnessImpl?(clamped)
    }

    func readCurrentHardwareBrightness() {
        if let val = getBrightnessImpl?() {
            hardwareBrightness = val
        }
    }

    // MARK: - Software Dimming (Gamma Overlay)

    func applySoftwareDimming() {
        let factor = max(0.01, min(1.0, softwareDimming))
        let displayID = CGMainDisplayID()
        let tableSize: UInt32 = 256

        var red = [CGGammaValue](repeating: 0, count: Int(tableSize))
        var green = [CGGammaValue](repeating: 0, count: Int(tableSize))
        var blue = [CGGammaValue](repeating: 0, count: Int(tableSize))

        for i in 0..<Int(tableSize) {
            let normalized = CGGammaValue(Float(i) / Float(tableSize - 1))
            // Apply gamma curve with dimming factor
            let dimmed = normalized * CGGammaValue(factor)
            red[i] = dimmed
            green[i] = dimmed
            blue[i] = dimmed
        }

        CGSetDisplayTransferByTable(displayID, tableSize, &red, &green, &blue)
    }

    // MARK: - Reset

    func resetGamma() {
        CGDisplayRestoreColorSyncSettings()
        softwareDimming = 1.0
    }

    func reset() {
        resetGamma()
        hardwareBrightness = 1.0
        applyHardwareBrightness()
    }
}
