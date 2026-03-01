import Foundation
import IOKit
import IOKit.ps

/// Holds all battery information read from IOKit.
struct BatteryInfo {
    let percentage: Int
    let isCharging: Bool
    let isPluggedIn: Bool
    let isFullyCharged: Bool
    let timeRemaining: Int?       // minutes
    let timeToFullCharge: Int?    // minutes
    let cycleCount: Int?
    let health: Int?              // percentage of design capacity
    let temperature: Double?      // Celsius
    let currentCapacity: Int?     // mAh
    let maxCapacity: Int?         // mAh
    let designCapacity: Int?      // mAh
    let voltage: Double?          // Volts
    let amperage: Double?         // Amps
    let wattage: Double?          // Watts
}

/// Reads battery data using IOPowerSources (high-level) and the
/// AppleSmartBattery IOService (low-level) for detailed metrics.
final class BatteryService: ObservableObject {

    @Published var batteryInfo: BatteryInfo

    private var runLoopSource: CFRunLoopSource?

    init() {
        self.batteryInfo = BatteryService.readBattery()
        startMonitoring()
    }

    deinit {
        stopMonitoring()
    }

    // MARK: - Public

    func refresh() {
        batteryInfo = BatteryService.readBattery()
    }

    // MARK: - Event-driven monitoring

    private func startMonitoring() {
        let context = UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
        guard let source = IOPSNotificationCreateRunLoopSource({ context in
            guard let context = context else { return }
            let service = Unmanaged<BatteryService>.fromOpaque(context).takeUnretainedValue()
            DispatchQueue.main.async {
                service.refresh()
            }
        }, context)?.takeRetainedValue() else { return }

        self.runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .defaultMode)
    }

    private func stopMonitoring() {
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .defaultMode)
            runLoopSource = nil
        }
    }

    // MARK: - Read battery (static so it can be used from tests)

    static func readBattery() -> BatteryInfo {
        // --- High-level: IOPowerSources ---
        let percentage = readPercentageFromPowerSources()
        let isPluggedIn = readIsOnACPower()

        // --- Low-level: AppleSmartBattery IOService ---
        let service = IOServiceGetMatchingService(
            kIOMainPortDefault,
            IOServiceNameMatching("AppleSmartBattery")
        )
        defer { if service != IO_OBJECT_NULL { IOObjectRelease(service) } }

        let isCharging     = registryBool(service, "IsCharging") ?? false
        let isFullyCharged = registryBool(service, "FullyCharged") ?? false
        let cycleCount     = registryInt(service, "CycleCount")
        let designCap      = registryInt(service, "DesignCapacity")
        let tempRaw        = registryInt(service, "Temperature")
        let voltageRaw     = registryInt(service, "Voltage")
        let amperageRaw    = registryInt(service, "Amperage")
        let timeRemRaw     = registryInt(service, "TimeRemaining")
        let timeToFull     = registryInt(service, "AvgTimeToFull")

        // On modern macOS, "MaxCapacity" may return 100 (percentage-based).
        // Use "AppleRawMaxCapacity" for actual mAh, with NominalChargeCapacity
        // as a fallback, then "MaxCapacity" only if it looks like real mAh.
        let rawMaxCap      = registryInt(service, "AppleRawMaxCapacity")
        let nominalCap     = registryInt(service, "NominalChargeCapacity")
        let ioMaxCap       = registryInt(service, "MaxCapacity")
        let maxCap: Int? = rawMaxCap
            ?? nominalCap
            ?? (ioMaxCap.map { $0 > 1000 ? $0 : nil } ?? nil)

        // Similarly for current capacity — use raw if available.
        let rawCurCap      = registryInt(service, "AppleRawCurrentCapacity")
        let ioCurCap       = registryInt(service, "CurrentCapacity")
        let currentCap: Int? = rawCurCap
            ?? (ioCurCap.map { $0 > 1000 ? $0 : nil } ?? nil)

        let temperature: Double? = tempRaw.map { Double($0) / 100.0 }
        let voltage: Double?     = voltageRaw.map { Double($0) / 1000.0 }
        let amperage: Double?    = amperageRaw.map { Double($0) / 1000.0 }
        let wattage: Double?     = (voltage != nil && amperage != nil)
            ? abs(voltage! * amperage!)
            : nil
        let health: Int? = (maxCap != nil && designCap != nil && designCap! > 0)
            ? Int(round(Double(maxCap!) / Double(designCap!) * 100.0))
            : nil

        let timeRemaining: Int? = {
            if isCharging { return nil }
            guard let t = timeRemRaw, t >= 0 && t < 6000 else { return nil }
            return t
        }()
        let timeToFullCharge: Int? = {
            if !isCharging { return nil }
            guard let t = timeToFull, t >= 0 && t < 6000 else { return nil }
            return t
        }()

        return BatteryInfo(
            percentage: percentage,
            isCharging: isCharging,
            isPluggedIn: isPluggedIn,
            isFullyCharged: isFullyCharged,
            timeRemaining: timeRemaining,
            timeToFullCharge: timeToFullCharge,
            cycleCount: cycleCount,
            health: health,
            temperature: temperature,
            currentCapacity: currentCap,
            maxCapacity: maxCap,
            designCapacity: designCap,
            voltage: voltage,
            amperage: amperage,
            wattage: wattage
        )
    }

    // MARK: - IOPowerSources helpers

    private static func readPercentageFromPowerSources() -> Int {
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as Array
        for source in sources {
            guard let desc = IOPSGetPowerSourceDescription(snapshot, source)?
                .takeUnretainedValue() as? [String: Any] else { continue }
            if desc[kIOPSTypeKey] as? String == kIOPSInternalBatteryType {
                return desc[kIOPSCurrentCapacityKey] as? Int ?? 0
            }
        }
        return 0
    }

    private static func readIsOnACPower() -> Bool {
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let type = IOPSGetProvidingPowerSourceType(snapshot).takeRetainedValue() as String
        return type == kIOPSACPowerValue
    }

    // MARK: - IORegistry helpers

    private static func registryInt(_ service: io_object_t, _ key: String) -> Int? {
        guard service != IO_OBJECT_NULL else { return nil }
        guard let value = IORegistryEntryCreateCFProperty(
            service, key as CFString, kCFAllocatorDefault, 0
        )?.takeRetainedValue() else { return nil }
        return value as? Int
    }

    private static func registryBool(_ service: io_object_t, _ key: String) -> Bool? {
        guard service != IO_OBJECT_NULL else { return nil }
        guard let value = IORegistryEntryCreateCFProperty(
            service, key as CFString, kCFAllocatorDefault, 0
        )?.takeRetainedValue() else { return nil }
        return value as? Bool
    }
}
