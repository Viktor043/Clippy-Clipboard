#!/usr/bin/env swift

/// Standalone test runner for BatteryBar — no XCTest dependency needed.
/// Validates BatteryService accuracy against pmset and tests icon rendering.

import Foundation
import IOKit
import IOKit.ps
import AppKit

// ============================================================
// MARK: - Minimal Test Framework
// ============================================================

var passCount = 0
var failCount = 0
var currentTest = ""

func test(_ name: String, _ body: () throws -> Void) {
    currentTest = name
    do {
        try body()
        passCount += 1
        print("  PASS  \(name)")
    } catch {
        failCount += 1
        print("  FAIL  \(name): \(error)")
    }
}

struct AssertionError: Error, CustomStringConvertible {
    let description: String
}

func assert(_ condition: Bool, _ message: String = "Assertion failed") throws {
    guard condition else { throw AssertionError(description: message) }
}

// ============================================================
// MARK: - Inline BatteryService (same logic as the app)
// ============================================================

struct BatteryInfo {
    let percentage: Int
    let isCharging: Bool
    let isPluggedIn: Bool
    let isFullyCharged: Bool
    let cycleCount: Int?
    let health: Int?
    let temperature: Double?
    let currentCapacity: Int?
    let maxCapacity: Int?
    let designCapacity: Int?
    let voltage: Double?
}

func readBattery() -> BatteryInfo {
    // IOPowerSources
    let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
    let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as Array
    var percentage = 0
    for source in sources {
        guard let desc = IOPSGetPowerSourceDescription(snapshot, source)?
            .takeUnretainedValue() as? [String: Any] else { continue }
        if desc[kIOPSTypeKey] as? String == kIOPSInternalBatteryType {
            percentage = desc[kIOPSCurrentCapacityKey] as? Int ?? 0
        }
    }
    let powerType = IOPSGetProvidingPowerSourceType(snapshot).takeRetainedValue() as String
    let isPluggedIn = powerType == kIOPSACPowerValue

    // AppleSmartBattery
    let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("AppleSmartBattery"))
    defer { if service != IO_OBJECT_NULL { IOObjectRelease(service) } }

    func regInt(_ key: String) -> Int? {
        guard service != IO_OBJECT_NULL else { return nil }
        return IORegistryEntryCreateCFProperty(service, key as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? Int
    }
    func regBool(_ key: String) -> Bool? {
        guard service != IO_OBJECT_NULL else { return nil }
        return IORegistryEntryCreateCFProperty(service, key as CFString, kCFAllocatorDefault, 0)?
            .takeRetainedValue() as? Bool
    }

    let isCharging = regBool("IsCharging") ?? false
    let isFullyCharged = regBool("FullyCharged") ?? false
    let cycleCount = regInt("CycleCount")
    let designCap = regInt("DesignCapacity")
    let tempRaw = regInt("Temperature")
    let voltRaw = regInt("Voltage")

    // Use raw capacity keys for accurate mAh values (modern macOS returns 100 for MaxCapacity)
    let rawMaxCap = regInt("AppleRawMaxCapacity")
    let nominalCap = regInt("NominalChargeCapacity")
    let ioMaxCap = regInt("MaxCapacity")
    let maxCap: Int? = rawMaxCap ?? nominalCap ?? (ioMaxCap.flatMap { $0 > 1000 ? $0 : nil })

    let rawCurCap = regInt("AppleRawCurrentCapacity")
    let ioCurCap = regInt("CurrentCapacity")
    let currentCap: Int? = rawCurCap ?? (ioCurCap.flatMap { $0 > 1000 ? $0 : nil })

    let temperature: Double? = tempRaw.map { Double($0) / 100.0 }
    let voltage: Double? = voltRaw.map { Double($0) / 1000.0 }
    let health: Int? = (maxCap != nil && designCap != nil && designCap! > 0)
        ? Int(round(Double(maxCap!) / Double(designCap!) * 100.0))
        : nil

    return BatteryInfo(
        percentage: percentage, isCharging: isCharging, isPluggedIn: isPluggedIn,
        isFullyCharged: isFullyCharged, cycleCount: cycleCount, health: health,
        temperature: temperature, currentCapacity: currentCap, maxCapacity: maxCap,
        designCapacity: designCap, voltage: voltage
    )
}

/// Parse pmset output for comparison.
func readPmset() -> (percentage: Int, isAC: Bool, isCharging: Bool)? {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
    process.arguments = ["-g", "batt"]
    let pipe = Pipe()
    process.standardOutput = pipe
    do { try process.run() } catch { return nil }
    process.waitUntilExit()
    let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

    // Parse percentage
    guard let pctRange = output.range(of: #"(\d{1,3})%"#, options: .regularExpression) else { return nil }
    guard let pct = Int(output[pctRange].dropLast()) else { return nil }

    let isAC = output.contains("AC Power")
    let isCharging = output.contains("charging") && !output.contains("discharging")
        && !output.contains("not charging") && !output.contains("charged")

    return (pct, isAC, isCharging)
}

// ============================================================
// MARK: - Test Suite
// ============================================================

print("\n=== BatteryBar Test Suite ===\n")
print("--- BatteryService Tests ---\n")

let info = readBattery()

test("Battery percentage is 0-100") {
    try assert(info.percentage >= 0 && info.percentage <= 100,
        "Got \(info.percentage)%")
}

test("Charging state is consistent (charging implies plugged in)") {
    if info.isCharging {
        try assert(info.isPluggedIn, "Charging but not plugged in")
    }
    if info.isFullyCharged {
        try assert(info.isPluggedIn, "Fully charged but not plugged in")
    }
}

test("Health is reasonable (1-150%) if available") {
    if let health = info.health {
        try assert(health > 0 && health <= 150, "Health: \(health)%")
    }
}

test("Cycle count is non-negative if available") {
    if let cycles = info.cycleCount {
        try assert(cycles >= 0, "Cycle count: \(cycles)")
    }
}

test("Temperature is 0-80°C if available") {
    if let temp = info.temperature {
        try assert(temp > 0 && temp < 80, "Temp: \(temp)°C")
    }
}

test("Voltage is 5-20V if available") {
    if let volts = info.voltage {
        try assert(volts > 5 && volts < 20, "Voltage: \(volts)V")
    }
}

test("Capacities are consistent") {
    if let cur = info.currentCapacity, let max = info.maxCapacity {
        try assert(max >= cur, "Max (\(max)) < Current (\(cur))")
    }
    if let max = info.maxCapacity, let des = info.designCapacity {
        try assert(max > 0 && des > 0, "Max: \(max), Design: \(des)")
    }
}

// --- Cross-validation with pmset ---

print("\n--- Cross-validation with pmset ---\n")

if let pmset = readPmset() {
    test("Percentage matches pmset (within 1%)") {
        let diff = abs(info.percentage - pmset.percentage)
        try assert(diff <= 1,
            "Ours: \(info.percentage)%, pmset: \(pmset.percentage)%, diff: \(diff)%")
    }

    test("AC power state matches pmset") {
        try assert(info.isPluggedIn == pmset.isAC,
            "Ours: \(info.isPluggedIn), pmset: \(pmset.isAC)")
    }

    test("Charging state matches pmset") {
        if pmset.isCharging {
            try assert(info.isCharging || info.isFullyCharged,
                "pmset says charging but we say isCharging=\(info.isCharging), isFullyCharged=\(info.isFullyCharged)")
        }
    }

    print("\n--- Comparison ---")
    print("  BatteryService: \(info.percentage)% | plugged: \(info.isPluggedIn) | charging: \(info.isCharging) | full: \(info.isFullyCharged)")
    print("  pmset:          \(pmset.percentage)% | AC: \(pmset.isAC) | charging: \(pmset.isCharging)")
} else {
    print("  SKIP  Could not parse pmset output")
}

// --- Icon Rendering Tests ---

print("\n--- BatteryIconRenderer Tests ---\n")

// Updated icon: white fill, yellow below 10%, black text, Apple bolt, rounder corners
let bodyWidth: CGFloat = 22.4
let expectedWidth: CGFloat = 0.5 + bodyWidth + 1 + 2.5 + 1

func createBatteryIcon(percentage: Int, isCharging: Bool) -> NSImage {
    let height: CGFloat = 16
    let image = NSImage(size: NSSize(width: expectedWidth, height: height), flipped: false) { _ in
        let bodyRect = NSRect(x: 0.5, y: 2, width: bodyWidth, height: 12)
        let bodyPath = NSBezierPath(roundedRect: bodyRect, xRadius: 3.5, yRadius: 3.5)
        bodyPath.lineWidth = 1.2
        NSColor.labelColor.withAlphaComponent(0.85).setStroke()
        bodyPath.stroke()

        let termX = 0.5 + bodyWidth + 0.5
        let termRect = NSRect(x: termX, y: 5, width: 2.5, height: 6)
        NSColor.labelColor.withAlphaComponent(0.85).setFill()
        NSBezierPath(roundedRect: termRect, xRadius: 1.5, yRadius: 1.5).fill()

        let pct = CGFloat(max(0, min(percentage, 100))) / 100.0
        let fillWidth = (bodyWidth - 3) * pct
        if fillWidth > 0 {
            let fillRect = NSRect(x: 2, y: 3.5, width: fillWidth, height: 9)
            let color: NSColor = percentage < 10
                ? NSColor.systemYellow.withAlphaComponent(0.85)
                : NSColor.white.withAlphaComponent(0.85)
            color.setFill()
            NSBezierPath(roundedRect: fillRect, xRadius: 2, yRadius: 2).fill()
        }

        let text = "\(percentage)"
        let font = NSFont.monospacedDigitSystemFont(ofSize: isCharging ? 7.5 : 8.5, weight: .semibold)
        let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.black]
        let textSize = text.size(withAttributes: attrs)
        let origin = NSPoint(x: 0.5 + bodyWidth/2 - textSize.width/2, y: 2 + 12/2 - textSize.height/2)
        text.draw(at: origin, withAttributes: attrs)
        return true
    }
    image.isTemplate = false
    return image
}

test("Icon has correct width") {
    let img = createBatteryIcon(percentage: 50, isCharging: false)
    try assert(img.size.width == expectedWidth,
        "Size: \(img.size), expected width: \(expectedWidth)")
}

test("Icon is not a template image (uses colors)") {
    let img = createBatteryIcon(percentage: 50, isCharging: false)
    try assert(!img.isTemplate, "Should not be template (custom colors)")
}

test("Low battery (below 10%) has different pixels than normal") {
    let low = createBatteryIcon(percentage: 5, isCharging: false)
    let normal = createBatteryIcon(percentage: 50, isCharging: false)
    try assert(low.tiffRepresentation != normal.tiffRepresentation,
        "Yellow fill (5%) should differ from green fill (50%)")
}

test("Icons render for all boundary percentages") {
    for pct in [0, 1, 5, 10, 20, 35, 50, 75, 99, 100] {
        let img = createBatteryIcon(percentage: pct, isCharging: false)
        try assert(img.size.width > 0, "Failed for \(pct)%")
    }
}

test("Charging icon renders") {
    let img = createBatteryIcon(percentage: 60, isCharging: true)
    try assert(img.size.width == expectedWidth, "Charging icon failed")
}

test("5% and 95% icons have different pixels") {
    let low = createBatteryIcon(percentage: 5, isCharging: false)
    let high = createBatteryIcon(percentage: 95, isCharging: false)
    try assert(low.tiffRepresentation != high.tiffRepresentation,
        "Should differ")
}

test("Charging vs non-charging icons differ") {
    let a = createBatteryIcon(percentage: 50, isCharging: false)
    let b = createBatteryIcon(percentage: 50, isCharging: true)
    try assert(a.tiffRepresentation != b.tiffRepresentation,
        "Should differ")
}

test("Out-of-range percentages do not crash") {
    let neg = createBatteryIcon(percentage: -5, isCharging: false)
    let over = createBatteryIcon(percentage: 120, isCharging: false)
    try assert(neg.size.width > 0 && over.size.width > 0, "Should not crash")
}

// ============================================================
// MARK: - Summary
// ============================================================

print("\n=== Results: \(passCount) passed, \(failCount) failed ===\n")

if failCount > 0 {
    print("SOME TESTS FAILED")
    exit(1)
} else {
    print("ALL TESTS PASSED")

    // Print detailed battery info
    print("\n--- Current Battery Status ---")
    print("  Percentage:    \(info.percentage)%")
    print("  Plugged In:    \(info.isPluggedIn)")
    print("  Charging:      \(info.isCharging)")
    print("  Fully Charged: \(info.isFullyCharged)")
    if let h = info.health { print("  Health:        \(h)%") }
    if let c = info.cycleCount { print("  Cycle Count:   \(c)") }
    if let t = info.temperature { print("  Temperature:   \(String(format: "%.1f", t))°C") }
    if let v = info.voltage { print("  Voltage:       \(String(format: "%.2f", v))V") }
    if let cur = info.currentCapacity { print("  Current Cap:   \(cur) mAh") }
    if let max = info.maxCapacity { print("  Max Capacity:  \(max) mAh") }
    if let des = info.designCapacity { print("  Design Cap:    \(des) mAh") }
}
