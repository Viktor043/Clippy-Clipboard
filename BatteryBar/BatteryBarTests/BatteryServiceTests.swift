import XCTest
@testable import BatteryBar

final class BatteryServiceTests: XCTestCase {

    // MARK: - Live battery reading tests

    /// Verify we can read battery info and percentage is in valid range.
    func testReadBatteryReturnsValidPercentage() {
        let info = BatteryService.readBattery()
        XCTAssertGreaterThanOrEqual(info.percentage, 0, "Percentage should be >= 0")
        XCTAssertLessThanOrEqual(info.percentage, 100, "Percentage should be <= 100")
    }

    /// Verify the charging/plugged state is a valid combination.
    func testChargingStateIsConsistent() {
        let info = BatteryService.readBattery()
        // If charging, must be plugged in
        if info.isCharging {
            XCTAssertTrue(info.isPluggedIn, "If charging, must be plugged in")
        }
        // If fully charged, must be plugged in
        if info.isFullyCharged {
            XCTAssertTrue(info.isPluggedIn, "If fully charged, must be plugged in")
        }
    }

    /// Verify health percentage is reasonable if available.
    func testHealthIsReasonable() {
        let info = BatteryService.readBattery()
        if let health = info.health {
            XCTAssertGreaterThan(health, 0, "Health should be > 0")
            XCTAssertLessThanOrEqual(health, 150, "Health should be <= 150")
        }
    }

    /// Verify cycle count is a non-negative number if available.
    func testCycleCountIsNonNegative() {
        let info = BatteryService.readBattery()
        if let cycles = info.cycleCount {
            XCTAssertGreaterThanOrEqual(cycles, 0, "Cycle count should be >= 0")
        }
    }

    /// Verify temperature is in a reasonable range if available.
    func testTemperatureIsReasonable() {
        let info = BatteryService.readBattery()
        if let temp = info.temperature {
            XCTAssertGreaterThan(temp, 0, "Temperature should be > 0 °C")
            XCTAssertLessThan(temp, 80, "Temperature should be < 80 °C")
        }
    }

    /// Verify voltage is in a reasonable range if available.
    func testVoltageIsReasonable() {
        let info = BatteryService.readBattery()
        if let volts = info.voltage {
            XCTAssertGreaterThan(volts, 5, "Voltage should be > 5V for a MacBook battery")
            XCTAssertLessThan(volts, 20, "Voltage should be < 20V")
        }
    }

    /// Verify capacities are consistent.
    func testCapacitiesAreConsistent() {
        let info = BatteryService.readBattery()
        if let current = info.currentCapacity, let max = info.maxCapacity {
            XCTAssertGreaterThanOrEqual(max, current,
                "Max capacity should be >= current capacity")
        }
        if let max = info.maxCapacity, let design = info.designCapacity {
            XCTAssertGreaterThan(max, 0, "Max capacity should be > 0")
            XCTAssertGreaterThan(design, 0, "Design capacity should be > 0")
        }
    }

    // MARK: - Cross-validation with pmset

    /// Compare our reading with `pmset -g batt` to verify accuracy.
    func testPercentageMatchesPmset() throws {
        let info = BatteryService.readBattery()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-g", "batt"]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()

        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        // Parse percentage from pmset output, e.g. "95%;"
        let pattern = #"(\d{1,3})%"#
        guard let match = output.range(of: pattern, options: .regularExpression),
              let pmsetPct = Int(output[match].dropLast()) else {
            XCTFail("Could not parse pmset percentage from: \(output)")
            return
        }

        // Allow 1% tolerance since the readings happen at slightly different times
        XCTAssertEqual(info.percentage, pmsetPct, accuracy: 1,
            "BatteryService (\(info.percentage)%) should match pmset (\(pmsetPct)%)")
    }

    /// Verify charging state matches pmset.
    func testChargingStateMatchesPmset() throws {
        let info = BatteryService.readBattery()

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/pmset")
        process.arguments = ["-g", "batt"]
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()

        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        let pmsetIsCharging = output.contains("charging") && !output.contains("discharging")
            && !output.contains("not charging")
        let pmsetIsOnAC = output.contains("AC Power")

        XCTAssertEqual(info.isPluggedIn, pmsetIsOnAC,
            "AC power state should match pmset (ours: \(info.isPluggedIn), pmset AC: \(pmsetIsOnAC))")

        if pmsetIsCharging {
            XCTAssertTrue(info.isCharging || info.isFullyCharged,
                "Should be charging or fully charged when pmset says charging")
        }
    }
}

// Helper to allow XCTAssertEqual with Int tolerance
func XCTAssertEqual(_ a: Int, _ b: Int, accuracy: Int,
                    _ message: @autoclosure () -> String = "",
                    file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertTrue(abs(a - b) <= accuracy, message(), file: file, line: line)
}
