import XCTest
@testable import BatteryBar

final class BatteryIconRendererTests: XCTestCase {

    /// Verify the icon has the correct dimensions.
    func testIconHasCorrectSize() {
        let image = BatteryIconRenderer.icon(percentage: 50, isCharging: false, isPluggedIn: false)
        XCTAssertEqual(image.size.width, 34, "Icon width should be 34pt")
        XCTAssertEqual(image.size.height, 16, "Icon height should be 16pt")
    }

    /// Verify the icon is not a template (so custom colors work).
    func testIconIsNotTemplate() {
        let image = BatteryIconRenderer.icon(percentage: 50, isCharging: false, isPluggedIn: false)
        XCTAssertFalse(image.isTemplate, "Icon should not be a template image")
    }

    /// Verify icons are generated for edge-case percentages without crashing.
    func testIconGenerationAtBoundaries() {
        let edgeCases = [0, 1, 5, 10, 20, 35, 50, 75, 99, 100]
        for pct in edgeCases {
            let image = BatteryIconRenderer.icon(percentage: pct, isCharging: false, isPluggedIn: false)
            XCTAssertNotNil(image, "Icon should be generated for \(pct)%")
            XCTAssertGreaterThan(image.size.width, 0)
        }
    }

    /// Verify charging icon is generated.
    func testChargingIconGeneration() {
        let image = BatteryIconRenderer.icon(percentage: 60, isCharging: true, isPluggedIn: true)
        XCTAssertNotNil(image)
        XCTAssertEqual(image.size.width, 34)
    }

    /// Verify icon content differs between low and high battery.
    func testIconsDifferByPercentage() {
        let low = BatteryIconRenderer.icon(percentage: 5, isCharging: false, isPluggedIn: false)
        let high = BatteryIconRenderer.icon(percentage: 95, isCharging: false, isPluggedIn: false)

        // Convert to TIFF data to compare pixel content
        let lowData = low.tiffRepresentation
        let highData = high.tiffRepresentation
        XCTAssertNotNil(lowData)
        XCTAssertNotNil(highData)
        XCTAssertNotEqual(lowData, highData,
            "5% and 95% icons should have different pixel content")
    }

    /// Verify charging vs non-charging icons differ.
    func testChargingIconDiffersFromNonCharging() {
        let notCharging = BatteryIconRenderer.icon(percentage: 50, isCharging: false, isPluggedIn: false)
        let charging = BatteryIconRenderer.icon(percentage: 50, isCharging: true, isPluggedIn: true)

        let data1 = notCharging.tiffRepresentation
        let data2 = charging.tiffRepresentation
        XCTAssertNotEqual(data1, data2,
            "Charging and non-charging icons should differ")
    }

    /// Verify out-of-range percentages are clamped without crashing.
    func testOutOfRangePercentages() {
        let negative = BatteryIconRenderer.icon(percentage: -5, isCharging: false, isPluggedIn: false)
        XCTAssertNotNil(negative)

        let over = BatteryIconRenderer.icon(percentage: 120, isCharging: false, isPluggedIn: false)
        XCTAssertNotNil(over)
    }
}
