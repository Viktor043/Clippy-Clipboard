import SwiftUI

@main
struct BatteryBarApp: App {
    @StateObject private var batteryService = BatteryService()

    var body: some Scene {
        MenuBarExtra {
            BatteryDetailView(batteryService: batteryService)
        } label: {
            let img = BatteryIconRenderer.icon(
                percentage: batteryService.batteryInfo.percentage,
                isCharging: batteryService.batteryInfo.isCharging,
                isPluggedIn: batteryService.batteryInfo.isPluggedIn
            )
            Image(nsImage: img)
        }
        .menuBarExtraStyle(.window)
    }
}
