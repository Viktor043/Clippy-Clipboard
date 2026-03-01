// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "BatteryBar",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "BatteryBar",
            path: "BatteryBar",
            exclude: ["Info.plist"],
            linkerSettings: [
                .linkedFramework("IOKit"),
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
            ]
        ),
        .testTarget(
            name: "BatteryBarTests",
            dependencies: ["BatteryBar"],
            path: "BatteryBarTests"
        ),
    ]
)
