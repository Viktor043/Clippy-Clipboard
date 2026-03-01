// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PhotonicTwin",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "PhotonicTwin",
            path: "Sources",
            resources: [
                .process("Shaders")
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("Charts", .when(platforms: [.macOS])),
            ]
        )
    ]
)
