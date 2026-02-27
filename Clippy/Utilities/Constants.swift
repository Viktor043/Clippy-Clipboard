import Foundation

enum AppConstants {
    static let pollInterval: TimeInterval = 0.5
    static let maxHistoryItems: Int = 100
    static let thumbnailMaxSize: CGFloat = 64
    static let popoverWidth: CGFloat = 320
    static let popoverHeight: CGFloat = 480
    static let clippyWindowSize = CGSize(width: 130, height: 160)
    static let clippyEdgeInset: CGFloat = 20
    static let balloonDismissDelay: TimeInterval = 4.0
    static let idleAnimationMinDelay: TimeInterval = 5.0
    static let idleAnimationMaxDelay: TimeInterval = 15.0
    static let selfCopyPasteboardType = "com.snrgystudios.clippy.self-copy"
    static let maxCharactersPerItem: Int = 20_000

    // UI consistency
    static let panelCornerRadius: CGFloat = 10

    // Clippy size scaling
    static let clippyScaleOptions: [Double] = [0.5, 0.75, 1.0, 1.25, 1.5]
    static let clippyDefaultScale: Double = 1.0

    // Detail preview
    static let defaultDetailDelay: Double = 2.0

    // Default font
    static let defaultFontName: String = "System"
    static let defaultFontSize: Double = 12.0

    // Menu size customisation
    static let popoverWidthKey = "popoverWidth"
    static let popoverHeightKey = "popoverHeight"
    static let defaultPopoverWidth: CGFloat = 320
    static let defaultPopoverHeight: CGFloat = 480
    static let minPopoverWidth: CGFloat = 250
    static let maxPopoverWidth: CGFloat = 500
    static let minPopoverHeight: CGFloat = 350
    static let maxPopoverHeight: CGFloat = 700

    // Panel appearance customisation
    static let panelOpacityKey = "panelOpacity"
    static let defaultPanelOpacity: Double = 1.0
    static let minPanelOpacity: Double = 0.5
    static let maxPanelOpacity: Double = 1.0
    static let panelBlurKey = "panelBlurIntensity"
    static let defaultPanelBlur: Double = 1.0
    static let minPanelBlur: Double = 0.3
    static let maxPanelBlur: Double = 1.0

    // UserDefaults keys
    static let showMenuBarIconKey = "showMenuBarIcon"
    static let clippyScaleKey = "clippyScale"
    static let positionLockedKey = "positionLocked"
    static let clippyPositionXKey = "clippyPositionX"
    static let clippyPositionYKey = "clippyPositionY"
    static let hasStoredPositionKey = "hasStoredClippyPosition"
    static let detailDelayKey = "detailHoverDelay"
    static let appFontNameKey = "appFontName"
    static let appFontSizeKey = "appFontSize"
}
