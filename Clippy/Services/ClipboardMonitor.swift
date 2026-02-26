import Foundation
import AppKit

final class ClipboardMonitor: ObservableObject {
    private var timer: Timer?
    private var lastChangeCount: Int = 0
    private let pasteboard = NSPasteboard.general
    private let pollInterval: TimeInterval

    var onNewClip: ((ClipboardItem) -> Void)?

    /// Dynamic character limit — reads from UserDefaults so settings take effect immediately
    private var maxCharacters: Int {
        let stored = UserDefaults.standard.integer(forKey: "maxCharactersPerItem")
        return stored > 0 ? stored : AppConstants.maxCharactersPerItem
    }

    init(pollInterval: TimeInterval = AppConstants.pollInterval) {
        self.pollInterval = pollInterval
        self.lastChangeCount = pasteboard.changeCount
    }

    func startMonitoring() {
        // Use .common run loop mode so the timer fires even when menus/popovers are open
        let timer = Timer(timeInterval: pollInterval, repeats: true) { [weak self] _ in
            self?.checkForChanges()
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - Private

    private func checkForChanges() {
        let currentCount = pasteboard.changeCount
        guard currentCount != lastChangeCount else { return }
        lastChangeCount = currentCount

        // Skip items placed by Clippy itself
        if let types = pasteboard.types,
           types.contains(NSPasteboard.PasteboardType(AppConstants.selfCopyPasteboardType)) {
            return
        }

        // Check if the source app is in the excluded list
        if let sourceApp = NSWorkspace.shared.frontmostApplication?.bundleIdentifier {
            let excludedApps = UserDefaults.standard.stringArray(forKey: "excludedApps") ?? []
            if excludedApps.contains(sourceApp) {
                return
            }
        }

        if let item = extractClipboardItem() {
            onNewClip?(item)
        }
    }

    /// Truncate text to the configured character limit
    private func truncateText(_ text: String?) -> String? {
        guard let text = text else { return nil }
        let limit = maxCharacters
        if text.count <= limit { return text }
        return String(text.prefix(limit)) + "…"
    }

    private func extractClipboardItem() -> ClipboardItem? {
        let types = pasteboard.types ?? []
        let typeStrings = types.map { $0.rawValue }
        let sourceApp = NSWorkspace.shared.frontmostApplication?.bundleIdentifier

        // File URLs
        if types.contains(.fileURL) {
            guard UserDefaults.standard.object(forKey: "captureFiles") as? Bool ?? true else { return nil }

            if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL],
               let url = urls.first {
                let icon = NSWorkspace.shared.icon(forFile: url.path)
                let thumbData = icon.thumbnailImage(maxSize: AppConstants.thumbnailMaxSize)?.pngData()

                return ClipboardItem(
                    id: UUID(),
                    type: .file,
                    timestamp: Date(),
                    textContent: url.path,
                    imageData: nil,
                    thumbnailData: thumbData,
                    rawData: url.absoluteString.data(using: .utf8),
                    sourceAppBundleID: sourceApp,
                    pasteboardTypes: typeStrings
                )
            }
        }

        // Images (PNG or TIFF)
        if types.contains(.png) || types.contains(.tiff) {
            guard UserDefaults.standard.object(forKey: "captureImages") as? Bool ?? true else { return nil }

            let imageData: Data?
            if types.contains(.png) {
                imageData = pasteboard.data(forType: .png)
            } else {
                imageData = pasteboard.data(forType: .tiff)
            }

            if let imageData = imageData, let image = NSImage(data: imageData) {
                let thumbData = image.thumbnailImage(maxSize: AppConstants.thumbnailMaxSize)?.pngData()
                let pngData = image.pngData()

                return ClipboardItem(
                    id: UUID(),
                    type: .image,
                    timestamp: Date(),
                    textContent: imageSizeDescription(image),
                    imageData: pngData,
                    thumbnailData: thumbData,
                    rawData: pngData,
                    sourceAppBundleID: sourceApp,
                    pasteboardTypes: typeStrings
                )
            }
        }

        // Rich text
        if types.contains(.rtf), let rtfData = pasteboard.data(forType: .rtf) {
            guard UserDefaults.standard.object(forKey: "captureRichText") as? Bool ?? true else { return nil }

            let attrString = NSAttributedString(rtf: rtfData, documentAttributes: nil)
            let plainText = truncateText(attrString?.string)

            return ClipboardItem(
                id: UUID(),
                type: .richText,
                timestamp: Date(),
                textContent: plainText,
                imageData: nil,
                thumbnailData: nil,
                rawData: rtfData,
                sourceAppBundleID: sourceApp,
                pasteboardTypes: typeStrings
            )
        }

        // Plain text
        if types.contains(.string), let text = pasteboard.string(forType: .string), !text.isEmpty {
            guard UserDefaults.standard.object(forKey: "captureText") as? Bool ?? true else { return nil }

            let truncated = truncateText(text)

            return ClipboardItem(
                id: UUID(),
                type: .text,
                timestamp: Date(),
                textContent: truncated,
                imageData: nil,
                thumbnailData: nil,
                rawData: truncated?.data(using: .utf8),
                sourceAppBundleID: sourceApp,
                pasteboardTypes: typeStrings
            )
        }

        return nil
    }

    private func imageSizeDescription(_ image: NSImage) -> String {
        let size = image.size
        return "\(Int(size.width)) × \(Int(size.height))"
    }
}
