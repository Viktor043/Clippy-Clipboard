import Foundation
import AppKit

struct ClipboardItem: Identifiable, Equatable {
    let id: UUID
    let type: ClipboardItemType
    let timestamp: Date
    let textContent: String?
    let imageData: Data?
    let thumbnailData: Data?
    let rawData: Data?
    let sourceAppBundleID: String?
    let pasteboardTypes: [String]

    var displayTitle: String {
        switch type {
        case .text:
            return textContent?.prefix(80).description ?? "Text"
        case .richText:
            return textContent?.prefix(80).description ?? "Rich Text"
        case .image:
            return "Image"
        case .file:
            if let path = textContent {
                return (path as NSString).lastPathComponent
            }
            return "File"
        }
    }

    var thumbnail: NSImage? {
        if let data = thumbnailData {
            return NSImage(data: data)
        }
        return nil
    }

    static func == (lhs: ClipboardItem, rhs: ClipboardItem) -> Bool {
        lhs.id == rhs.id
    }
}
