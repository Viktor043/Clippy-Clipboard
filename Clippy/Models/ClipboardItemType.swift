import Foundation

enum ClipboardItemType: String, Codable, CaseIterable {
    case text
    case richText
    case image
    case file
}
