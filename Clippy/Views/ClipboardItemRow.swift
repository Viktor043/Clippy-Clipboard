import SwiftUI

struct ClipboardItemRow: View {
    let item: ClipboardItem
    var onPaste: () -> Void
    var onDelete: () -> Void
    var onExpand: (() -> Void)?
    var onHoverItem: ((Bool) -> Void)?

    @State private var isHovering = false

    var body: some View {
        HStack(spacing: 6) {
            // Clickable content area — uses Button with custom style for reliable press highlight
            Button(action: onPaste) {
                HStack(spacing: 8) {
                    // Thumbnail for images, icon for files
                    if item.type == .image || item.type == .file, let thumbnail = item.thumbnail {
                        Image(nsImage: thumbnail)
                            .resizable()
                            .aspectRatio(contentMode: item.type == .image ? .fill : .fit)
                            .frame(width: 28, height: 28)
                            .cornerRadius(4)
                    }

                    // Content preview
                    VStack(alignment: .leading, spacing: 2) {
                        contentPreview
                            .lineLimit(1)
                            .font(.system(size: 12, weight: .semibold))

                        Text(relativeTime(item.timestamp))
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(RowPasteButtonStyle())

            // Buttons — always in layout, shown/hidden via opacity to prevent hover flicker
            // Expand button
            Button(action: { onExpand?() }) {
                HStack(spacing: 2) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 8, weight: .semibold))
                    Text("Expand")
                        .font(.system(size: 9, weight: .medium))
                }
                .foregroundColor(.accentColor)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.accentColor.opacity(0.1))
                .cornerRadius(4)
            }
            .buttonStyle(.plain)
            .help("Show full preview")
            .opacity(isHovering ? 1 : 0)
            .allowsHitTesting(isHovering)

            // Delete button
            Button(action: onDelete) {
                Image(systemName: "xmark")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .opacity(isHovering ? 1 : 0)
            .allowsHitTesting(isHovering)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .contentShape(Rectangle())  // Full row hit area for hover + click
        .background(isHovering ? Color.primary.opacity(0.06) : Color.clear)
        .cornerRadius(4)
        .onHover { hovering in
            isHovering = hovering
            onHoverItem?(hovering)
        }
    }

    @ViewBuilder
    private var contentPreview: some View {
        switch item.type {
        case .text, .richText:
            Text(item.textContent ?? "Empty")
                .foregroundColor(.primary)

        case .image:
            Text("Image — \(item.textContent ?? "")")
                .foregroundColor(.primary)

        case .file:
            Text(item.displayTitle)
                .foregroundColor(.primary)
        }
    }

    private func relativeTime(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Custom ButtonStyle for paste rows (press highlight without DragGesture)

/// Provides a press-highlight effect using SwiftUI's built-in `isPressed` state,
/// which doesn't interfere with `.onHover` tracking like `DragGesture(minimumDistance: 0)` does.
private struct RowPasteButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .background(configuration.isPressed ? Color.accentColor.opacity(0.18) : Color.clear)
            .cornerRadius(4)
    }
}
