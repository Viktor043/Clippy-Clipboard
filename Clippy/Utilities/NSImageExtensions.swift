import AppKit

extension NSImage {
    func thumbnailImage(maxSize: CGFloat) -> NSImage? {
        let currentSize = self.size
        guard currentSize.width > 0, currentSize.height > 0 else { return nil }

        let scale: CGFloat
        if currentSize.width > currentSize.height {
            scale = maxSize / currentSize.width
        } else {
            scale = maxSize / currentSize.height
        }

        if scale >= 1.0 { return self }

        let newSize = NSSize(
            width: currentSize.width * scale,
            height: currentSize.height * scale
        )

        let thumbnail = NSImage(size: newSize)
        thumbnail.lockFocus()
        self.draw(
            in: NSRect(origin: .zero, size: newSize),
            from: NSRect(origin: .zero, size: currentSize),
            operation: .copy,
            fraction: 1.0
        )
        thumbnail.unlockFocus()
        return thumbnail
    }

    func pngData() -> Data? {
        guard let tiffData = self.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else { return nil }
        return bitmap.representation(using: .png, properties: [:])
    }
}
