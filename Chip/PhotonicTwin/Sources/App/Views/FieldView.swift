import SwiftUI
import AppKit

/// Renders a 2D float array as a heatmap image (viridis-inspired colormap).
struct FieldView: View {
    let data: [Float]
    let nx: Int
    let ny: Int

    var body: some View {
        if let image = renderHeatmap() {
            Image(nsImage: image)
                .resizable()
                .interpolation(.none)
                .aspectRatio(contentMode: .fit)
                .background(Color.black)
                .cornerRadius(6)
        } else {
            Rectangle()
                .fill(Color.black)
                .cornerRadius(6)
                .overlay(Text("No data").foregroundColor(.gray))
        }
    }

    private func renderHeatmap() -> NSImage? {
        guard data.count == nx * ny, nx > 0, ny > 0 else { return nil }

        // Find range for normalization
        var minVal: Float = .infinity
        var maxVal: Float = -.infinity
        for v in data {
            if v < minVal { minVal = v }
            if v > maxVal { maxVal = v }
        }
        let range = maxVal - minVal
        let invRange: Float = range > 0 ? 1.0 / range : 1.0

        // Create RGBA pixel buffer
        var pixels = [UInt8](repeating: 0, count: nx * ny * 4)
        for i in 0..<(nx * ny) {
            let t = (data[i] - minVal) * invRange  // 0..1
            let (r, g, b) = viridis(t)
            pixels[i * 4 + 0] = r
            pixels[i * 4 + 1] = g
            pixels[i * 4 + 2] = b
            pixels[i * 4 + 3] = 255
        }

        // Create CGImage
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &pixels,
            width: ny,   // columns = width
            height: nx,  // rows = height
            bitsPerComponent: 8,
            bytesPerRow: ny * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cgImage = context.makeImage() else { return nil }

        return NSImage(cgImage: cgImage, size: NSSize(width: ny, height: nx))
    }

    /// Viridis-inspired colormap: dark purple → blue → green → yellow
    private func viridis(_ t: Float) -> (UInt8, UInt8, UInt8) {
        let t = max(0, min(1, t))
        let r: Float
        let g: Float
        let b: Float

        if t < 0.25 {
            let s = t / 0.25
            r = 0.267 + s * (0.282 - 0.267)
            g = 0.004 + s * (0.140 - 0.004)
            b = 0.329 + s * (0.458 - 0.329)
        } else if t < 0.5 {
            let s = (t - 0.25) / 0.25
            r = 0.282 + s * (0.127 - 0.282)
            g = 0.140 + s * (0.566 - 0.140)
            b = 0.458 + s * (0.551 - 0.458)
        } else if t < 0.75 {
            let s = (t - 0.5) / 0.25
            r = 0.127 + s * (0.741 - 0.127)
            g = 0.566 + s * (0.873 - 0.566)
            b = 0.551 + s * (0.150 - 0.551)
        } else {
            let s = (t - 0.75) / 0.25
            r = 0.741 + s * (0.993 - 0.741)
            g = 0.873 + s * (0.906 - 0.873)
            b = 0.150 + s * (0.144 - 0.150)
        }

        return (UInt8(r * 255), UInt8(g * 255), UInt8(b * 255))
    }
}
