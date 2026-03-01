import AppKit

/// Draws a battery icon with the percentage number rendered inside it.
/// White fill, yellow below 10%, Michroma font. Green plug icon when plugged in.
enum BatteryIconRenderer {

    private static let iconHeight: CGFloat = 16

    // Battery body (20% shorter than original 28pt = 22.4pt)
    private static let bodyX: CGFloat = 0.5
    private static let bodyY: CGFloat = 2
    private static let bodyWidth: CGFloat = 22.4
    private static let bodyHeight: CGFloat = 12
    private static let bodyRadius: CGFloat = 3.5

    // Terminal nub
    private static let termWidth: CGFloat = 2.5
    private static let termHeight: CGFloat = 6
    private static let termRadius: CGFloat = 1.5

    // Fill inset from the body outline
    private static let fillInset: CGFloat = 1.5

    // Outline color
    private static let outlineColor = NSColor.white.withAlphaComponent(0.55)

    /// Michroma font, falling back to Futura then system font.
    private static func numberFont(size: CGFloat) -> NSFont {
        return NSFont(name: "Michroma-Regular", size: size)
            ?? NSFont(name: "Futura-CondensedExtraBold", size: size)
            ?? NSFont.monospacedDigitSystemFont(ofSize: size, weight: .heavy)
    }

    /// Create the battery status bar image.
    static func icon(percentage: Int, isCharging: Bool, isPluggedIn: Bool) -> NSImage {
        let width = bodyX + bodyWidth + 1 + termWidth + 1
        let size = NSSize(width: width, height: iconHeight)

        let image = NSImage(size: size, flipped: false) { _ in
            drawBatteryBody()
            drawFill(percentage: percentage)
            if isPluggedIn {
                drawPluggedInLayout(percentage: percentage)
            } else {
                drawCenteredText(percentage: percentage)
            }
            return true
        }
        image.isTemplate = false
        return image
    }

    // MARK: - Drawing helpers

    private static func drawBatteryBody() {
        let bodyRect = NSRect(x: bodyX, y: bodyY, width: bodyWidth, height: bodyHeight)
        let bodyPath = NSBezierPath(roundedRect: bodyRect, xRadius: bodyRadius, yRadius: bodyRadius)
        bodyPath.lineWidth = 1.2
        outlineColor.setStroke()
        bodyPath.stroke()

        // Terminal nub
        let termX = bodyX + bodyWidth + 0.5
        let termY = bodyY + (bodyHeight - termHeight) / 2
        let termRect = NSRect(x: termX, y: termY, width: termWidth, height: termHeight)
        let termPath = NSBezierPath(roundedRect: termRect, xRadius: termRadius, yRadius: termRadius)
        outlineColor.setFill()
        termPath.fill()
    }

    private static func drawFill(percentage: Int) {
        let pct = CGFloat(max(0, min(percentage, 100))) / 100.0
        let maxFillWidth = bodyWidth - fillInset * 2
        let fillWidth = maxFillWidth * pct
        guard fillWidth > 0 else { return }

        let fillRect = NSRect(
            x: bodyX + fillInset,
            y: bodyY + fillInset,
            width: fillWidth,
            height: bodyHeight - fillInset * 2
        )
        let fillPath = NSBezierPath(roundedRect: fillRect, xRadius: 2, yRadius: 2)

        let color: NSColor = percentage < 10
            ? NSColor.systemYellow.withAlphaComponent(0.85)
            : NSColor.white.withAlphaComponent(0.85)
        color.setFill()
        fillPath.fill()
    }

    /// Not plugged in: percentage number centered in the battery body.
    private static func drawCenteredText(percentage: Int) {
        let text = "\(percentage)"
        let font = numberFont(size: 8.6)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.black,
        ]
        let textSize = text.size(withAttributes: attributes)
        let origin = NSPoint(
            x: bodyX + bodyWidth / 2 - textSize.width / 2,
            y: bodyY + bodyHeight / 2 - textSize.height / 2
        )
        text.draw(at: origin, withAttributes: attributes)
    }

    /// Plugged in: number on the left, plug icon on the right.
    private static func drawPluggedInLayout(percentage: Int) {
        let innerLeft = bodyX + fillInset + 1
        let centerY = bodyY + bodyHeight / 2

        // Percentage number on the left
        let text = "\(percentage)"
        let font = numberFont(size: 8.6)
        let textAttrs: [NSAttributedString.Key: Any] = [
            .font: font,
            .foregroundColor: NSColor.black,
        ]
        let textSize = text.size(withAttributes: textAttrs)
        let textOrigin = NSPoint(
            x: innerLeft,
            y: centerY - textSize.height / 2
        )
        text.draw(at: textOrigin, withAttributes: textAttrs)

        // Plug icon: 20% shifted left from previous position
        let originalPlugLeft = innerLeft + textSize.width + 1.5
        let plugShift = (bodyX + bodyWidth - originalPlugLeft) * 0.20
        let plugLeft = originalPlugLeft - plugShift
        let plugWidth: CGFloat = 10.5 * 1.5  // 50% wider
        let plugHeight: CGFloat = iconHeight
        let plugRect = NSRect(
            x: plugLeft,
            y: 0,
            width: plugWidth,
            height: plugHeight
        )
        drawPlug(in: plugRect)
    }

    // MARK: - Plug icon

    /// Draws a power plug icon with shadow: two prongs on top, body in the middle, cord at the bottom.
    /// Fully opaque green, no transparency.
    private static func drawPlug(in rect: NSRect) {
        let x = rect.origin.x
        let y = rect.origin.y
        let w = rect.size.width
        let h = rect.size.height
        let midX = x + w / 2
        let color = NSColor.systemGreen

        let prongWidth: CGFloat = 1.4
        let prongHeight: CGFloat = h * 0.22
        let prongGap: CGFloat = w * 0.32
        let prongTop = y + h
        let leftProngX = midX - prongGap / 2 - prongWidth / 2
        let rightProngX = midX + prongGap / 2 - prongWidth / 2
        let prongBottom = prongTop - prongHeight

        let plugBodyHeight: CGFloat = h * 0.30
        let plugBodyWidth: CGFloat = w * 0.70
        let plugBodyTop = prongBottom
        let plugBodyBottom = plugBodyTop - plugBodyHeight

        let cordWidth: CGFloat = 1.4
        let cordTop = plugBodyBottom
        let cordBottom = y

        // Save graphics state for shadow
        NSGraphicsContext.current?.saveGraphicsState()

        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.5)
        shadow.shadowOffset = NSSize(width: 0.5, height: -0.5)
        shadow.shadowBlurRadius = 2.0
        shadow.set()

        color.setFill()

        // Left prong
        NSBezierPath(roundedRect:
            NSRect(x: leftProngX, y: prongBottom, width: prongWidth, height: prongHeight),
            xRadius: 0.4, yRadius: 0.4).fill()
        // Right prong
        NSBezierPath(roundedRect:
            NSRect(x: rightProngX, y: prongBottom, width: prongWidth, height: prongHeight),
            xRadius: 0.4, yRadius: 0.4).fill()
        // Plug body
        NSBezierPath(roundedRect:
            NSRect(x: midX - plugBodyWidth / 2, y: plugBodyBottom, width: plugBodyWidth, height: plugBodyHeight),
            xRadius: 1.2, yRadius: 1.2).fill()
        // Cord
        NSBezierPath(roundedRect:
            NSRect(x: midX - cordWidth / 2, y: cordBottom, width: cordWidth, height: cordTop - cordBottom),
            xRadius: 0.4, yRadius: 0.4).fill()

        NSGraphicsContext.current?.restoreGraphicsState()
    }
}
