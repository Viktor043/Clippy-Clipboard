import AppKit

final class ClippyBalloonView: NSView {
    private let messageLabel: NSTextField
    private let padding: CGFloat = 8

    init(message: String) {
        messageLabel = NSTextField(labelWithString: message)
        super.init(frame: .zero)

        wantsLayer = true
        // Use system colors for dark mode support
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        layer?.cornerRadius = 8
        layer?.borderWidth = 1
        layer?.borderColor = NSColor.separatorColor.cgColor
        shadow = NSShadow()
        shadow?.shadowColor = NSColor.black.withAlphaComponent(0.15)
        shadow?.shadowOffset = NSSize(width: 0, height: -2)
        shadow?.shadowBlurRadius = 4

        messageLabel.font = NSFont.systemFont(ofSize: 11)
        messageLabel.textColor = .labelColor
        messageLabel.alignment = .center
        messageLabel.lineBreakMode = .byWordWrapping
        messageLabel.maximumNumberOfLines = 3
        messageLabel.translatesAutoresizingMaskIntoConstraints = false

        addSubview(messageLabel)

        NSLayoutConstraint.activate([
            messageLabel.topAnchor.constraint(equalTo: topAnchor, constant: padding),
            messageLabel.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -padding),
            messageLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: padding),
            messageLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -padding)
        ])
    }

    // Update colors when system appearance changes (light ↔ dark mode)
    override func updateLayer() {
        layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        layer?.borderColor = NSColor.separatorColor.cgColor
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
