// AgentParser - Main agent container, loads .agent bundles
// Based on Cosmo/Clippy (https://github.com/Cosmo/Clippy) - adapted for personal use

import Foundation
import AppKit
import SpriteKit

enum AgentError: Error {
    case frameOutOfBounds
}

struct Agent {
    var character: AgentCharacter
    var balloon: AgentBalloon
    var animations: [AgentAnimation]
    var states: [AgentState]

    var agentURL: URL
    var resourceName: String
    var resourceNameWithSuffix: String
    var spriteMap: CGImage
    let soundsURL: URL

    init?(agentURL: URL) {
        self.agentURL = agentURL
        self.soundsURL = agentURL.appendingPathComponent("sounds")
        self.resourceNameWithSuffix = agentURL.lastPathComponent
        self.resourceName = resourceNameWithSuffix.replacingOccurrences(of: ".agent", with: "")

        let fileURL = agentURL.appendingPathComponent("\(resourceName).acd")
        let imageURL = agentURL.appendingPathComponent("\(resourceName)_sprite_map.png")

        guard let fileContent = try? String(contentsOf: fileURL, encoding: .utf8) else { return nil }

        // Character
        guard let characterText = fileContent.fetchInclusive("DefineCharacter", until: "EndCharacter").first else { return nil }
        guard let character = AgentCharacter.parse(content: characterText) else { return nil }

        // Balloon
        guard let balloonText = fileContent.fetchInclusive("DefineBalloon", until: "EndBalloon").first else { return nil }
        guard let balloon = AgentBalloon.parse(content: balloonText) else { return nil }

        // Animations
        let animationTexts = fileContent.fetchInclusive("DefineAnimation", until: "EndAnimation")
        let animations = animationTexts.compactMap { AgentAnimation.parse(content: $0) }

        // States
        let stateTexts = fileContent.fetchInclusive("DefineState", until: "EndState")
        let states = stateTexts.compactMap { AgentState.parse(content: $0) }

        // Sprite Map
        guard let image = NSImage(contentsOf: imageURL)?.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }

        self.character = character
        self.balloon = balloon
        self.animations = animations
        self.states = states
        self.spriteMap = image
    }

    init?(path: String) {
        self.init(agentURL: URL(fileURLWithPath: path))
    }

    func soundURL(forIndex index: Int) -> URL {
        let fileName = "\(resourceName)_\(index).mp3"
        return soundsURL.appendingPathComponent(fileName)
    }

    func findAnimation(_ name: String) -> AgentAnimation? {
        animations.first { $0.name == name }
    }
}

// MARK: - Texture Extraction

extension Agent {
    var columns: Int {
        Int(spriteMap.width) / character.width
    }

    var rows: Int {
        Int(spriteMap.height) / character.height
    }

    func textureAtPosition(x: Int, y: Int) throws -> CGImage {
        guard (0...rows ~= y && 0...columns ~= x) else { throw AgentError.frameOutOfBounds }
        let textureWidth = character.width
        let textureHeight = character.height
        let rect = CGRect(x: x * textureWidth, y: y * textureHeight, width: textureWidth, height: textureHeight)
        return spriteMap.cropping(to: rect)!
    }

    func textureAtIndex(index: Int) throws -> CGImage {
        let x = index % columns
        let y = index / columns
        return try textureAtPosition(x: x, y: y)
    }

    func imageForFrame(_ frame: AgentFrame) -> CGImage {
        let cgImages = frame.images.reversed().map { try! textureAtIndex(index: $0.imageNumber) }
        if let mergedImage = CGImage.mergeImages(cgImages) {
            return mergedImage
        } else {
            return try! textureAtIndex(index: 0)
        }
    }
}

// MARK: - CGImage Merging

extension CGImage {
    static func mergeImages(_ images: [CGImage]) -> CGImage? {
        guard let first = images.first else { return nil }
        if images.count == 1 { return first }

        let width = first.width
        let height = first.height

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        for image in images {
            context.draw(image, in: rect)
        }

        return context.makeImage()
    }
}
