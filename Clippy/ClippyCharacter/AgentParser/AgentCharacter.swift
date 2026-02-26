// AgentParser - Character metadata (dimensions, style, infos)
// Based on Cosmo/Clippy (https://github.com/Cosmo/Clippy) - adapted for personal use

import Foundation

struct AgentCharacter {
    let infos: [AgentInfo]
    let guid: String
    let width: Int
    let height: Int
    let transparency: Int
    let defaultFrameDuration: Int
    let style: String
    let colorTable: String
}

extension AgentCharacter {
    static func parse(content: String) -> AgentCharacter? {
        let infoTexts = content.fetchInclusive("DefineInfo", until: "EndInfo")
        let infos = infoTexts.compactMap { AgentInfo.parse(content: $0) }
        var guid: String?
        var width: Int?
        var height: Int?
        var transparency: Int?
        var defaultFrameDuration: Int?
        var style: String?
        var colorTable: String?

        content.enumerateLines { (line: String, stop: inout Bool) in
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedLine.starts(with: "GUID ") {
                guid = trimmedLine.stringValueOfKeyValue(onKey: "GUID")?.removedCurlyBraces()
            }
            if trimmedLine.starts(with: "Width ") {
                width = trimmedLine.intValueOfKeyValue(onKey: "Width")
            }
            if trimmedLine.starts(with: "Height ") {
                height = trimmedLine.intValueOfKeyValue(onKey: "Height")
            }
            if trimmedLine.starts(with: "Transparency ") {
                transparency = trimmedLine.intValueOfKeyValue(onKey: "Transparency")
            }
            if trimmedLine.starts(with: "DefaultFrameDuration ") {
                defaultFrameDuration = trimmedLine.intValueOfKeyValue(onKey: "DefaultFrameDuration")
            }
            if trimmedLine.starts(with: "Style ") {
                style = trimmedLine.stringValueOfKeyValue(onKey: "Style")
            }
            if trimmedLine.starts(with: "ColorTable ") {
                colorTable = trimmedLine.stringValueOfKeyValue(onKey: "ColorTable")?.removedQuotes()
            }
        }

        if let guid = guid, let width = width, let height = height,
           let transparency = transparency, let defaultFrameDuration = defaultFrameDuration,
           let style = style, let colorTable = colorTable {
            return AgentCharacter(infos: infos, guid: guid, width: width, height: height,
                                transparency: transparency, defaultFrameDuration: defaultFrameDuration,
                                style: style, colorTable: colorTable)
        }
        return nil
    }
}

extension AgentCharacter {
    var size: CGSize {
        CGSize(width: width, height: height)
    }
}
