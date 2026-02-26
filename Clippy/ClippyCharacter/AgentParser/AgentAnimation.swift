// AgentParser - Named animation with frame sequence
// Based on Cosmo/Clippy (https://github.com/Cosmo/Clippy) - adapted for personal use

import Foundation

struct AgentAnimation {
    let name: String
    let transitionType: Int
    let frames: [AgentFrame]
}

extension AgentAnimation {
    static func parse(content: String) -> AgentAnimation? {
        var animationName: String?
        var transitionType: Int?
        let frameTexts = content.fetchInclusive("DefineFrame", until: "EndFrame")
        let frames = frameTexts.compactMap { AgentFrame.parse(content: $0) }

        content.enumerateLines { (line: String, stop: inout Bool) in
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedLine.starts(with: "DefineAnimation ") {
                animationName = trimmedLine.stringValueOfDefinition(onKey: "DefineAnimation")?.removedQuotes()
            }
            if trimmedLine.starts(with: "TransitionType ") {
                transitionType = trimmedLine.intValueOfKeyValue(onKey: "TransitionType")
            }
        }

        if let animationName = animationName, let transitionType = transitionType {
            return AgentAnimation(name: animationName, transitionType: transitionType, frames: frames)
        }
        return nil
    }
}
