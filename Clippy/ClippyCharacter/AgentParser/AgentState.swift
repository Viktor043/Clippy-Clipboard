// AgentParser - Animation state (maps state names to animation names)
// Based on Cosmo/Clippy (https://github.com/Cosmo/Clippy) - adapted for personal use

import Foundation

struct AgentState {
    let name: String
    let animationNames: [String]
}

extension AgentState {
    static func parse(content: String) -> AgentState? {
        var name: String?
        var animationNames: [String] = []

        content.enumerateLines { (line: String, stop: inout Bool) in
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedLine.starts(with: "DefineState ") {
                name = trimmedLine.stringValueOfDefinition(onKey: "DefineState")?.removedQuotes()
            }
            if trimmedLine.starts(with: "Animation ") {
                if let animationName = trimmedLine.stringValueOfKeyValue(onKey: "Animation")?.removedQuotes() {
                    animationNames.append(animationName)
                }
            }
        }

        if let name = name {
            return AgentState(name: name, animationNames: animationNames)
        }
        return nil
    }
}
