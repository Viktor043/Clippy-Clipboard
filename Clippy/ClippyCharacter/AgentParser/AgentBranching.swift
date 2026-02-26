// AgentParser - Branching logic for animation transitions
// Based on Cosmo/Clippy (https://github.com/Cosmo/Clippy) - adapted for personal use

import Foundation

struct AgentBranching {
    let branchTo: Int
    let probability: Int
}

extension AgentBranching {
    static func parse(content: String) -> [AgentBranching] {
        var branchings: [AgentBranching] = []
        var branchToBuffer: Int?
        var probabilityBuffer: Int?

        content.enumerateLines { (line: String, stop: inout Bool) in
            let trimmedLine = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedLine.starts(with: "BranchTo ") {
                branchToBuffer = trimmedLine.intValueOfKeyValue(onKey: "BranchTo")
            }
            if trimmedLine.starts(with: "Probability ") {
                probabilityBuffer = trimmedLine.intValueOfKeyValue(onKey: "Probability")
            }
            if let branchTo = branchToBuffer, let probability = probabilityBuffer {
                branchings.append(AgentBranching(branchTo: branchTo, probability: probability))
                branchToBuffer = nil
                probabilityBuffer = nil
            }
        }

        return branchings
    }
}
