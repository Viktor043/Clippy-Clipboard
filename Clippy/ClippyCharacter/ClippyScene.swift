import SpriteKit

final class ClippyScene: SKScene {
    let agentSprite = SKSpriteNode()

    override func didMove(to view: SKView) {
        backgroundColor = .clear
        scaleMode = .resizeFill

        agentSprite.anchorPoint = CGPoint(x: 0.5, y: 0.5)
        agentSprite.position = CGPoint(x: size.width / 2, y: size.height / 2)
        addChild(agentSprite)
    }

    func updateSpriteSize(_ size: CGSize) {
        agentSprite.size = size
    }
}
