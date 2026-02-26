import Foundation
import AppKit
import SpriteKit

final class ClippyAnimationController {
    private let agent: Agent
    private let sprite: SKSpriteNode
    private var idleTimer: Timer?
    private var currentSound: NSSound?
    private(set) var isAnimating = false

    init(agent: Agent, sprite: SKSpriteNode) {
        self.agent = agent
        self.sprite = sprite
    }

    deinit {
        idleTimer?.invalidate()
    }

    // MARK: - Initial Frame

    func showInitialFrame() {
        guard let cgImage = try? agent.textureAtIndex(index: 0) else {
            print("Warning: Could not load initial frame texture")
            return
        }
        sprite.texture = SKTexture(cgImage: cgImage)
        sprite.size = agent.character.size
    }

    // MARK: - Playback

    func play(animationNamed name: String, completion: (() -> Void)? = nil) {
        guard !isAnimating else { return }
        guard let anim = agent.findAnimation(name) else {
            playRandom(completion: completion)
            return
        }
        playAnimation(anim, completion: completion)
    }

    func playRandom(completion: (() -> Void)? = nil) {
        guard !isAnimating, let anim = agent.animations.randomElement() else {
            completion?()
            return
        }
        playAnimation(anim, completion: completion)
    }

    // MARK: - Click to animate

    func handleClick() {
        guard !isAnimating else { return }
        // Pick a fun reaction animation on click
        let clickAnimations = ["Wave", "Greeting", "GetAttention", "Congratulate",
                               "GetArtsy", "GetTechy", "GetWizardy", "Explain"]
        let available = clickAnimations.compactMap { agent.findAnimation($0) }
        guard let anim = available.randomElement() else {
            playRandom { [weak self] in self?.showInitialFrame(); self?.startIdleTimer() }
            return
        }
        stopIdleTimer()
        playAnimation(anim) { [weak self] in
            self?.showInitialFrame()
            self?.startIdleTimer()
        }
    }

    // MARK: - Idle Timer

    func startIdleTimer() {
        scheduleNextIdleAnimation()
    }

    func stopIdleTimer() {
        idleTimer?.invalidate()
        idleTimer = nil
    }

    private func scheduleNextIdleAnimation() {
        let delay = Double.random(
            in: AppConstants.idleAnimationMinDelay...AppConstants.idleAnimationMaxDelay
        )
        idleTimer?.invalidate()
        // Use .common mode so the timer fires even when menus/popovers are open
        let timer = Timer(timeInterval: delay, repeats: false) { [weak self] _ in
            self?.playIdleAnimation()
        }
        RunLoop.main.add(timer, forMode: .common)
        idleTimer = timer
    }

    private func playIdleAnimation() {
        // Actual idle animation names from clippit.acd
        let idleNames = ["IdleFingerTap", "IdleSideToSide", "IdleEyeBrowRaise",
                         "IdleRopePile", "IdleSnooze", "IdleHeadScratch",
                         "IdleAtom", "Idle1_1",
                         "LookUp", "LookDown", "LookLeft", "LookRight",
                         "LookUpLeft", "LookUpRight", "LookDownLeft", "LookDownRight",
                         "Thinking", "Processing", "CheckingSomething"]

        let available = idleNames.compactMap { agent.findAnimation($0) }
        let anim = available.randomElement() ?? agent.animations.randomElement()

        guard let animation = anim else {
            scheduleNextIdleAnimation()
            return
        }

        playAnimation(animation) { [weak self] in
            self?.showInitialFrame()
            self?.scheduleNextIdleAnimation()
        }
    }

    // MARK: - Animation Engine

    private func playAnimation(_ animation: AgentAnimation, completion: (() -> Void)? = nil) {
        isAnimating = true
        stopIdleTimer()

        DispatchQueue.global(qos: .userInitiated).async { [self] in
            var actions: [SKAction] = []

            for frame in animation.frames {
                // Sound — check UserDefaults each time so setting changes take effect immediately
                let soundEnabled = UserDefaults.standard.object(forKey: "playSounds") as? Bool ?? true
                if soundEnabled, let soundIndex = frame.soundIndex {
                    let soundURL = agent.soundURL(forIndex: soundIndex)
                    actions.append(SKAction.run { [weak self] in
                        self?.currentSound?.stop()
                        let sound = NSSound(contentsOf: soundURL, byReference: true)
                        sound?.play()
                        self?.currentSound = sound
                    })
                }

                // Texture
                let texture = SKTexture(cgImage: agent.imageForFrame(frame))
                texture.filteringMode = .nearest
                let action = SKAction.animate(with: [texture], timePerFrame: frame.durationInSeconds)
                actions.append(action)
            }

            DispatchQueue.main.async { [weak self] in
                self?.sprite.removeAllActions()
                self?.sprite.run(SKAction.sequence(actions)) {
                    self?.isAnimating = false
                    completion?()
                }
            }
        }
    }
}
