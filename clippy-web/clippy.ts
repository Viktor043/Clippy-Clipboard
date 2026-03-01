/**
 * ClippyRenderer — Authentic Microsoft Clippy character renderer
 *
 * Renders the original Clippy (Clippit) character with all 43 animations
 * using HTML5 Canvas and the original sprite sheet.
 *
 * Usage:
 *   const clippy = new ClippyRenderer(document.getElementById('clippy-container')!);
 *   await clippy.load('/clippy-animations.json', '/clippit_sprite_map.png');
 *   clippy.startIdleTimer();
 *
 * The renderer creates a <canvas> inside the container element.
 * Click on the canvas to trigger a random reaction animation.
 */

// ---- Types ----

interface Branch {
  branchTo: number;
  probability: number;
}

interface FrameData {
  imageIndex: number | number[]; // sprite index(es), empty array = blank frame
  duration: number;              // milliseconds
  exitBranch: number | null;     // absolute frame index for early exit
  branches: Branch[];            // probabilistic transitions
  sound: number | null;          // sound file index (0-14) or null
}

interface AnimationData {
  frames: FrameData[];
}

interface ClippyData {
  character: { width: number; height: number };
  spriteSheet: { columns: number; rows: number; imageUrl: string };
  sounds: { baseUrl: string; files: string[] };
  animations: Record<string, AnimationData>;
  states: Record<string, string[]>;
  idleAnimations: string[];
  clickAnimations: string[];
}

interface ClippyOptions {
  scale?: number;          // CSS scale factor (default: 1)
  sounds?: boolean;        // enable sound playback (default: false)
  soundsBaseUrl?: string;  // override sound files base URL
}

// ---- Renderer ----

export class ClippyRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private spriteSheet: HTMLImageElement | null = null;
  private data: ClippyData | null = null;

  private frameWidth = 124;
  private frameHeight = 93;
  private cols = 33;

  private isAnimating = false;
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private animationAbort: (() => void) | null = null;

  private soundEnabled: boolean;
  private soundsBaseUrl = '';
  private audioCache: Map<number, HTMLAudioElement> = new Map();

  private readonly IDLE_MIN_DELAY = 5000;
  private readonly IDLE_MAX_DELAY = 15000;

  constructor(
    private container: HTMLElement,
    options: ClippyOptions = {}
  ) {
    const scale = options.scale ?? 1;
    this.soundEnabled = options.sounds ?? false;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = 124;
    this.canvas.height = 93;
    this.canvas.style.width = `${124 * scale}px`;
    this.canvas.style.height = `${93 * scale}px`;
    this.canvas.style.imageRendering = 'pixelated';
    this.canvas.style.cursor = 'pointer';

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context not available');
    this.ctx = ctx;
    this.ctx.imageSmoothingEnabled = false;

    this.container.appendChild(this.canvas);

    // Click handler
    this.canvas.addEventListener('click', () => this.handleClick());
  }

  /**
   * Load animation data and sprite sheet. Must be called before any playback.
   */
  async load(animDataUrl: string, spriteSheetUrl?: string): Promise<void> {
    // Load animation JSON
    const resp = await fetch(animDataUrl);
    this.data = await resp.json() as ClippyData;

    this.frameWidth = this.data.character.width;
    this.frameHeight = this.data.character.height;
    this.cols = this.data.spriteSheet.columns;
    this.soundsBaseUrl = this.data.sounds.baseUrl;

    // Load sprite sheet
    const imgUrl = spriteSheetUrl ?? this.data.spriteSheet.imageUrl;
    this.spriteSheet = await this.loadImage(imgUrl);

    // Show rest pose
    this.showRestPose();
  }

  // ---- Public API ----

  /** Play a named animation. Returns a promise that resolves when done. */
  async play(name: string): Promise<void> {
    if (!this.data) return;
    const anim = this.data.animations[name];
    if (!anim) {
      console.warn(`ClippyRenderer: animation "${name}" not found`);
      return;
    }
    await this.playAnimation(anim);
  }

  /** Play a random animation from all available. */
  async playRandom(): Promise<void> {
    if (!this.data) return;
    const names = Object.keys(this.data.animations);
    const name = names[Math.floor(Math.random() * names.length)];
    await this.play(name);
  }

  /** Handle a click — plays a random reaction animation. */
  handleClick(): void {
    if (this.isAnimating || !this.data) return;

    const available = this.data.clickAnimations.filter(
      n => this.data!.animations[n]
    );
    if (available.length === 0) return;

    const name = available[Math.floor(Math.random() * available.length)];

    this.stopIdleTimer();
    this.isAnimating = true;

    this.playAnimation(this.data.animations[name]).then(() => {
      this.showRestPose();
      this.startIdleTimer();
    });
  }

  /** Start the idle animation timer. */
  startIdleTimer(): void {
    this.scheduleNextIdle();
  }

  /** Stop the idle animation timer. */
  stopIdleTimer(): void {
    if (this.idleTimer !== null) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }

  /** Show the rest pose (sprite index 0). */
  showRestPose(): void {
    this.drawSpriteIndex(0);
  }

  /** Stop all animations and timers, remove canvas. */
  destroy(): void {
    this.stopIdleTimer();
    if (this.animationAbort) this.animationAbort();
    this.canvas.remove();
  }

  /** Get list of all animation names. */
  getAnimationNames(): string[] {
    return this.data ? Object.keys(this.data.animations) : [];
  }

  // ---- Private: Rendering ----

  private drawSpriteIndex(index: number): void {
    if (!this.spriteSheet) return;

    const sx = (index % this.cols) * this.frameWidth;
    const sy = Math.floor(index / this.cols) * this.frameHeight;

    this.ctx.clearRect(0, 0, this.frameWidth, this.frameHeight);
    this.ctx.drawImage(
      this.spriteSheet,
      sx, sy, this.frameWidth, this.frameHeight,
      0, 0, this.frameWidth, this.frameHeight
    );
  }

  private drawFrame(frame: FrameData): void {
    const indices = Array.isArray(frame.imageIndex)
      ? frame.imageIndex
      : [frame.imageIndex];

    if (indices.length === 0) {
      // Empty frame — clear canvas (character invisible)
      this.ctx.clearRect(0, 0, this.frameWidth, this.frameHeight);
      return;
    }

    if (indices.length === 1) {
      this.drawSpriteIndex(indices[0]);
      return;
    }

    // Multi-image: composite in reverse order (background first)
    this.ctx.clearRect(0, 0, this.frameWidth, this.frameHeight);
    const reversed = [...indices].reverse();
    for (const idx of reversed) {
      const sx = (idx % this.cols) * this.frameWidth;
      const sy = Math.floor(idx / this.cols) * this.frameHeight;
      this.ctx.drawImage(
        this.spriteSheet!,
        sx, sy, this.frameWidth, this.frameHeight,
        0, 0, this.frameWidth, this.frameHeight
      );
    }
  }

  // ---- Private: Animation Engine ----

  private playAnimation(anim: AnimationData): Promise<void> {
    return new Promise<void>((resolve) => {
      if (this.isAnimating && !this.animationAbort) {
        resolve();
        return;
      }

      this.isAnimating = true;
      let cancelled = false;

      this.animationAbort = () => {
        cancelled = true;
        this.isAnimating = false;
        resolve();
      };

      const frames = anim.frames;
      let frameIndex = 0;

      const playNextFrame = () => {
        if (cancelled || frameIndex >= frames.length) {
          this.isAnimating = false;
          this.animationAbort = null;
          resolve();
          return;
        }

        const frame = frames[frameIndex];

        // Play sound if enabled
        if (this.soundEnabled && frame.sound !== null) {
          this.playSound(frame.sound);
        }

        // Draw this frame
        this.drawFrame(frame);

        // Determine next frame
        let nextIndex = frameIndex + 1;

        // Check branching
        if (frame.branches.length > 0) {
          const roll = Math.random() * 100;
          let cumulative = 0;
          for (const branch of frame.branches) {
            cumulative += branch.probability;
            if (roll < cumulative) {
              nextIndex = branch.branchTo;
              break;
            }
          }
        }

        frameIndex = nextIndex;

        // Wait for frame duration then continue
        setTimeout(playNextFrame, frame.duration);
      };

      playNextFrame();
    });
  }

  // ---- Private: Idle System ----

  private scheduleNextIdle(): void {
    this.stopIdleTimer();
    const delay = this.IDLE_MIN_DELAY +
      Math.random() * (this.IDLE_MAX_DELAY - this.IDLE_MIN_DELAY);

    this.idleTimer = setTimeout(() => {
      this.playIdleAnimation();
    }, delay);
  }

  private playIdleAnimation(): void {
    if (this.isAnimating || !this.data) {
      this.scheduleNextIdle();
      return;
    }

    const available = this.data.idleAnimations.filter(
      n => this.data!.animations[n]
    );
    if (available.length === 0) {
      this.scheduleNextIdle();
      return;
    }

    const name = available[Math.floor(Math.random() * available.length)];

    this.playAnimation(this.data.animations[name]).then(() => {
      this.showRestPose();
      this.scheduleNextIdle();
    });
  }

  // ---- Private: Sound ----

  private playSound(index: number): void {
    if (!this.data) return;
    const url = this.soundsBaseUrl + this.data.sounds.files[index];

    let audio = this.audioCache.get(index);
    if (!audio) {
      audio = new Audio(url);
      this.audioCache.set(index, audio);
    }

    audio.currentTime = 0;
    audio.play().catch(() => {
      // Autoplay blocked — ignore
    });
  }

  // ---- Private: Helpers ----

  private loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });
  }
}

// ---- Default export for convenience ----
export default ClippyRenderer;
