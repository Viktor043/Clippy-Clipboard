/**
 * ClippyRenderer — Authentic Microsoft Clippy character renderer
 *
 * Renders the original Clippy (Clippit) character with all 43 animations
 * using HTML5 Canvas and the original sprite sheet.
 *
 * Usage:
 *   const clippy = new ClippyRenderer(document.getElementById('clippy-container'));
 *   await clippy.load('/clippy-animations.json', '/clippit_sprite_map.png');
 *   clippy.startIdleTimer();
 *
 * The renderer creates a <canvas> inside the container element.
 * Click on the canvas to trigger a random reaction animation.
 */

class ClippyRenderer {
  /**
   * @param {HTMLElement} container - DOM element to render Clippy into
   * @param {{ scale?: number, sounds?: boolean, soundsBaseUrl?: string }} options
   */
  constructor(container, options = {}) {
    this.container = container;
    this.spriteSheet = null;
    this.data = null;

    this.frameWidth = 124;
    this.frameHeight = 93;
    this.cols = 33;

    this.isAnimating = false;
    this.idleTimer = null;
    this.animationAbort = null;

    this.soundEnabled = options.sounds ?? false;
    this.soundsBaseUrl = options.soundsBaseUrl ?? '';
    this.audioCache = new Map();

    this.IDLE_MIN_DELAY = 5000;
    this.IDLE_MAX_DELAY = 15000;

    const scale = options.scale ?? 1;

    // Create canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = 124;
    this.canvas.height = 93;
    this.canvas.style.width = `${124 * scale}px`;
    this.canvas.style.height = `${93 * scale}px`;
    this.canvas.style.imageRendering = 'pixelated';
    this.canvas.style.cursor = 'pointer';

    this.ctx = this.canvas.getContext('2d');
    if (!this.ctx) throw new Error('Canvas 2D context not available');
    this.ctx.imageSmoothingEnabled = false;

    this.container.appendChild(this.canvas);

    // Click handler
    this.canvas.addEventListener('click', () => this.handleClick());
  }

  /**
   * Load animation data and sprite sheet. Must be called before any playback.
   * @param {string} animDataUrl - URL to clippy-animations.json
   * @param {string} [spriteSheetUrl] - URL to sprite sheet PNG (optional, uses JSON default)
   */
  async load(animDataUrl, spriteSheetUrl) {
    const resp = await fetch(animDataUrl);
    this.data = await resp.json();

    this.frameWidth = this.data.character.width;
    this.frameHeight = this.data.character.height;
    this.cols = this.data.spriteSheet.columns;
    this.soundsBaseUrl = this.data.sounds.baseUrl;

    const imgUrl = spriteSheetUrl ?? this.data.spriteSheet.imageUrl;
    this.spriteSheet = await this._loadImage(imgUrl);

    this.showRestPose();
  }

  // ---- Public API ----

  /**
   * Play a named animation.
   * @param {string} name - Animation name (e.g. "Wave", "IdleFingerTap")
   * @returns {Promise<void>}
   */
  async play(name) {
    if (!this.data) return;
    const anim = this.data.animations[name];
    if (!anim) {
      console.warn(`ClippyRenderer: animation "${name}" not found`);
      return;
    }
    await this._playAnimation(anim);
  }

  /** Play a random animation from all available. */
  async playRandom() {
    if (!this.data) return;
    const names = Object.keys(this.data.animations);
    const name = names[Math.floor(Math.random() * names.length)];
    await this.play(name);
  }

  /** Handle a click — plays a random reaction animation. */
  handleClick() {
    if (this.isAnimating || !this.data) return;

    const available = this.data.clickAnimations.filter(
      (n) => this.data.animations[n]
    );
    if (available.length === 0) return;

    const name = available[Math.floor(Math.random() * available.length)];

    this.stopIdleTimer();
    this.isAnimating = true;

    this._playAnimation(this.data.animations[name]).then(() => {
      this.showRestPose();
      this.startIdleTimer();
    });
  }

  /** Start the idle animation timer (random animation every 5-15s). */
  startIdleTimer() {
    this._scheduleNextIdle();
  }

  /** Stop the idle animation timer. */
  stopIdleTimer() {
    if (this.idleTimer !== null) {
      clearTimeout(this.idleTimer);
      this.idleTimer = null;
    }
  }

  /** Show the rest pose (sprite index 0). */
  showRestPose() {
    this._drawSpriteIndex(0);
  }

  /** Stop all animations and timers, remove canvas from DOM. */
  destroy() {
    this.stopIdleTimer();
    if (this.animationAbort) this.animationAbort();
    this.canvas.remove();
  }

  /** Get list of all animation names. */
  getAnimationNames() {
    return this.data ? Object.keys(this.data.animations) : [];
  }

  // ---- Private: Rendering ----

  /** @private */
  _drawSpriteIndex(index) {
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

  /** @private */
  _drawFrame(frame) {
    const indices = Array.isArray(frame.imageIndex)
      ? frame.imageIndex
      : [frame.imageIndex];

    if (indices.length === 0) {
      // Empty frame — clear canvas (character invisible, e.g. Hide/GoodBye)
      this.ctx.clearRect(0, 0, this.frameWidth, this.frameHeight);
      return;
    }

    if (indices.length === 1) {
      this._drawSpriteIndex(indices[0]);
      return;
    }

    // Multi-image: composite layers in reverse order (background first)
    this.ctx.clearRect(0, 0, this.frameWidth, this.frameHeight);
    const reversed = [...indices].reverse();
    for (const idx of reversed) {
      const sx = (idx % this.cols) * this.frameWidth;
      const sy = Math.floor(idx / this.cols) * this.frameHeight;
      this.ctx.drawImage(
        this.spriteSheet,
        sx, sy, this.frameWidth, this.frameHeight,
        0, 0, this.frameWidth, this.frameHeight
      );
    }
  }

  // ---- Private: Animation Engine ----

  /** @private */
  _playAnimation(anim) {
    return new Promise((resolve) => {
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
          this._playSound(frame.sound);
        }

        // Draw this frame
        this._drawFrame(frame);

        // Determine next frame index
        let nextIndex = frameIndex + 1;

        // Check probabilistic branching
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

        // Schedule next frame
        setTimeout(playNextFrame, frame.duration);
      };

      playNextFrame();
    });
  }

  // ---- Private: Idle System ----

  /** @private */
  _scheduleNextIdle() {
    this.stopIdleTimer();
    const delay =
      this.IDLE_MIN_DELAY +
      Math.random() * (this.IDLE_MAX_DELAY - this.IDLE_MIN_DELAY);

    this.idleTimer = setTimeout(() => {
      this._playIdleAnimation();
    }, delay);
  }

  /** @private */
  _playIdleAnimation() {
    if (this.isAnimating || !this.data) {
      this._scheduleNextIdle();
      return;
    }

    const available = this.data.idleAnimations.filter(
      (n) => this.data.animations[n]
    );
    if (available.length === 0) {
      this._scheduleNextIdle();
      return;
    }

    const name = available[Math.floor(Math.random() * available.length)];

    this._playAnimation(this.data.animations[name]).then(() => {
      this.showRestPose();
      this._scheduleNextIdle();
    });
  }

  // ---- Private: Sound ----

  /** @private */
  _playSound(index) {
    if (!this.data) return;
    const url = this.soundsBaseUrl + this.data.sounds.files[index];

    let audio = this.audioCache.get(index);
    if (!audio) {
      audio = new Audio(url);
      this.audioCache.set(index, audio);
    }

    audio.currentTime = 0;
    audio.play().catch(() => {
      // Autoplay may be blocked — silently ignore
    });
  }

  // ---- Private: Helpers ----

  /** @private */
  _loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
      img.src = url;
    });
  }
}

// Export for module systems, also available as global
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ClippyRenderer };
}
