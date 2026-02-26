# Clippy Clipboard Manager — Web Module Specification

> **Target platform:** Lovable web app module (React + TypeScript + Tailwind CSS)
> **Module type:** Self-contained tool module that plugs into a multi-tool website
> **Clipboard access:** Browser Clipboard API (`navigator.clipboard`, `document.addEventListener('copy'/'cut')`)

---

## 1. Module Overview

Clippy is a clipboard history manager with an animated assistant character. It captures clipboard events within the browser, stores history locally, provides search/preview, and features an animated Clippy character that reacts to copy events.

### What transfers from native to web
| Native feature | Web equivalent | Notes |
|---|---|---|
| System-wide clipboard monitoring | `copy`/`cut` event listeners on `document` | Only captures copies made within the browser tab |
| Core Data persistence | IndexedDB (via Dexie.js or idb) | Persistent across sessions |
| NSPopover / NSPanel | Floating div panels with Tailwind | Positioned via CSS absolute/fixed |
| NSVisualEffectView (frosted glass) | `backdrop-blur` + semi-transparent bg | `bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl` |
| SpriteKit animation | CSS sprite sheet animation or `<canvas>` | Sprite map + `requestAnimationFrame` |
| NSSound playback | Web Audio API or `<audio>` elements | MP3 files from agent bundle |
| UserDefaults | `localStorage` | JSON serialized settings |
| Multi-select text (Cmd+click) | Custom selection logic with `Range` API | Or simplified to standard browser selection |

### What does NOT transfer
- System-wide clipboard capture (web is browser-only)
- File clipboard items (no file system access)
- Paste injection to other apps (web can only write to clipboard)
- App exclusion lists (no concept in browser)
- Dock/menu bar integration

---

## 2. Data Model

### ClipboardItem

```typescript
interface ClipboardItem {
  id: string;              // UUID v4
  type: 'text' | 'richText' | 'image';
  timestamp: number;       // Unix ms
  textContent: string | null;
  imageDataUrl: string | null;    // base64 data URL for images
  thumbnailDataUrl: string | null; // smaller version for list
  sourceUrl: string | null;       // page URL where copy happened
}
```

### Settings

```typescript
interface ClippySettings {
  maxHistoryItems: number;       // default: 100, options: [25, 50, 100, 200, 500]
  maxCharsPerItem: number;       // default: 20000, options: [5000, 10000, 20000, 50000]
  showClippyCharacter: boolean;  // default: true
  playSounds: boolean;           // default: true
  captureText: boolean;          // default: true
  captureRichText: boolean;      // default: true
  captureImages: boolean;        // default: true
  clippyScale: number;           // default: 1.0, options: [0.5, 0.75, 1.0, 1.25, 1.5]
  fontSize: number;              // default: 12, options: [10, 11, 12, 13, 14, 16]
  fontFamily: string;            // default: 'system-ui'
}
```

### Storage

```typescript
// IndexedDB for clipboard history (survives page reloads, large storage)
// localStorage for settings (small, synchronous reads)

// IndexedDB schema:
// Database: "clippy-db"
// Object store: "items"
//   keyPath: "id"
//   indexes: ["timestamp"] (for sorting)

// localStorage keys:
// "clippy-settings" -> JSON.stringify(ClippySettings)
// "clippy-position" -> JSON.stringify({ x, y, locked })
```

---

## 3. Module Architecture

```
src/tools/clippy/
├── index.tsx                    # Module entry point & exports
├── ClippyModule.tsx             # Main component that orchestrates everything
├── components/
│   ├── ClipboardHistoryPanel.tsx # Main popup (search + list + settings)
│   ├── ClipboardItemRow.tsx      # Single row in the list
│   ├── DetailPanel.tsx           # Expanded preview panel
│   ├── SettingsPanel.tsx         # Inline settings view
│   ├── ClippyCharacter.tsx       # Animated character (canvas or sprite)
│   ├── ClippyBalloon.tsx         # Speech bubble overlay
│   └── SearchBar.tsx             # Search input component
├── hooks/
│   ├── useClipboardMonitor.ts    # Clipboard event listener hook
│   ├── useClipboardStore.ts      # IndexedDB CRUD operations
│   ├── useClippySettings.ts      # Settings with localStorage
│   ├── useClippyAnimations.ts    # Sprite animation controller
│   └── usePanelPosition.ts       # Panel positioning & screen clamping
├── utils/
│   ├── agentParser.ts            # Parse .agent character files
│   ├── spriteExtractor.ts        # Extract frames from sprite map
│   └── constants.ts              # All magic numbers and defaults
├── types.ts                      # TypeScript interfaces
└── assets/
    └── clippit/                  # Clippy character files
        ├── sprite_map.png
        └── sounds/               # MP3 sound effects
```

---

## 4. Constants

```typescript
// constants.ts
export const PANEL_WIDTH = 320;
export const PANEL_HEIGHT = 480;
export const PANEL_CORNER_RADIUS = 10;     // rounded-[10px] or rounded-xl
export const ROW_CORNER_RADIUS = 4;        // rounded
export const SECTION_CORNER_RADIUS = 8;    // rounded-lg

export const CLIPPY_BASE_SIZE = { width: 130, height: 160 };
export const CLIPPY_EDGE_INSET = 20;

export const BALLOON_DISMISS_DELAY = 4000; // ms
export const IDLE_ANIMATION_MIN = 5000;    // ms
export const IDLE_ANIMATION_MAX = 15000;   // ms

export const SCALE_OPTIONS = [
  { label: 'XS', value: 0.5 },
  { label: 'S',  value: 0.75 },
  { label: 'M',  value: 1.0 },
  { label: 'L',  value: 1.25 },
  { label: 'XL', value: 1.5 },
];

export const HISTORY_LIMIT_OPTIONS = [25, 50, 100, 200, 500];
export const CHAR_LIMIT_OPTIONS = [5000, 10000, 20000, 50000];
export const FONT_SIZE_OPTIONS = [10, 11, 12, 13, 14, 16];

// Copy reaction messages
export const COPY_REACTIONS = {
  text:     { animation: 'Greeting',  message: 'Copied text!' },
  richText: { animation: 'Greeting',  message: 'Got that rich text!' },
  image:    { animation: 'Searching', message: 'Nice screenshot!' },
};
```

---

## 5. Component Specifications

### 5.1 ClippyModule (Root)

The top-level component that the Lovable app mounts as a tool module.

```tsx
// Manages global state and renders all Clippy sub-components
// This is the module's public interface

export function ClippyModule() {
  // Hooks
  const store = useClipboardStore();
  const settings = useClippySettings();
  const monitor = useClipboardMonitor(store, settings);

  // Panel visibility state
  const [showHistory, setShowHistory] = useState(false);
  const [showDetail, setShowDetail] = useState<ClipboardItem | null>(null);

  return (
    <>
      {/* Floating Clippy character */}
      {settings.showClippyCharacter && (
        <ClippyCharacter
          onCopyEvent={monitor.lastEvent}
          onClick={() => setShowHistory(prev => !prev)}
          onRightClick={/* context menu */}
          scale={settings.clippyScale}
          playSounds={settings.playSounds}
        />
      )}

      {/* Main clipboard history panel */}
      {showHistory && (
        <ClipboardHistoryPanel
          store={store}
          settings={settings}
          onPaste={(item) => { pasteItem(item); setShowHistory(false); }}
          onExpand={(item) => setShowDetail(item)}
          onClose={() => setShowHistory(false)}
        />
      )}

      {/* Detail preview panel */}
      {showDetail && (
        <DetailPanel
          item={showDetail}
          settings={settings}
          onClose={() => setShowDetail(null)}
          anchorRef={/* history panel ref for positioning */}
        />
      )}
    </>
  );
}
```

### 5.2 ClipboardHistoryPanel

**Dimensions:** 320px wide, 480px tall
**Position:** Fixed, anchored to the Clippy character (above it) or to a trigger button
**Background:** Frosted glass effect

```
┌─────────────────────────────────────┐
│  < Settings          🗑️  ⚙️        │  ← Header (swap between "Clippy" / "Settings")
├─────────────────────────────────────┤
│  🔍 Search clipboard...        ✕   │  ← Search bar (only in list mode)
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐    │
│  │ Copied text preview...      │    │  ← ClipboardItemRow
│  │ 2 min ago        Expand  ✕  │    │     hover reveals Expand + Delete
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ 📷 Image — 1920x1080       │    │  ← Image item with thumbnail
│  │ 5 min ago        Expand  ✕  │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Another clipboard item      │    │
│  │ 1 hr ago         Expand  ✕  │    │
│  └─────────────────────────────┘    │
│  ...scrollable...                   │
└─────────────────────────────────────┘
```

**Styling:**

```css
/* Panel container */
.panel {
  width: 320px;
  height: 480px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.8);    /* light mode */
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12), 0 2px 8px rgba(0, 0, 0, 0.08);
}

/* Dark mode */
.dark .panel {
  background: rgba(30, 30, 30, 0.85);
}
```

**Tailwind equivalent:**
```
bg-white/80 dark:bg-gray-900/85 backdrop-blur-xl rounded-[10px]
shadow-[0_8px_32px_rgba(0,0,0,0.12),0_2px_8px_rgba(0,0,0,0.08)]
```

**Header bar:**
- Height: ~40px
- Padding: 12px horizontal, 8px vertical
- Divider below: 1px border-b

**States:**
1. **List mode** (default): Header shows "Clippy" title + trash icon + gear icon
2. **Settings mode**: Header shows back arrow + "Settings" title. Content transitions with slide animation.

**Transitions:**
- List <-> Settings: Slide left/right with 200ms ease-in-out + opacity fade

**Click-outside behavior:**
- Clicking outside the panel AND the detail panel closes the history panel
- Clicking inside the detail panel keeps the history panel open
- Use a `useClickOutside` hook with refs for both panels

### 5.3 ClipboardItemRow

**Layout:** Horizontal flex, 12px horizontal padding, 6px vertical padding

```
┌──────────────────────────────────────────────┐
│ [thumb] Content preview line...   [Expand] ✕ │
│         2 min ago                            │
└──────────────────────────────────────────────┘
```

**Elements:**
1. **Thumbnail** (optional, 28x28px, rounded-[4px]) — only for image items
2. **Content area** (flex-1):
   - Line 1: Text preview, 1 line max, 12px semibold, truncated with ellipsis
   - Line 2: Relative timestamp, 10px, secondary color
3. **Expand button** (hidden until hover): Pill shape with left-chevron icon + "Expand" text
   - Font: 9px medium
   - Icon: 8px semibold
   - Colors: accent foreground, accent/10 background
   - Padding: 5px horizontal, 2px vertical
   - Corner radius: 4px
4. **Delete button** (hidden until hover): "×" icon, 10px, secondary color

**States:**
- **Default:** Transparent background, Expand/Delete hidden (opacity: 0)
- **Hover:** `bg-gray-900/[0.06] dark:bg-white/[0.06]`, Expand/Delete shown (opacity: 1)
- **Active/Press:** `bg-blue-500/[0.18]` on the content area
- Transitions: opacity 150ms

**Full-width hover:** The entire row is hoverable, not just the text. Use `cursor-pointer` and a full-width click target.

**Click action:** Clicking the content area copies the item to clipboard and triggers `onPaste`.

### 5.4 DetailPanel

**Dimensions:** 320px wide, 480px tall (same as history panel)
**Position:** To the LEFT of the history panel, 2px gap, vertically centered
**Background:** Same frosted glass as history panel

```
┌─────────────────────────────────────┐
│  Preview           Copy All    >    │  ← Header
├─────────────────────────────────────┤
│                                     │
│  [Full image preview if image]      │
│                                     │
│  Full text content here with        │  ← Scrollable text area
│  multi-line display and word        │     Selectable text
│  wrapping for long clipboard        │     (Cmd+C supported)
│  entries...                         │
│                                     │
├─────────────────────────────────────┤
│  Type: text                         │  ← Metadata footer
│  Copied: Feb 26, 2026 3:45 PM      │
│  ⌘+click to add selections         │
└─────────────────────────────────────┘
```

**Header elements:**
- "Preview" title (headline font)
- "Copy All" button: Same pill style as Expand button (chevron-left icon + text)
  - On click: Copies full content, shows green "Copied" state for 600ms, then closes
- Close button: chevron-right icon, 12px semibold, accent color

**Content area:**
- Image type: Thumbnail display (max 160px tall, rounded-[4px]), then text below
- Text type: Full scrollable text, selectable, using configured font
- Padding: 4px horizontal for text, 12px for images

**Metadata footer:**
- Padding: 12px horizontal, 8px vertical
- "Type: {type}" — 10px, secondary color
- "Copied: {formatted date}" — 10px, secondary color
- Selection help text — 9px, secondary/60 opacity

**Click behavior:**
- Clicking the close button (">") only closes the detail panel, NOT the history panel
- The detail panel is independent — closing the history panel also closes the detail panel

### 5.5 SettingsPanel (Inline)

Replaces the list content when gear icon is clicked. Slides in from the right.

**Sections** (each in a rounded-lg card with bg-gray-900/[0.03] background, 10px padding):

#### General
- Toggle: "Show menu bar icon" (not applicable for web — could control a toolbar button instead)

#### Clips Management
- Select: "Remember up to" → [25, 50, 100, 200, 500] items
- Select: "Max chars per item" → [5K, 10K, 20K, 50K]

#### Appearance
- Select: "Font" → [System, monospace, serif, sans-serif] (web-safe fonts)
- Select: "Font size" → [10, 11, 12, 13, 14, 16]

#### Clippy Animation
- Toggle: "Show Clippy character"
- Toggle: "Play sounds"
- Segmented picker: "Size" → [XS, S, M, L, XL]

#### Advanced
- Toggle: "Capture plain text"
- Toggle: "Capture rich text"
- Toggle: "Capture images"

#### Bottom
- "Clear All History" button (destructive, with confirmation dialog)

**Visual styling per section:**
- Label: 12px semibold, secondary color, with icon
- Controls: 12px regular
- Section spacing: 16px
- Inner spacing: 8px
- Background: primary color at 3% opacity
- Corner radius: 8px

### 5.6 ClippyCharacter

**Rendering approach:** `<canvas>` element or CSS background-position sprite animation

**Size:** 130x160px base, scaled by `clippyScale` setting
**Position:** Fixed, draggable, defaults to bottom-right corner (20px inset from edges)
**Z-index:** High (above page content, below modals)

**Sprite system:**
```typescript
interface SpriteAnimation {
  name: string;
  frames: SpriteFrame[];
}

interface SpriteFrame {
  imageIndices: number[];     // indices into sprite map grid
  duration: number;           // milliseconds
  soundIndex: number | null;  // index for sound playback
}

// Extract frame from sprite map:
// columns = spriteMapWidth / frameWidth
// x = (imageIndex % columns) * frameWidth
// y = Math.floor(imageIndex / columns) * frameHeight
```

**Animations to implement:**

| Category | Names | Trigger |
|---|---|---|
| Click reactions | Wave, Greeting, GetAttention, Congratulate, GetArtsy, GetTechy, GetWizardy, Explain | Click on character (random pick) |
| Idle | IdleFingerTap, IdleSideToSide, IdleEyeBrowRaise, IdleRopePile, IdleSnooze, IdleHeadScratch, Thinking, Processing, LookUp, LookDown, LookLeft, LookRight | Auto-play every 5-15 seconds |
| Copy reaction | Greeting (text), Searching (image) | Clipboard capture event |

**Interactions:**
1. **Click:** Play random click animation + toggle clipboard panel
2. **Right-click:** Show context menu:
   - "Play Animation"
   - Separator
   - "Size" submenu → XS / S / M / L / XL (checkmark on current)
   - "Lock Position" / "Unlock Position"
   - "Mute Sounds" / "Unmute Sounds"
   - Separator
   - "Hide Clippy"
3. **Drag:** Move character position (unless locked). Save to localStorage.
4. **Idle:** Random animations at 5-15 second intervals

**Position persistence:**
```typescript
interface ClippyPosition {
  x: number;
  y: number;
  locked: boolean;
}
// Save to localStorage("clippy-position")
// Clamp to viewport on window resize
```

### 5.7 ClippyBalloon

**Position:** Above the Clippy character, centered horizontally
**Duration:** 4 seconds, then 300ms fade-out

**Styling:**
```
bg-white dark:bg-gray-800
border border-gray-200 dark:border-gray-700
rounded-lg (8px)
shadow-[0_2px_8px_rgba(0,0,0,0.15)]
px-3 py-2
text-[11px]
max-w-[180px]
text-center
line-clamp-3
```

### 5.8 SearchBar

**Layout:** Horizontal flex inside the panel, between header and list

```
🔍 [Search clipboard...              ] [✕]
```

- Magnifying glass icon: secondary color
- Input: plain style (no border), placeholder "Search clipboard..."
- Clear button: circle-x icon, only visible when text is entered
- Padding: 12px horizontal, 6px vertical
- Divider below

**Filtering:** Case-insensitive `includes()` on `textContent`. Filter runs on every keystroke.

---

## 6. Hooks Specification

### 6.1 useClipboardMonitor

```typescript
function useClipboardMonitor(store: ClipboardStore, settings: ClippySettings) {
  // Listen for 'copy' and 'cut' events on document
  // Extract clipboard content via navigator.clipboard.readText() / read()
  // Check settings for which types to capture
  // Add to store with deduplication
  // Return last copy event for animation triggers

  useEffect(() => {
    const handler = async (e: ClipboardEvent) => {
      // 1. Check if captures are enabled for this type
      // 2. Read from event.clipboardData (synchronous, more reliable)
      //    - text/plain → text item
      //    - text/html → richText item
      //    - image/* → image item (read as blob, convert to dataURL)
      // 3. Deduplicate (if same text as most recent item, just bump timestamp)
      // 4. Truncate text to maxCharsPerItem
      // 5. Generate thumbnail for images (canvas resize to 64px max dimension)
      // 6. Add to IndexedDB store
      // 7. Prune old items beyond maxHistoryItems
      // 8. Set lastEvent to trigger animation
    };

    document.addEventListener('copy', handler);
    document.addEventListener('cut', handler);
    return () => {
      document.removeEventListener('copy', handler);
      document.removeEventListener('cut', handler);
    };
  }, [settings]);

  return { lastEvent };
}
```

### 6.2 useClipboardStore

```typescript
function useClipboardStore() {
  // IndexedDB operations via Dexie.js or idb
  // Reactive: components re-render when items change

  return {
    items: ClipboardItem[];           // sorted by timestamp desc
    addItem: (item: ClipboardItem) => Promise<void>;
    deleteItem: (id: string) => Promise<void>;
    clearAll: () => Promise<void>;
    // Automatically prunes beyond maxHistoryItems on add
  };
}
```

### 6.3 useClippySettings

```typescript
function useClippySettings() {
  // Read/write localStorage("clippy-settings")
  // Returns reactive settings object + update function

  return {
    settings: ClippySettings;
    updateSetting: <K extends keyof ClippySettings>(key: K, value: ClippySettings[K]) => void;
  };
}
```

### 6.4 useClippyAnimations

```typescript
function useClippyAnimations(canvasRef: RefObject<HTMLCanvasElement>, spriteMap: HTMLImageElement) {
  // Manages animation playback on canvas
  // Handles idle timer (5-15s random intervals)
  // Plays sounds via Web Audio API

  return {
    playAnimation: (name: string, onComplete?: () => void) => void;
    playRandom: (category: 'click' | 'idle') => void;
    showInitialFrame: () => void;
    isPlaying: boolean;
  };
}
```

### 6.5 usePanelPosition

```typescript
function usePanelPosition(anchorRef: RefObject<HTMLElement>, panelSize: { width: number; height: number }) {
  // Calculate panel position relative to anchor element
  // Clamp to viewport bounds
  // Handle window resize

  return {
    style: CSSProperties;  // { position, top, left }
  };
}
```

---

## 7. Clipboard Capture Implementation

### Reading clipboard from copy/cut events

```typescript
async function extractClipboardData(event: ClipboardEvent): Promise<ClipboardItem | null> {
  const clipboardData = event.clipboardData;
  if (!clipboardData) return null;

  const id = crypto.randomUUID();
  const timestamp = Date.now();

  // Check for images first (higher priority)
  const imageFile = Array.from(clipboardData.files).find(f => f.type.startsWith('image/'));
  if (imageFile) {
    const dataUrl = await fileToDataUrl(imageFile);
    const thumbnail = await generateThumbnail(dataUrl, 64);
    return {
      id, type: 'image', timestamp,
      textContent: `Image - ${imageFile.type}`,
      imageDataUrl: dataUrl,
      thumbnailDataUrl: thumbnail,
      sourceUrl: window.location.href,
    };
  }

  // Check for HTML (rich text)
  const html = clipboardData.getData('text/html');
  if (html) {
    const text = clipboardData.getData('text/plain') || html;
    return {
      id, type: 'richText', timestamp,
      textContent: text,
      imageDataUrl: null, thumbnailDataUrl: null,
      sourceUrl: window.location.href,
    };
  }

  // Plain text
  const text = clipboardData.getData('text/plain');
  if (text) {
    return {
      id, type: 'text', timestamp,
      textContent: text,
      imageDataUrl: null, thumbnailDataUrl: null,
      sourceUrl: window.location.href,
    };
  }

  return null;
}
```

### Writing to clipboard (paste from history)

```typescript
async function pasteItem(item: ClipboardItem): Promise<void> {
  try {
    if (item.type === 'image' && item.imageDataUrl) {
      const blob = await dataUrlToBlob(item.imageDataUrl);
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type]: blob })
      ]);
    } else if (item.textContent) {
      await navigator.clipboard.writeText(item.textContent);
    }
  } catch (err) {
    // Fallback: use deprecated execCommand
    const textarea = document.createElement('textarea');
    textarea.value = item.textContent || '';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }
}
```

---

## 8. Sprite Animation System

### Agent file structure (from native app)

The native app uses `.agent` bundles with an `.acd` text definition file and a `sprite_map.png`. For the web module, pre-parse these into a JSON manifest:

```typescript
// Pre-processed agent data (ship as JSON + PNG)
interface AgentData {
  character: {
    width: number;    // frame width in px (e.g., 124)
    height: number;   // frame height in px (e.g., 93)
  };
  animations: {
    [name: string]: {
      frames: Array<{
        imageIndices: number[];   // sprite map grid indices
        duration: number;         // centiseconds (divide by 100 for seconds)
        soundIndex: number | null;
      }>;
    };
  };
  spriteMapUrl: string;           // URL to sprite_map.png
  soundBaseUrl: string;           // URL prefix for sound files
}
```

### Canvas rendering

```typescript
function renderFrame(
  ctx: CanvasRenderingContext2D,
  spriteMap: HTMLImageElement,
  frameWidth: number,
  frameHeight: number,
  imageIndices: number[]
) {
  const columns = Math.floor(spriteMap.width / frameWidth);

  ctx.clearRect(0, 0, frameWidth, frameHeight);

  // Draw each layer (some frames have multiple overlapping images)
  for (const index of imageIndices) {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const sx = col * frameWidth;
    const sy = row * frameHeight;

    ctx.drawImage(
      spriteMap,
      sx, sy, frameWidth, frameHeight,  // source rect
      0, 0, frameWidth, frameHeight     // dest rect
    );
  }
}
```

### Animation playback loop

```typescript
function playAnimation(animation: AgentAnimation, onComplete?: () => void) {
  let frameIndex = 0;

  function playNextFrame() {
    if (frameIndex >= animation.frames.length) {
      onComplete?.();
      return;
    }

    const frame = animation.frames[frameIndex];
    renderFrame(ctx, spriteMap, charWidth, charHeight, frame.imageIndices);

    // Play sound if present
    if (frame.soundIndex !== null && settings.playSounds) {
      playSound(frame.soundIndex);
    }

    frameIndex++;
    setTimeout(playNextFrame, frame.duration * 10); // centiseconds to ms
  }

  playNextFrame();
}
```

### Idle timer

```typescript
function startIdleTimer() {
  const delay = IDLE_ANIMATION_MIN + Math.random() * (IDLE_ANIMATION_MAX - IDLE_ANIMATION_MIN);

  idleTimerRef.current = setTimeout(() => {
    const idleAnimations = ['IdleFingerTap', 'IdleSideToSide', 'IdleEyeBrowRaise',
      'IdleRopePile', 'IdleSnooze', 'IdleHeadScratch', 'Thinking', 'Processing',
      'LookUp', 'LookDown', 'LookLeft', 'LookRight'];

    const name = idleAnimations[Math.floor(Math.random() * idleAnimations.length)];
    playAnimation(name, () => {
      showInitialFrame();
      startIdleTimer(); // schedule next
    });
  }, delay);
}
```

---

## 9. Drag-and-Drop Character Positioning

```typescript
function useCharacterDrag(ref: RefObject<HTMLElement>, locked: boolean) {
  const [position, setPosition] = useState(() => {
    const saved = localStorage.getItem('clippy-position');
    if (saved) return JSON.parse(saved);
    return {
      x: window.innerWidth - CLIPPY_BASE_SIZE.width - CLIPPY_EDGE_INSET,
      y: window.innerHeight - CLIPPY_BASE_SIZE.height - CLIPPY_EDGE_INSET,
    };
  });

  // Mouse/touch drag handlers
  // Threshold: 10px movement before starting drag (to distinguish from clicks)
  // Save position to localStorage on drag end
  // Clamp to viewport on window resize

  return { position, style: { left: position.x, top: position.y } };
}
```

---

## 10. Context Menu (Right-Click)

Implement as a custom dropdown menu (not browser native `contextmenu`).

**Menu items:**
```
┌──────────────────────┐
│ Play Animation       │
├──────────────────────┤
│ Size              ▸  │ → XS  S  ✓M  L  XL
│ Lock Position        │ (toggles to "Unlock Position")
│ Mute Sounds          │ (toggles to "Unmute Sounds")
├──────────────────────┤
│ Hide Clippy          │
└──────────────────────┘
```

**Styling:**
```
bg-white dark:bg-gray-800
border border-gray-200 dark:border-gray-700
rounded-lg shadow-lg
min-w-[180px]
py-1
```

**Menu item styling:**
```
px-3 py-1.5
text-[13px]
hover:bg-blue-500 hover:text-white
```

---

## 11. Responsive & Theme Considerations

### Dark mode
All colors must support both light and dark via Tailwind `dark:` variants:

| Element | Light | Dark |
|---|---|---|
| Panel background | `bg-white/80` | `bg-gray-900/85` |
| Text primary | `text-gray-900` | `text-gray-100` |
| Text secondary | `text-gray-500` | `text-gray-400` |
| Hover background | `bg-gray-900/[0.06]` | `bg-white/[0.06]` |
| Press background | `bg-blue-500/[0.18]` | `bg-blue-400/[0.18]` |
| Section card bg | `bg-gray-900/[0.03]` | `bg-white/[0.03]` |
| Dividers | `border-gray-200` | `border-gray-700` |
| Balloon bg | `bg-white` | `bg-gray-800` |

### Viewport clamping
- All panels clamp to viewport bounds
- Character position validated on `window.resize`
- Detail panel flips to right side if not enough space on left

---

## 12. Animation & Transition Specs

| Animation | Duration | Easing | Property |
|---|---|---|---|
| Settings slide in/out | 200ms | ease-in-out | transform + opacity |
| Row hover highlight | 150ms | ease | background-color, opacity |
| Expand/Delete button appear | 150ms | ease | opacity |
| Panel open | 200ms | ease-out | opacity + scale(0.95 → 1) |
| Panel close | 150ms | ease-in | opacity + scale(1 → 0.95) |
| Balloon appear | 200ms | ease-out | opacity + translateY |
| Balloon dismiss | 300ms | ease-in | opacity |
| Copy confirmation | 200ms | ease-in-out | color (accent → green) |

---

## 13. Module Integration API

For the Lovable multi-tool website, export a clean module interface:

```typescript
// index.tsx — Module public API

export { ClippyModule } from './ClippyModule';

// Module metadata for the tool registry
export const moduleInfo = {
  id: 'clippy-clipboard',
  name: 'Clippy Clipboard Manager',
  description: 'Clipboard history with animated assistant',
  icon: 'clipboard',          // icon name for the tool grid
  version: '1.0.0',
  category: 'productivity',
};

// Optional: expose programmatic API for other modules
export function getClipboardHistory(): Promise<ClipboardItem[]>;
export function clearClipboardHistory(): Promise<void>;
```

---

## 14. Browser Compatibility Notes

| Feature | Chrome | Firefox | Safari | Fallback |
|---|---|---|---|---|
| `navigator.clipboard.readText()` | Yes | Yes (with permission) | Yes (13.1+) | `event.clipboardData` |
| `navigator.clipboard.write()` | Yes | Yes (127+) | Yes (13.1+) | `execCommand('copy')` |
| `ClipboardEvent.clipboardData` | Yes | Yes | Yes | — |
| `backdrop-filter: blur()` | Yes | Yes (103+) | Yes | Solid bg fallback |
| IndexedDB | Yes | Yes | Yes | localStorage fallback |
| `crypto.randomUUID()` | Yes (92+) | Yes (95+) | Yes (15.4+) | uuid library |
| Web Audio API | Yes | Yes | Yes | `<audio>` elements |

### Permission considerations
- `navigator.clipboard.read()` requires user gesture + Permissions API grant
- `copy`/`cut` event listeners work without permissions (synchronous via `event.clipboardData`)
- For best reliability, capture from `event.clipboardData` in the event handler, not from `navigator.clipboard` asynchronously

---

## 15. File Deliverables

When building in Lovable, create these files in this order:

1. **`types.ts`** — All interfaces
2. **`constants.ts`** — All magic numbers
3. **`useClippySettings.ts`** — Settings hook (localStorage)
4. **`useClipboardStore.ts`** — IndexedDB store hook
5. **`useClipboardMonitor.ts`** — Clipboard capture hook
6. **`SearchBar.tsx`** — Search input component
7. **`ClipboardItemRow.tsx`** — Row component with hover states
8. **`ClipboardHistoryPanel.tsx`** — Main panel with list + settings
9. **`DetailPanel.tsx`** — Preview panel
10. **`SettingsPanel.tsx`** — Inline settings
11. **`ClippyBalloon.tsx`** — Speech bubble
12. **`spriteExtractor.ts`** — Sprite map frame extraction
13. **`useClippyAnimations.ts`** — Animation controller
14. **`ClippyCharacter.tsx`** — Animated character
15. **`ClippyModule.tsx`** — Root orchestrator
16. **`index.tsx`** — Public exports

### Assets to include
- `clippit_sprite_map.png` — Sprite sheet (extract from native app bundle)
- `clippit_animations.json` — Pre-parsed animation data (convert from .acd file)
- Sound MP3 files in `assets/sounds/`

---

## 16. Key Differences from Native App

| Aspect | Native (macOS) | Web Module |
|---|---|---|
| Clipboard scope | System-wide (all apps) | Browser tab only |
| Item types | text, richText, image, file | text, richText, image (no files) |
| Persistence | Core Data (SQLite) | IndexedDB |
| Settings | UserDefaults | localStorage |
| Character rendering | SpriteKit (SKScene) | Canvas 2D or CSS sprites |
| Panel rendering | NSPopover + NSPanel | Floating divs with backdrop-blur |
| Sounds | NSSound | Web Audio API |
| Positioning | NSWindow frame manipulation | CSS fixed positioning |
| Multi-display | Validated across all screens | Single viewport |
| App exclusion | Bundle ID based | Not applicable |
| Keyboard shortcut | None (click only) | Could add Ctrl+Shift+V or similar |
| Drag | NSWindow mouseDragged | mousedown/mousemove/mouseup |
