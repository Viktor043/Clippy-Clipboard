# Clippy — Clipboard Manager for macOS

## Getting Started

### Installation

1. Open **Clippy.dmg**
2. Drag **Clippy** into the **Applications** folder
3. Open Clippy from Applications
4. **First launch:** macOS may show a security warning since the app isn't from the App Store. To open it:
   - **Right-click** Clippy.app → click **Open** → click **Open** again in the dialog
   - You only need to do this once

### What You'll See

When Clippy launches, two things appear:

- **A paperclip icon** in your menu bar (top-right of your screen)
- **The Clippy character** — a friendly animated paperclip in the corner of your screen

Clippy runs quietly in the background, saving everything you copy.

---

## How It Works

Every time you copy something — text, an image, a file, or rich text — Clippy saves it to your clipboard history. You can go back and find anything you've copied before, even after you've copied something else.

---

## Using Clippy

### Viewing Your Clipboard History

You have two ways to open your clipboard history:

- **Click the paperclip icon** in your menu bar, OR
- **Click the Clippy character** on your screen

A panel opens showing all your recent copies, newest first. Each entry shows:
- A preview of what was copied
- How long ago it was copied (e.g., "2 min ago")

### Searching Your History

At the top of the clipboard panel there's a search bar. Just start typing to filter your history. Click the **X** to clear your search.

### Pasting from History

Click any item in the list to copy it back to your clipboard. The panel closes automatically. Then just **Cmd+V** in any app to paste it.

### Previewing Full Content

Hover over any item and you'll see an **Expand** button appear. Click it to open a preview panel to the left showing the full content.

In the preview panel you can:
- **Select text** — click and drag to highlight
- **Multi-select** — hold **Cmd** and click to select multiple separate sections
- **Copy selected text** — press **Cmd+C** to copy just what you've selected
- **Copy everything** — click the **Copy All** button at the top
- **Clear selections** — double-click anywhere in the text
- **Close the preview** — click the **>** button (this only closes the preview, not the main panel)

### Deleting Items

Hover over any item and click the **X** button on the right to remove it from your history.

### Clearing All History

Click the **trash icon** in the header. You'll be asked to confirm before everything is deleted.

---

## The Clippy Character

The animated Clippy character lives on your screen and reacts when you copy things:

- **Copy text:** Clippy waves and says "Copied text!"
- **Copy rich text:** "Got that rich text!"
- **Copy an image:** "Nice screenshot!"
- **Copy a file:** "File copied!"

Clippy also plays idle animations when nothing's happening — looking around, tapping fingers, thinking, and more.

### Click Clippy

Click the character to open your clipboard history (same as clicking the menu bar icon).

### Right-Click Clippy

Right-click the character for a menu with these options:

| Option | What it does |
|---|---|
| **Play Animation** | Triggers a random animation |
| **Size** | Change Clippy's size: Extra Small, Small, Medium, Large, Extra Large |
| **Lock Position** | Prevents accidentally dragging Clippy around |
| **Mute Sounds** | Turns off animation sound effects |
| **Hide Clippy** | Hides the character (you can still use the menu bar icon) |

### Moving Clippy

Click and drag the character to move it anywhere on your screen. Your position is remembered between app launches.

---

## Settings

Click the **gear icon** in the clipboard panel to access settings. Click the **back arrow** to return to your clipboard history.

### General

- **Show menu bar icon** — Toggle the paperclip icon in your menu bar on or off

### Clips Management

- **Remember up to** — How many items to keep: 25, 50, 100, 200, or 500
- **Max chars per item** — Maximum text length per entry: 5K, 10K, 20K, or 50K characters

### Appearance

- **Font** — Choose any font installed on your Mac
- **Font size** — Pick from 10pt to 16pt

### Clippy Animation

- **Show Clippy character** — Show or hide the animated character
- **Play sounds** — Toggle animation sound effects
- **Size** — XS, S, M, L, or XL

### Advanced

**Capture types** — Choose what Clippy saves:
- Plain text
- Rich text (formatted text from apps like Pages or Word)
- Images and screenshots
- Files

**Excluded apps** — Apps whose clipboard activity Clippy ignores. Useful for password managers or sensitive apps. You can:
- Type a bundle ID manually (e.g., `com.1password.app`)
- Pick from currently running apps
- Pick from installed apps

### Quit

The **Quit Clippy** button at the bottom of settings closes the app entirely.

---

## Tips & Tricks

- **Quick paste workflow:** Click menu bar icon → click any item → Cmd+V in your target app. The whole thing takes about 2 seconds.

- **Finding old copies:** Use the search bar to find something you copied hours or days ago.

- **Copying parts of a long text:** Expand an item, select just the part you need, press Cmd+C.

- **Multi-select text:** In the preview panel, hold Cmd and click different parts of the text to select multiple sections at once. Cmd+C copies all selected sections.

- **Sensitive apps:** Add password managers to the excluded apps list so Clippy never saves passwords.

- **Declutter:** If Clippy's character is distracting, right-click it and choose "Hide Clippy" — you can still use everything from the menu bar icon.

- **Re-show Clippy:** If you hid the character, go to Settings and toggle "Show Clippy character" back on.

---

## Keyboard Shortcuts

| Shortcut | Where | What it does |
|---|---|---|
| **Cmd+C** | Preview panel | Copy selected text |
| **Cmd+Click** | Preview panel | Add to selection (multi-select) |
| **Double-click** | Preview panel | Clear all selections |

---

## Troubleshooting

**Clippy doesn't appear after opening**
- Check your menu bar for the paperclip icon — Clippy may be running but the character is hidden
- Go to Settings → turn on "Show Clippy character"

**The menu bar icon is missing**
- Open Clippy from Applications
- When the settings window appears, make sure "Show menu bar icon" is turned on

**Clippy doesn't save my copies**
- Make sure the right capture types are enabled in Settings → Advanced
- Check that the app you're copying from isn't in the excluded apps list

**macOS won't let me open Clippy**
- Right-click the app → Open → Open again
- Or go to System Settings → Privacy & Security → click "Open Anyway"

**Clippy is in the way**
- Drag it to a corner, or right-click → Lock Position
- Or right-click → Hide Clippy to remove it from screen entirely
