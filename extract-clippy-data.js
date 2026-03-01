#!/usr/bin/env node
/**
 * extract-clippy-data.js
 * Parses clippit.acd (Microsoft Agent Character Definition) and outputs
 * clippy-animations.json for the web renderer.
 */

const fs = require('fs');
const path = require('path');

const ACD_PATH = path.join(__dirname, 'Resources', 'clippit.agent', 'clippit.acd');
const OUTPUT_PATH = path.join(__dirname, 'clippy-web', 'clippy-animations.json');

const content = fs.readFileSync(ACD_PATH, 'utf-8');

// ---- Helpers ----

function fetchBlocks(text, startTag, endTag) {
  const blocks = [];
  let searchFrom = 0;
  while (true) {
    const startIdx = text.indexOf(startTag, searchFrom);
    if (startIdx === -1) break;
    const endIdx = text.indexOf(endTag, startIdx);
    if (endIdx === -1) break;
    blocks.push(text.substring(startIdx, endIdx + endTag.length));
    searchFrom = endIdx + endTag.length;
  }
  return blocks;
}

function getKeyValue(line, key) {
  const regex = new RegExp(`^\\s*${key}\\s*=\\s*(.+)$`);
  const match = line.match(regex);
  return match ? match[1].trim() : null;
}

function getIntValue(text, key) {
  for (const line of text.split('\n')) {
    const val = getKeyValue(line.trim(), key);
    if (val !== null) return parseInt(val, 10);
  }
  return null;
}

function getStringValue(text, key) {
  for (const line of text.split('\n')) {
    const val = getKeyValue(line.trim(), key);
    if (val !== null) return val.replace(/^"|"$/g, '');
  }
  return null;
}

function getDefinitionName(text, keyword) {
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith(keyword + ' ')) {
      const rest = trimmed.substring(keyword.length).trim();
      return rest.replace(/^"|"$/g, '');
    }
  }
  return null;
}

// ---- Parse Character ----

const charBlock = fetchBlocks(content, 'DefineCharacter', 'EndCharacter')[0];
const character = {
  width: getIntValue(charBlock, 'Width'),
  height: getIntValue(charBlock, 'Height'),
};

const spriteSheetCols = 4092 / character.width; // 33
const spriteSheetRows = 2604 / character.height; // 28

// ---- Parse Animations ----

function parseImage(imageBlock) {
  const filename = getStringValue(imageBlock, 'Filename');
  if (!filename) return null;
  // "Images\0045.bmp" -> 45
  const match = filename.match(/(\d+)\.bmp$/);
  return match ? parseInt(match[1], 10) : 0;
}

function parseBranching(branchBlock) {
  const branches = [];
  let branchTo = null;
  let probability = null;

  for (const line of branchBlock.split('\n')) {
    const trimmed = line.trim();
    const btVal = getKeyValue(trimmed, 'BranchTo');
    if (btVal !== null) branchTo = parseInt(btVal, 10);
    const prVal = getKeyValue(trimmed, 'Probability');
    if (prVal !== null) probability = parseInt(prVal, 10);

    if (branchTo !== null && probability !== null) {
      branches.push({ branchTo, probability });
      branchTo = null;
      probability = null;
    }
  }

  return branches;
}

function parseFrame(frameBlock) {
  const duration = getIntValue(frameBlock, 'Duration');
  if (duration === null) return null;

  const exitBranch = getIntValue(frameBlock, 'ExitBranch');

  // Sound
  let sound = null;
  const soundEffect = getStringValue(frameBlock, 'SoundEffect');
  if (soundEffect) {
    const soundMatch = soundEffect.match(/(\d+)\.wav$/);
    if (soundMatch) sound = parseInt(soundMatch[1], 10);
  }

  // Images
  const imageBlocks = fetchBlocks(frameBlock, 'DefineImage', 'EndImage');
  const imageIndices = imageBlocks.map(parseImage).filter(i => i !== null);

  // Branching
  const branchBlocks = fetchBlocks(frameBlock, 'DefineBranching', 'EndBranching');
  let branches = [];
  if (branchBlocks.length > 0) {
    branches = parseBranching(branchBlocks[0]);
  }

  return {
    // For single-image frames, store as number; for multi-image, store as array
    imageIndex: imageIndices.length === 1 ? imageIndices[0] : imageIndices,
    duration: duration * 10, // centiseconds -> milliseconds
    exitBranch: exitBranch !== null ? exitBranch : null,
    branches: branches.length > 0 ? branches : [],
    sound: sound,
  };
}

function parseAnimation(animBlock) {
  const name = getDefinitionName(animBlock, 'DefineAnimation');
  if (!name) return null;

  const frameBlocks = fetchBlocks(animBlock, 'DefineFrame', 'EndFrame');
  const frames = frameBlocks.map(parseFrame).filter(f => f !== null);

  return { name, frames };
}

const animBlocks = fetchBlocks(content, 'DefineAnimation', 'EndAnimation');
const animationsArray = animBlocks.map(parseAnimation).filter(a => a !== null);

const animations = {};
for (const anim of animationsArray) {
  animations[anim.name] = { frames: anim.frames };
}

// ---- Parse States ----

function parseState(stateBlock) {
  const name = getDefinitionName(stateBlock, 'DefineState');
  if (!name) return null;

  const animationNames = [];
  for (const line of stateBlock.split('\n')) {
    const trimmed = line.trim();
    const val = getKeyValue(trimmed, 'Animation');
    if (val !== null) {
      animationNames.push(val.replace(/^"|"$/g, ''));
    }
  }

  return { name, animationNames };
}

const stateBlocks = fetchBlocks(content, 'DefineState', 'EndState');
const statesArray = stateBlocks.map(parseState).filter(s => s !== null);

const states = {};
for (const state of statesArray) {
  states[state.name] = state.animationNames;
}

// ---- Output ----

const output = {
  character,
  spriteSheet: {
    columns: spriteSheetCols,
    rows: spriteSheetRows,
    imageUrl: 'clippit_sprite_map.png',
  },
  sounds: {
    baseUrl: 'clippy-sounds/',
    files: Array.from({ length: 15 }, (_, i) => `clippit_${i}.mp3`),
  },
  animations,
  states,
  // Convenience lists for the renderer
  idleAnimations: [
    'IdleFingerTap', 'IdleSideToSide', 'IdleEyeBrowRaise',
    'IdleRopePile', 'IdleSnooze', 'IdleHeadScratch',
    'IdleAtom', 'Idle1_1',
    'LookUp', 'LookDown', 'LookLeft', 'LookRight',
    'LookUpLeft', 'LookUpRight', 'LookDownLeft', 'LookDownRight',
    'Thinking', 'Processing', 'CheckingSomething',
  ],
  clickAnimations: [
    'Wave', 'Greeting', 'GetAttention', 'Congratulate',
    'GetArtsy', 'GetTechy', 'GetWizardy', 'Explain',
  ],
};

// Ensure output directory exists
const outputDir = path.dirname(OUTPUT_PATH);
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

fs.writeFileSync(OUTPUT_PATH, JSON.stringify(output, null, 2));

console.log(`Extracted ${Object.keys(animations).length} animations, ${Object.keys(states).length} states`);
console.log(`Output: ${OUTPUT_PATH}`);
