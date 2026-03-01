#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="BrightnessControl"
BUILD_DIR="build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"

echo "Building $APP_NAME..."

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Compile all Swift sources
swiftc \
    -o "$BUILD_DIR/$APP_NAME" \
    -framework Cocoa \
    -framework IOKit \
    -framework CoreGraphics \
    -target arm64-apple-macos13 \
    -O \
    -swift-version 5 \
    Sources/*.swift

# Create .app bundle
mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"
cp "$BUILD_DIR/$APP_NAME" "$APP_BUNDLE/Contents/MacOS/"
cp Info.plist "$APP_BUNDLE/Contents/"

# Ad-hoc code sign
codesign --force --deep --sign - "$APP_BUNDLE"

echo ""
echo "Build successful: $APP_BUNDLE"
echo ""
echo "To run:  open $APP_BUNDLE"
echo "To install: cp -r $APP_BUNDLE /Applications/"
echo ""
echo "SAFETY NOTE: If your screen gets too dark, sleep/wake your Mac"
echo "             or run: $SCRIPT_DIR/reset-gamma.sh"
