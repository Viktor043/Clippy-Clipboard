#!/bin/bash
# Emergency gamma reset — run this if BrightnessControl made your screen too dark
# Alternatively: sleep/wake your Mac (close and open lid) to reset gamma
python3 -c "
import ctypes
import ctypes.util

cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreGraphics'))
cg.CGDisplayRestoreColorSyncSettings()
print('Gamma reset to defaults.')
"
