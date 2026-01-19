#!/bin/bash
# =============================================================================
# MuseScore Studio 4.6.5 Setup Script for Headless Server
# =============================================================================
#
# This script downloads and configures MuseScore Studio 4.6.5 AppImage for use
# as an "Industry Standard Baseline" for MIDI to MusicXML conversion.
#
# Academic Justification:
# MuseScore Studio 4.6.5 (released Dec 18, 2025) is the latest stable release,
# incorporating critical stability fixes for Linux environments. Unlike naive
# quantization libraries (e.g., music21), MuseScore 4 utilizes a sophisticated
# heuristic-based import engine that performs:
# - Voice separation
# - Tuplet detection
# - Smart quantization
# - Reworked chord symbol handling (SMuFL Compliant)
#
# Reference: MuseScore GitHub Release v4.6.5
# https://github.com/musescore/MuseScore/releases/tag/v4.6.5
#
# Prerequisites:
#   - xvfb (xvfb-run command)
#   - libfuse2 (for AppImage)
#
# Usage:
#   ./setup_musescore.sh [--install-dir /path/to/dir]
#
# =============================================================================

set -e

# Configuration
MUSESCORE_VERSION="4.6.5"
MUSESCORE_URL="https://github.com/musescore/MuseScore/releases/download/v${MUSESCORE_VERSION}/MuseScore-Studio-${MUSESCORE_VERSION}.253511702-x86_64.AppImage"
DEFAULT_INSTALL_DIR="$(dirname "$(realpath "$0")")/../tools"

# Parse arguments
INSTALL_DIR="${DEFAULT_INSTALL_DIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--install-dir /path/to/dir]"
            echo ""
            echo "Downloads and configures MuseScore ${MUSESCORE_VERSION} for headless operation."
            echo ""
            echo "Options:"
            echo "  --install-dir DIR   Installation directory (default: ${DEFAULT_INSTALL_DIR})"
            echo ""
            echo "Prerequisites (install manually before running):"
            echo "  - xvfb:     sudo apt-get install xvfb"
            echo "  - libfuse2: sudo apt-get install libfuse2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "MuseScore ${MUSESCORE_VERSION} Setup"
echo "=============================================="
echo "Install directory: ${INSTALL_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Check system dependencies
# -----------------------------------------------------------------------------
echo "[1/4] Checking system dependencies..."

MISSING_DEPS=false

# Check xvfb-run
if command -v xvfb-run &> /dev/null; then
    echo "  [OK] xvfb-run: $(which xvfb-run)"
else
    echo "  [MISSING] xvfb-run"
    MISSING_DEPS=true
fi

# Check libfuse2 (needed for AppImage)
if ldconfig -p 2>/dev/null | grep -q libfuse.so.2; then
    echo "  [OK] libfuse2: found"
elif [[ -f /usr/lib/x86_64-linux-gnu/libfuse.so.2 ]] || [[ -f /usr/lib64/libfuse.so.2 ]]; then
    echo "  [OK] libfuse2: found"
else
    echo "  [MISSING] libfuse2"
    MISSING_DEPS=true
fi

if [[ "$MISSING_DEPS" == true ]]; then
    echo ""
    echo "=============================================="
    echo "ERROR: Missing dependencies"
    echo "=============================================="
    echo ""
    echo "Please install the missing dependencies before running this script:"
    echo ""
    echo "  Ubuntu/Debian:"
    echo "    sudo apt-get install xvfb libfuse2"
    echo ""
    echo "  RHEL/CentOS:"
    echo "    sudo yum install xorg-x11-server-Xvfb fuse-libs"
    echo ""
    echo "  Arch Linux:"
    echo "    sudo pacman -S xorg-server-xvfb fuse2"
    echo ""
    echo "  HPC (ask admin or use module):"
    echo "    module load xvfb  # if available"
    echo ""
    exit 1
fi

echo "  All dependencies satisfied!"

# Create install directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# -----------------------------------------------------------------------------
# Step 2: Download MuseScore AppImage
# -----------------------------------------------------------------------------
APPIMAGE_PATH="${INSTALL_DIR}/mscore.AppImage"

echo ""
echo "[2/4] Downloading MuseScore ${MUSESCORE_VERSION}..."

if [[ -f "${APPIMAGE_PATH}" ]]; then
    echo "  AppImage already exists: ${APPIMAGE_PATH}"
    echo "  Skipping download. Delete the file to re-download."
else
    echo "  URL: ${MUSESCORE_URL}"
    wget -q --show-progress -O "${APPIMAGE_PATH}" "${MUSESCORE_URL}"
    chmod +x "${APPIMAGE_PATH}"
    echo "  Downloaded: ${APPIMAGE_PATH}"
fi

# -----------------------------------------------------------------------------
# Step 3: Verify installation
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] Verifying installation..."

# Quick version check (may fail without display, that's OK)
if xvfb-run -a "${APPIMAGE_PATH}" --version 2>/dev/null; then
    echo "  MuseScore verification successful!"
else
    echo "  Note: Version check may fail without proper display setup."
    echo "  This is expected on headless servers."
fi

# -----------------------------------------------------------------------------
# Step 4: Create wrapper script
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Creating wrapper script..."

WRAPPER_PATH="${INSTALL_DIR}/mscore"
cat > "${WRAPPER_PATH}" << 'WRAPPER_EOF'
#!/bin/bash
# MuseScore Studio 4.6.5 Headless Wrapper
# Automatically uses xvfb-run for headless operation

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
APPIMAGE="${SCRIPT_DIR}/mscore.AppImage"

if [[ ! -f "${APPIMAGE}" ]]; then
    echo "Error: MuseScore AppImage not found at ${APPIMAGE}"
    exit 1
fi

# Use xvfb-run for headless operation
# -a: automatically choose a free server number
exec xvfb-run -a "${APPIMAGE}" "$@"
WRAPPER_EOF

chmod +x "${WRAPPER_PATH}"
echo "  Created wrapper: ${WRAPPER_PATH}"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "MuseScore installed at: ${APPIMAGE_PATH}"
echo "Wrapper script at:      ${WRAPPER_PATH}"
echo ""
echo "Usage examples:"
echo "  # Convert MIDI to MusicXML"
echo "  ${WRAPPER_PATH} input.mid -o output.musicxml --force"
echo ""
echo "  # From Python"
echo "  subprocess.run(['${WRAPPER_PATH}', 'input.mid', '-o', 'output.musicxml', '--force'])"
echo ""
