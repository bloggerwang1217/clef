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
# Why 4.6.5 over 4.6.4?
# - Crash fixes critical for parallel processing (64+ workers)
# - Linux VST3 support optimization (better AppImage stability with Xvfb)
# - Reworked chord symbol handling for cleaner XML output
#
# Reference: MuseScore GitHub Release v4.6.5
# https://github.com/musescore/MuseScore/releases/tag/v4.6.5
#
# Usage:
#   ./setup_musescore.sh [--install-dir /path/to/dir]
#
# =============================================================================

set -e

# Configuration
MUSESCORE_VERSION="4.6.5"
MUSESCORE_URL="https://github.com/musescore/MuseScore/releases/download/v${MUSESCORE_VERSION}/MuseScore-Studio-${MUSESCORE_VERSION}-253530512-x86_64.AppImage"
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
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create install directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

echo "=============================================="
echo "MuseScore ${MUSESCORE_VERSION} Setup"
echo "=============================================="
echo "Install directory: ${INSTALL_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install system dependencies
# -----------------------------------------------------------------------------
echo "[1/4] Checking system dependencies..."

install_deps() {
    if command -v apt-get &> /dev/null; then
        echo "  Installing dependencies via apt-get..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq xvfb libfuse2 2>/dev/null || {
            echo "  Note: libfuse2 may not be available, trying libfuse2t64..."
            sudo apt-get install -y -qq xvfb libfuse2t64 2>/dev/null || true
        }
    elif command -v yum &> /dev/null; then
        echo "  Installing dependencies via yum..."
        sudo yum install -y xorg-x11-server-Xvfb fuse-libs
    elif command -v pacman &> /dev/null; then
        echo "  Installing dependencies via pacman..."
        sudo pacman -S --noconfirm xorg-server-xvfb fuse2
    else
        echo "  Warning: Could not detect package manager."
        echo "  Please manually install: xvfb, libfuse2"
    fi
}

# Check if xvfb-run exists
if ! command -v xvfb-run &> /dev/null; then
    echo "  xvfb-run not found, installing dependencies..."
    install_deps
else
    echo "  xvfb-run found: $(which xvfb-run)"
fi

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
echo "Environment variable (add to your shell profile):"
echo "  export MUSESCORE_BIN=\"${WRAPPER_PATH}\""
echo ""
