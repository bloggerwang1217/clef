#!/bin/bash
# Filter MT3 MIDI results to match exactly Zeng's 3700 test samples
# Usage: ./scripts/filter_mt3_by_zeng_samples.sh
#
# This script copies only the MIDI files that correspond to 
# Zeng's test/target/*.pkl samples

set -e

ZENG_TARGET_DIR="/home/bloggerwang/piano-a2s/workspace/feature.asap/test/target"
MIDI_DIR="/home/bloggerwang/clef/data/zeng_5bar_midi"
OUTPUT_DIR="/home/bloggerwang/clef/data/zeng_5bar_midi_test_only"

echo "=========================================="
echo "Filtering MT3 MIDI by Zeng Test Samples"
echo "=========================================="
echo ""

# Check directories exist
if [ ! -d "$ZENG_TARGET_DIR" ]; then
  echo "Error: Zeng target directory not found at $ZENG_TARGET_DIR"
  exit 1
fi

if [ ! -d "$MIDI_DIR" ]; then
  echo "Error: MIDI directory not found at $MIDI_DIR"
  exit 1
fi

# Create/clean output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count expected samples
expected=$(ls "$ZENG_TARGET_DIR"/*.pkl 2>/dev/null | wc -l)
echo "Expected samples from Zeng: $expected"
echo ""

# Filter MIDI files based on Zeng target filenames
echo "Filtering MIDI files..."
count=0
missing=0

for pkl in "$ZENG_TARGET_DIR"/*.pkl; do
  sample=$(basename "$pkl" .pkl)
  midi_file="$MIDI_DIR/${sample}.mid"
  
  if [ -f "$midi_file" ]; then
    cp "$midi_file" "$OUTPUT_DIR/"
    count=$((count + 1))
  else
    missing=$((missing + 1))
    if [ $missing -le 10 ]; then
      echo "  Missing: ${sample}.mid"
    fi
  fi
done

if [ $missing -gt 10 ]; then
  echo "  ... and $((missing - 10)) more missing"
fi

echo ""
echo "=========================================="
echo "✓ Filtering complete!"
echo "=========================================="
total_files=$(ls "$OUTPUT_DIR"/*.mid 2>/dev/null | wc -l)
echo "Copied MIDI files: $total_files"
echo "Missing MIDI files: $missing"
echo "Expected: $expected"
echo ""
echo "Results saved to: $OUTPUT_DIR"
