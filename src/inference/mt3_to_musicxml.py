#!/usr/bin/env python3
"""
MT3 MIDI to MusicXML Conversion Pipeline

This script converts MT3's single-track MIDI output to MusicXML format
for evaluation against end-to-end audio-to-score systems.

Design Rationale (Academic Justification):
==========================================

1. QUANTIZATION SETTINGS
   - Default: quarterLengthDivisors=(4, 3)
   - 4 = sixteenth notes (1/16)
   - 3 = eighth-note triplets (1/8T)
   - Reference: music21 default settings
     https://www.music21.org/music21docs/moduleReference/moduleMidiTranslate.html

   This matches the temporal resolution of most Western classical music notation.
   Finer divisions (e.g., 32nd notes) are rare in the ASAP dataset repertoire.

2. HAND SEPARATION ALGORITHM
   - Method: Pitch-based heuristic (split at Middle C, MIDI pitch 60)
   - Reference: Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
     https://www.cemfi.de/wp-content/papercite-data/pdf/hadjakos-2019-detectinghands.pdf

   From Hadjakos et al. (2019), Section 1 Introduction:
   > "So far, most systems allocate notes by splitting at the middle C:
   >  deeper notes are assigned to the lower staff; all others to the top.
   >  This approach is highly inaccurate..."

   We adopt this "Middle C split" heuristic precisely because it is the
   industry-default rule that “most systems” use (MuseScore, Finale, etc.).
   It is intentionally naive and stands in as the straw-man pipeline baseline.

   NOTE: Hadjakos et al. also propose Kalman Filter / RNN improvements,
   but we do NOT adopt them; the baseline stays purely rule-based without
   learned post-processing to reflect the industrial default.

3. REPRODUCIBILITY
   - All parameters are deterministic and documented
   - music21 is the de facto standard for computational musicology
   - Reference: Cuthbert & Ariza (2010) "music21: A Toolkit for Computer-Aided Musicology and Symbolic Music Data"

Usage:
    python mt3_to_musicxml.py --input mt3_output.mid --output result.musicxml
    python mt3_to_musicxml.py --input_dir midi/ --output_dir xml/

Author: Clef Project
License: Apache-2.0
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import music21
from music21 import converter, stream, instrument, clef, meter, key

# Suppress music21 warnings for cleaner output (backward-compatible)
if hasattr(music21, "Music21DeprecationWarning"):
    warnings.filterwarnings('ignore', category=music21.Music21DeprecationWarning)
else:
    warnings.filterwarnings('ignore', category=UserWarning, message=".*music21.*deprecated.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION: Academic Justification for Each Parameter
# =============================================================================

class QuantizationConfig:
    """
    Quantization settings with academic justification.

    Reference: music21.midi.translate module documentation
    https://www.music21.org/music21docs/moduleReference/moduleMidiTranslate.html
    """

    # Default: sixteenth notes + eighth-note triplets
    # This is music21's default and covers most classical piano repertoire
    QUARTER_LENGTH_DIVISORS: Tuple[int, ...] = (4, 3)

    # Alternative settings (documented for ablation studies):
    # (2,)     = eighth notes only (coarse, may lose detail)
    # (4,)     = sixteenth notes only (no triplets)
    # (4, 6)   = sixteenth + sixteenth triplets (finer)
    # (8, 6)   = 32nd notes + sixteenth triplets (very fine)
    # (4, 12)  = sixteenth + 32nd triplets (very fine triplets)

    # Process both note onsets and durations
    PROCESS_OFFSETS: bool = True
    PROCESS_DURATIONS: bool = True


class HandSeparationConfig:
    """
    Hand separation settings with academic justification.

    Reference: Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
    - Naive pitch-based split is used as baseline
    - Split point at Middle C (MIDI 60) is industry standard

    Reference: MuseScore MIDI import default behavior
    https://musescore.org/en/handbook/3/midi-import
    """

    # Split point: Middle C (C4) = MIDI pitch 60
    # Notes >= 60 go to right hand (treble clef)
    # Notes < 60 go to left hand (bass clef)
    SPLIT_POINT: int = 60

    # Whether to attempt voice separation within each hand
    # Disabled by default: MT3 doesn't provide voice information
    ENABLE_VOICE_SEPARATION: bool = False

    # Minimum notes required to create a hand part
    # Prevents empty parts if all notes are in one hand
    MIN_NOTES_PER_HAND: int = 1


# =============================================================================
# CORE CONVERSION FUNCTIONS
# =============================================================================

def load_midi(midi_path: str, quantize: bool = True) -> music21.stream.Score:
    """
    Load MIDI file and optionally quantize.

    Args:
        midi_path: Path to MIDI file
        quantize: Whether to apply quantization during parsing

    Returns:
        music21 Score object
    """
    logger.info(f"Loading MIDI: {midi_path}")

    if quantize:
        # Use music21's built-in quantization during parsing
        score = converter.parse(
            midi_path,
            format='midi',
            forceSource=True,
            quantizePost=True,
            quarterLengthDivisors=QuantizationConfig.QUARTER_LENGTH_DIVISORS
        )
    else:
        score = converter.parse(
            midi_path,
            format='midi',
            forceSource=True,
            quantizePost=False
        )

    return score


def quantize_score(
    score: music21.stream.Score,
    divisors: Tuple[int, ...] = QuantizationConfig.QUARTER_LENGTH_DIVISORS
) -> music21.stream.Score:
    """
    Apply quantization to a score.

    This is the critical step that converts continuous MIDI timing
    to discrete musical notation values.

    Args:
        score: music21 Score object
        divisors: quarterLengthDivisors for quantization grid

    Returns:
        Quantized score

    Academic Note:
        Quantization is REQUIRED for score-level evaluation.
        MV2H's note value metric (F_val) requires discrete durations.
        Without quantization, we cannot compute tree edit distance
        or voice separation accuracy.
    """
    logger.info(f"Quantizing with divisors: {divisors}")

    score.quantize(
        quarterLengthDivisors=list(divisors),
        processOffsets=QuantizationConfig.PROCESS_OFFSETS,
        processDurations=QuantizationConfig.PROCESS_DURATIONS,
        inPlace=True
    )

    return score


def separate_hands(
    score: music21.stream.Score,
    split_point: int = HandSeparationConfig.SPLIT_POINT
) -> music21.stream.Score:
    """
    Separate notes into right hand (treble) and left hand (bass) parts.

    Algorithm: Pitch-based heuristic split at Middle C (MIDI 60)

    Args:
        score: music21 Score object (typically single-track from MT3)
        split_point: MIDI pitch number for split (default: 60 = Middle C)

    Returns:
        New score with two parts (right hand, left hand)
    """
    logger.info(f"Separating hands at MIDI pitch {split_point}")

    # Create new score with piano instrument
    new_score = stream.Score()

    # Create right hand (treble) and left hand (bass) parts
    right_hand = stream.Part()
    right_hand.id = 'Right Hand'
    right_hand.insert(0, instrument.Piano())
    right_hand.insert(0, clef.TrebleClef())

    left_hand = stream.Part()
    left_hand.id = 'Left Hand'
    left_hand.insert(0, instrument.Piano())
    left_hand.insert(0, clef.BassClef())

    # Collect all notes from the original score
    all_notes = list(score.flatten().notesAndRests)

    # Track statistics for logging
    rh_count = 0
    lh_count = 0

    for element in all_notes:
        if isinstance(element, music21.note.Note):
            # Single note: assign based on pitch
            if element.pitch.midi >= split_point:
                right_hand.insert(element.offset, element)
                rh_count += 1
            else:
                left_hand.insert(element.offset, element)
                lh_count += 1

        elif isinstance(element, music21.chord.Chord):
            # Chord: split by pitch, may result in separate chords per hand
            rh_pitches = []
            lh_pitches = []

            for pitch in element.pitches:
                if pitch.midi >= split_point:
                    rh_pitches.append(pitch)
                else:
                    lh_pitches.append(pitch)

            # Create separate chords for each hand
            if rh_pitches:
                rh_chord = music21.chord.Chord(rh_pitches)
                rh_chord.duration = element.duration
                right_hand.insert(element.offset, rh_chord)
                rh_count += len(rh_pitches)

            if lh_pitches:
                lh_chord = music21.chord.Chord(lh_pitches)
                lh_chord.duration = element.duration
                left_hand.insert(element.offset, lh_chord)
                lh_count += len(lh_pitches)

        elif isinstance(element, music21.note.Rest):
            # Rests: add aligned rests to both hands
            ql = element.duration.quarterLength
            right_hand.insert(element.offset, music21.note.Rest(quarterLength=ql))
            left_hand.insert(element.offset, music21.note.Rest(quarterLength=ql))

    logger.info(f"Hand separation complete: RH={rh_count} notes, LH={lh_count} notes")

    # Only add parts that have notes
    if rh_count >= HandSeparationConfig.MIN_NOTES_PER_HAND:
        new_score.insert(0, right_hand)
    if lh_count >= HandSeparationConfig.MIN_NOTES_PER_HAND:
        new_score.insert(0, left_hand)

    return new_score


def add_notation_elements(
    score: music21.stream.Score,
    time_signature: str = '4/4',
    key_signature: str = 'C'
) -> music21.stream.Score:
    """
    Add essential notation elements for valid MusicXML output.

    Args:
        score: music21 Score object
        time_signature: Time signature string (e.g., '4/4', '3/4')
        key_signature: Key signature string (e.g., 'C', 'G', 'F#')

    Returns:
        Score with notation elements added
    """
    logger.info(f"Adding notation: time={time_signature}, key={key_signature}")

    # Add time signature and key signature to each part
    for part in score.parts:
        # Add at the beginning if not present
        if not part.flatten().getElementsByClass(meter.TimeSignature):
            part.insert(0, meter.TimeSignature(time_signature))

        if not part.flatten().getElementsByClass(key.KeySignature):
            part.insert(0, key.Key(key_signature))

    # Make notation (add measures, beams, etc.)
    score.makeNotation(inPlace=True)

    return score


def convert_mt3_midi_to_musicxml(
    midi_path: str,
    output_path: str,
    time_signature: str = '4/4',
    key_signature: str = 'C',
    quantization_divisors: Optional[Tuple[int, ...]] = None
) -> str:
    """
    Full conversion pipeline: MT3 MIDI → MusicXML

    Pipeline Steps:
    1. Load MIDI with quantization
    2. Apply additional quantization (if needed)
    3. Separate hands using pitch-based heuristic
    4. Add notation elements
    5. Export to MusicXML

    Args:
        midi_path: Input MIDI file path
        output_path: Output MusicXML file path
        time_signature: Default time signature
        key_signature: Default key signature
        quantization_divisors: Override default quantization settings

    Returns:
        Path to output MusicXML file
    """
    # Step 1: Load MIDI
    score = load_midi(midi_path, quantize=True)

    # Step 2: Additional quantization if custom divisors specified
    if quantization_divisors:
        score = quantize_score(score, quantization_divisors)

    # Step 3: Separate hands
    score = separate_hands(score)

    # Step 4: Add notation elements
    score = add_notation_elements(score, time_signature, key_signature)

    # Step 5: Export to MusicXML
    logger.info(f"Writing MusicXML: {output_path}")
    score.write('musicxml', fp=output_path)

    return output_path


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_directory(
    input_dir: str,
    output_dir: str,
    time_signature: str = '4/4',
    key_signature: str = 'C'
) -> List[Tuple[str, str]]:
    """
    Process all MIDI files in a directory.

    Args:
        input_dir: Directory containing MIDI files
        output_dir: Directory for output MusicXML files
        time_signature: Default time signature for all files
        key_signature: Default key signature for all files

    Returns:
        List of (input_path, output_path) tuples
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, str]] = []
    midi_files = list(input_path.rglob('*.mid')) + list(input_path.rglob('*.midi'))

    logger.info(f"Found {len(midi_files)} MIDI files in {input_dir}")

    for midi_file in midi_files:
        rel = midi_file.relative_to(input_path)
        output_file = (output_path / rel).with_suffix('.musicxml')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            convert_mt3_midi_to_musicxml(
                str(midi_file),
                str(output_file),
                time_signature=time_signature,
                key_signature=key_signature
            )
            results.append((str(midi_file), str(output_file)))
            logger.info(f"✓ Converted: {midi_file.name}")
        except Exception as e:
            logger.error(f"✗ Failed: {midi_file.name} - {e}")

    logger.info(f"Completed: {len(results)}/{len(midi_files)} files")
    return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert MT3 MIDI output to MusicXML for evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file conversion
  python mt3_to_musicxml.py --input mt3_output.mid --output result.musicxml

  # Batch processing
  python mt3_to_musicxml.py --input_dir midi/ --output_dir xml/

  # Custom quantization (32nd notes)
  python mt3_to_musicxml.py --input file.mid --output file.xml --divisors 8

        """
    )

    # Input/output arguments
    parser.add_argument('--input', '-i', type=str, help='Input MIDI file path')
    parser.add_argument('--output', '-o', type=str, help='Output MusicXML file path')
    parser.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output_dir', type=str, help='Output directory for batch processing')

    # Conversion settings
    parser.add_argument(
        '--time_sig', type=str, default='4/4',
        help='Default time signature (default: 4/4)'
    )
    parser.add_argument(
        '--key', type=str, default='C',
        help='Default key signature (default: C)'
    )
    parser.add_argument(
        '--divisors', type=int, nargs='+', default=[4, 3],
        help='Quantization divisors (default: 4 3 for sixteenth + triplets)'
    )

    args = parser.parse_args()

    # Single file or batch mode
    if args.input and args.output:
        convert_mt3_midi_to_musicxml(
            args.input,
            args.output,
            time_signature=args.time_sig,
            key_signature=args.key,
            quantization_divisors=tuple(args.divisors)
        )
        print(f"✓ Converted: {args.input} → {args.output}")

    elif args.input_dir and args.output_dir:
        results = process_directory(
            args.input_dir,
            args.output_dir,
            time_signature=args.time_sig,
            key_signature=args.key
        )
        print(f"✓ Batch complete: {len(results)} files converted")

    else:
        parser.print_help()
        print("\nError: Must specify either --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
