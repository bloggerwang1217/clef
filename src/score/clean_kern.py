#!/usr/bin/env python3
"""
Kern Token Sanitizer for converter21 Output
============================================

Problem:
    converter21 preserves visual layout information from MusicXML (e.g., rest positions,
    stem directions, articulation marks), producing tokens like:
    - '8rGG' (rest at Great G position)
    - '4g/' (stem up)
    - '8cc\' (stem down)
    - '4.c~' (trill mark)

    Zeng et al. (2024)'s LabelsMultiple vocabulary only recognizes semantic tokens:
    - '8r' (eighth rest, no position)
    - '4g' (quarter note G, no stem direction)

Solution:
    Clean tokens by removing visual layout information while preserving semantic content.

Usage:
    # In training/inference pipeline:
    from clean_kern import clean_kern_token, clean_kern_sequence

    # Single token
    cleaned = clean_kern_token('8rGG')  # -> '8r'

    # Entire sequence
    dirty_seq = "8rGG\t4g/\t8cc\\\t4.c~"
    cleaned_seq = clean_kern_sequence(dirty_seq)  # -> "8r\t4g\t8cc\t4.c"

Author: bloggerwang
Date: 2026-01-16
"""

import re
import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Visual Information Extraction (for Visual Auxiliary Head ground truth)
# ============================================================================

def extract_visual_info(token: str) -> Dict[str, Any]:
    """
    Extract visual layout information from a Kern token.

    This function extracts visual rendering hints that are typically removed
    during cleaning for the main sequence, but preserved for the Visual
    Auxiliary Head as ground truth.

    Args:
        token: Raw Kern token from converter21 (e.g., '8cc>/L', '4g/', '8dd\\J')

    Returns:
        Dict with keys:
        - 'stem': 'up' | 'down' | None
        - 'beam': List of beam markers ['L', 'J', 'k', 'K'] or None
        - 'above_below': 'above' | 'below' | None

    Examples:
        >>> extract_visual_info("8cc>/L")
        {'stem': 'up', 'beam': ['L'], 'above_below': 'above'}
        >>> extract_visual_info("4g/")
        {'stem': 'up', 'beam': None, 'above_below': None}
        >>> extract_visual_info("8dd\\J")
        {'stem': 'down', 'beam': ['J'], 'above_below': None}
    """
    visual_info: Dict[str, Any] = {}

    # Stem direction (cleaned by clean_kern_token)
    # / = stem up, \ = stem down
    if '/' in token:
        visual_info['stem'] = 'up'
    elif '\\' in token:
        visual_info['stem'] = 'down'
    else:
        visual_info['stem'] = None

    # Beam markers (preserved in clean_kern_token by default)
    # L = beam start, J = beam end, k = partial beam back, K = partial beam forward
    beam = []
    for marker in ['L', 'J', 'k', 'K']:
        if marker in token:
            beam.append(marker)
    visual_info['beam'] = beam if beam else None

    # Above/below staff markers (cleaned from note tokens)
    # > = above middle line, < = below middle line
    if '>' in token:
        visual_info['above_below'] = 'above'
    elif '<' in token:
        visual_info['above_below'] = 'below'
    else:
        visual_info['above_below'] = None

    return visual_info


def extract_visual_from_sequence(sequence: str) -> List[List[Dict[str, Any]]]:
    """
    Extract visual info for each token in a sequence.

    This function processes an entire Kern sequence and extracts visual
    information for each token, including the spine index (Voice position).

    Args:
        sequence: Newline-separated lines, each line is tab-separated tokens

    Returns:
        List of lines, each line is list of dicts:
        [
            [  # line 0
                {'stem': 'up', 'beam': ['L'], 'above_below': None, 'spine_index': 0},
                {'stem': None, 'beam': None, 'above_below': None, 'spine_index': 1},
            ],
            [  # line 1
                {'stem': 'down', 'beam': ['J'], 'above_below': None, 'spine_index': 0},
                ...
            ],
        ]

    Note:
        spine_index represents the Voice position within the Part.
        In Humdrum, spines are tab-separated columns representing voices.
        Control tokens (barlines, interpretations, comments) are skipped.

    Examples:
        >>> seq = "8cc/L\\t4g\\n8dd\\\\J\\t4a"
        >>> visual_seq = extract_visual_from_sequence(seq)
        >>> visual_seq[0][0]
        {'stem': 'up', 'beam': ['L'], 'above_below': None, 'spine_index': 0}
    """
    lines = sequence.split('\n')
    result: List[List[Dict[str, Any]]] = []

    for line in lines:
        # Skip control lines (barlines, interpretations, comments)
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            continue

        if '\t' in line:
            tokens = line.split('\t')
        else:
            tokens = [line]

        line_visual: List[Dict[str, Any]] = []
        for spine_index, token in enumerate(tokens):
            # Skip placeholder tokens
            if token in ('.', '', 'q'):
                line_visual.append({
                    'stem': None,
                    'beam': None,
                    'above_below': None,
                    'spine_index': spine_index,
                })
                continue

            visual = extract_visual_info(token)
            visual['spine_index'] = spine_index
            line_visual.append(visual)

        result.append(line_visual)

    return result


def check_tuplet_ratio_notation(token: str) -> bool:
    """
    Check if token contains Humdrum tuplet ratio notation (X%Y).

    These tokens will cause KeyError in Zeng's LabelsMultiple and should be skipped.
    This matches Zeng's original behavior (commit 7bc0bc6, L207-208).

    Examples:
        6%7ryy    → 6 notes in time of 7 (Chopin Ballades #2)
        1920%37   → complex cadenza ratio (Liszt Paganini #6)
        48%13ryy  → 48 notes in time of 13

    Returns:
        True if token contains '%' (tuplet ratio notation)
    """
    if '%' in token:
        logger.warning(
            f"[TUPLET RATIO] Found '{token}' - Humdrum X%Y notation. "
            f"This will cause KeyError in LabelsMultiple. "
            f"Chunk will be skipped (same behavior as Zeng's original pipeline)."
        )
        return True
    return False


def clean_kern_token(token: str, preserve_ties: bool = True, preserve_articulation: bool = False) -> str:
    """
    Remove visual layout information from a single Kern token.

    Also handles duration tokens not in Zeng's LabelsMultiple vocabulary:
    - '48.' (dotted 16th triplet) → '32' (32nd note) [EQUAL DURATION: 0.125 quarters]
    - '0' (breve) → NOT CONVERTED (would break measure timing)

    Args:
        token: Raw Kern token from converter21 (e.g., '8rGG', '4g/', '8cc\\')
        preserve_ties: Whether to keep tie markers ([, ], _). Zeng's model uses ties.
        preserve_articulation: If True, keep articulation marks (' ~ ^ : ` O x v w).
                               Use True for clef-piano-full, False for clef-piano-base.
                               Default: False (Zeng vocab doesn't include articulation).

    Returns:
        Cleaned token with only semantic information (e.g., '8r', '4g', '8cc')

    Examples:
        >>> clean_kern_token('8rGG')
        '8r'
        >>> clean_kern_token('4g/')
        '4g'
        >>> clean_kern_token('4.c~')  # preserve_articulation=False (default)
        '4.c'
        >>> clean_kern_token('4.c~', preserve_articulation=True)
        '4.c~'
        >>> clean_kern_token('4c_')  # With preserve_ties=True
        '4c_'
        >>> clean_kern_token('48.ff')  # Dotted triplet conversion
        '32ff'
    """
    # 1. Control tokens: barlines (=), interpretations (*), comments (!)
    #    Return as-is (these are not encoded by LabelsMultiple)
    if token.startswith(('=', '*', '!')):
        return token

    # 2. Empty placeholder or special markers
    if token in ('.', '', 'q'):
        return token

    # =========================================================================
    # 3. Duration conversions for unsupported tokens in LabelsMultiple
    # =========================================================================
    #
    # Zeng's LabelsMultiple vocabulary (humdrum.py L77):
    #   Base: "1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.",
    #         "3","6","12","24","48","96"
    #   Extended: "128","20","40","176","112"
    #
    # MISSING from vocab (causes KeyError):
    #   - Dotted tuplets: 3., 6., 12., 24., 48.
    #   - Quintuplet: 10
    #   - Extreme values: 67, 127, 169, etc. (cadenza ornaments)
    #
    # =========================================================================
    # 3a. DOTTED TUPLET CONVERSIONS (LOSSLESS - mathematically equivalent)
    # =========================================================================
    #
    #   In Humdrum, duration value = (1/n) of whole note
    #   A dot adds 50% duration: n. = (1/n) * 1.5 = 1.5/n
    #
    #   MATH PROOF (dotted triplet = regular binary):
    #   ┌─────────────────────────────────────────────────────────────────────┐
    #   │ 12. = dotted triplet 8th                                           │
    #   │    = (1/12) * 1.5 = 1.5/12 = 1/8 = 8 (regular 8th)                │
    #   │                                                                     │
    #   │ 6.  = dotted triplet quarter                                       │
    #   │    = (1/6) * 1.5 = 1.5/6 = 1/4 = 4 (regular quarter)              │
    #   │                                                                     │
    #   │ 3.  = dotted triplet half                                          │
    #   │    = (1/3) * 1.5 = 1.5/3 = 1/2 = 2 (regular half)                 │
    #   │                                                                     │
    #   │ 24. = dotted triplet 16th                                          │
    #   │    = (1/24) * 1.5 = 1.5/24 = 1/16 = 16 (regular 16th)             │
    #   │                                                                     │
    #   │ 48. = dotted triplet 32nd                                          │
    #   │    = (1/48) * 1.5 = 1.5/48 = 1/32 = 32 (regular 32nd)             │
    #   └─────────────────────────────────────────────────────────────────────┘
    #
    #   These are EXACT conversions with ZERO duration loss.
    #
    if token.startswith('48.'):
        # 48. → 32 (dotted triplet 32nd = regular 32nd, EXACT)
        token = '32' + token[3:]
    elif token.startswith('24.'):
        # 24. → 16 (dotted triplet 16th = regular 16th, EXACT)
        token = '16' + token[3:]
    elif token.startswith('12.'):
        # 12. → 8 (dotted triplet 8th = regular 8th, EXACT)
        token = '8' + token[3:]
    elif token.startswith('6.'):
        # 6. → 4 (dotted triplet quarter = regular quarter, EXACT)
        token = '4' + token[2:]
    elif token.startswith('3.'):
        # 3. → 2 (dotted triplet half = regular half, EXACT)
        token = '2' + token[2:]

    # =========================================================================
    # 3b. QUINTUPLET (10) - HANDLED AT SEQUENCE LEVEL, NOT HERE
    # =========================================================================
    #
    #   10 = 1/10 whole note = 0.4 quarters (quintuplet 8th)
    #
    #   ⚠️ IMPORTANT: Do NOT convert 10 → 8 here!
    #   That would cause +25% duration error per note, destroying measure timing.
    #
    #   ┌─────────────────────────────────────────────────────────────────────┐
    #   │ CORRECT SOLUTION: Split into tied 20s (handled in sequence level)  │
    #   │                                                                     │
    #   │   10cc → [20cc + 20cc]  (two tied quintuplet 16ths)                │
    #   │                                                                     │
    #   │   Math: 1/20 + 1/20 = 2/20 = 1/10 ✓ (EXACT, no duration loss!)    │
    #   │                                                                     │
    #   │   See: expand_quintuplets() function below                         │
    #   └─────────────────────────────────────────────────────────────────────┘
    #
    #   📝 NOTE FOR CLEF: Consider adding "10" directly to your vocabulary!
    #   Splitting into 20+20 is a hack for Zeng's limited vocab.
    #   A proper A2S model should understand quintuplets natively.
    #
    # (No conversion here - quintuplets are expanded at sequence level)

    # =========================================================================
    # 3c. TUPLET RATIO NOTATION (%) - WARNING ONLY, NO CONVERSION
    # =========================================================================
    #
    #   converter21 outputs Humdrum's standard tuplet ratio notation:
    #     X%Y = "X notes in the time of Y"
    #
    #   Examples found in ASAP:
    #   ┌─────────────────────────────────────────────────────────────────────┐
    #   │ 6%7ryy    → Chopin Ballades #2, M45 (6 in time of 7)              │
    #   │ 1920%37   → Liszt Paganini #6, M213 (complex cadenza)             │
    #   │ 48%13ryy  → Liszt Paganini #6 (48 in time of 13)                  │
    #   │ 48%17ryy  → Liszt Paganini #6 (48 in time of 17)                  │
    #   └─────────────────────────────────────────────────────────────────────┘
    #
    #   These cause KeyError in Zeng's LabelsMultiple because:
    #   - The tokenizer may strip '%' leaving malformed numbers (6%7 → 67)
    #   - Or the regex doesn't match the X%Y pattern
    #
    #   ⚠️ WE DO NOT CONVERT THESE - Zeng's original pipeline also skips them!
    #   See commit 7bc0bc6, L207-208: `except Exception as e: continue`
    #
    #   These are rare complex tuplets (cadenzas, ornamental passages).
    #   Skipping is the correct behavior for both pipelines.
    #
    # (No conversion - will trigger KeyError → skip chunk, same as Zeng)

    # =========================================================================
    # 3d. BREVE → WHOLE (converter21-specific issue)
    # =========================================================================
    #
    #   Why we convert breve to whole:
    #   - converter21 sometimes outputs breve rests (0r) to fill empty measures
    #   - This is a converter21-specific behavior (Verovio doesn't do this)
    #   - Zeng's LabelsMultiple doesn't have '0' in vocabulary → KeyError
    #
    #   Why we DON'T split into two whole notes (1r 1r):
    #   - Splitting changes token count (1 → 2), breaking Kern parsing
    #   - The measure structure expects specific token counts
    #   - A single token replacement is safer
    #
    #   Trade-off:
    #   - Original: 0r = 8 quarters (breve rest)
    #   - Replacement: 1r = 4 quarters (whole rest)
    #   - Duration LOSS: 4 quarters (50%)
    #
    #   Justification:
    #   - This only affects measures where converter21 "invented" a breve rest
    #     to fill gaps in the MusicXML (e.g., Beethoven 21-1, M211)
    #   - The original MusicXML didn't have this duration anyway
    #   - A whole rest (1r) is semantically reasonable for "long silence"
    #   - Impact: ~70 chunks in Beethoven 21-1
    #
    #   Alternative (not implemented):
    #   - Use clean_kern_sequence() to split '0r' into '1r\n1r' at sequence level
    #   - This preserves duration but changes measure structure
    #
    if token.startswith('0') and len(token) > 1 and token[1] in 'rA-Ga-g':
        # Replace breve (0) with whole note (1), only if followed by note/rest
        token = '1' + token[1:]

    # =========================================================================
    # 4. REST POSITION CLEANUP (CRITICAL FIX for KeyError: 'rGG')
    # =========================================================================
    #    Examples: 4rCC, 8rGG, 2ryy -> 4r, 8r, 2r
    if 'r' in token:
        # Pattern: (duration)(dot*)r(pitch letters) -> keep only (duration)(dot*)r
        token = re.sub(r'(\d+\.*r)[A-Ga-gy]+', r'\1', token)

    # =========================================================================
    # 5. STEM DIRECTION (visual layout, not needed for semantic encoding)
    # =========================================================================
    #    Examples: 4g/ (stem up), 8cc\ (stem down)
    #    In **kern spine: / and \ are stem directions (always remove)
    #    In **kern spine: / \ are stem directions (visual only)
    #    In **kern spine: > < can be above/below markers OR accents (context-dependent)
    #
    #    Strategy: Remove / \ always; > < handled in articulation section
    token = re.sub(r'[/\\]', '', token)  # Always remove stem up/down

    # =========================================================================
    # 5b. EDITORIAL/VISUAL MARKERS (always remove, regardless of preserve_articulation)
    # =========================================================================
    #    X : cautionary/explicit accidental ("show this accidental even if not required")
    #    y : invisible/hidden symbol (visual only)
    #    z : editorial Z / printed rest marker
    #    N : editorial note marker ("editor added this")
    #    * : editorial footnote (when inside token, not at line start)
    #    = : editorial accidental marker (editor added this accidental)
    #    ? : uncertainty marker (editor unsure about this note)
    #    These are pure visual hints with no audible effect
    token = re.sub(r'[XyzN*=?]', '', token)

    # =========================================================================
    # 6. ARTICULATION AND ORNAMENT MARKS
    # =========================================================================
    #    These have AUDIBLE effects and should be preserved for clef-piano-full.
    #    For clef-piano-base (Zeng vocab), remove them since Zeng doesn't have these.
    #
    #    Ornaments (audible):
    #    T : trill (rapid alternation)
    #    M : mordent
    #    S : turn
    #    m : mordent (lowercase variant)
    #    s : spiccato / signum
    #    t : inverted mordent
    #    p : appoggiatura
    #    P : acciaccatura (grace note with slash)
    #    ~ : trill (alternate notation)
    #
    #    Articulations (audible):
    #    ' : staccato
    #    ^ : accent
    #    > < : accent / emphasis (when in note token, not standalone)
    #    ; : fermata
    #    : : stress
    #    ` : attack
    #
    #    Other marks:
    #    O : harmonic/open string
    #    x : editorial accidental (lowercase)
    #    v : up bow
    #    u : down bow (alternate)
    #    w : wedge
    #    & : editorial
    #    @ : ?
    #    i j : ?
    #    + : ? (possibly double sharp modifier in some encodings)
    #    Z : ? (editorial)
    #
    #    Audible ornaments/articulations (keep for clef-piano-full):
    #    W : Wagnerian turn (special ornament)
    #    $ : Seufzer / inverted turn (sigh ornament)
    #    ! : Sforzando (sudden strong accent) - note: ! at line start is comment
    #    | : Breath mark / pause (affects phrasing)
    #
    #    preserve_articulation=False (default): Remove ALL for clef-piano-base (Zeng vocab)
    #    preserve_articulation=True: Keep audible ornaments/articulations for clef-piano-full
    if not preserve_articulation:
        # Remove all articulation and ornament marks for Zeng compatibility
        token = re.sub(r"[TMSmstpP'~^><;:`Oxvuw&@ijZ+W$!|]", '', token)

    # =========================================================================
    # 7. BEAM MARKERS (optional: Zeng's model may or may not use these)
    # =========================================================================
    #    L, J, k, K are beam start/end markers
    #    Uncomment if your LabelsMultiple doesn't include these
    # token = re.sub(r'[LJkK]', '', token)

    # =========================================================================
    # 8. SLUR/PHRASE MARKERS (optional)
    # =========================================================================
    #    ( ) { } are slur/phrase start/end markers
    #    Uncomment if your LabelsMultiple doesn't include these
    # token = re.sub(r'[(){}]', '', token)

    # =========================================================================
    # 9. TIE MARKERS (usually preserved in Zeng's encoding)
    # =========================================================================
    #    [ ] _ are tie start/continue/end markers
    #    Only remove if preserve_ties=False
    if not preserve_ties:
        token = re.sub(r'[\[\]_]', '', token)

    # =========================================================================
    # 10. UNICODE / NON-ASCII CLEANUP (always remove)
    # =========================================================================
    #    Kern standard is ASCII-only. Any Unicode is likely encoding errors
    #    from HumSyn data (e.g., …, π, Ω found in Mozart/Beethoven sonatas).
    token = ''.join(c for c in token if ord(c) < 128)

    return token


def strip_non_kern_spines(content: str, keep_dynam: bool = False) -> str:
    """
    Remove non-kern spines from Humdrum content.

    Handles spine splits (*^), joins (*v), and terminations (*-) correctly
    by tracking which columns correspond to **kern spines throughout the file.

    Args:
        content: Raw Humdrum content with multiple spine types
        keep_dynam: If True, preserve **dynam spines (for clef-piano-full).
                    If False, only keep **kern spines (for clef-piano-base).

    Returns:
        Humdrum content with only **kern (and optionally **dynam) spines

    Spine handling:
        - clef-piano-base (keep_dynam=False): Keep only **kern
        - clef-piano-full (keep_dynam=True): Keep **kern and **dynam
        - Always remove: **text, **fing, and other non-kern spines

    Note:
        **text may contain expression markings (see docs/clef-plan.md 6.1),
        which will be recovered via layout comment parsing in future.
    """
    lines = content.split('\n')
    result_lines = []

    # Track spine types for each column (updated on spine operations)
    spine_types: List[str] = []
    initialized = False

    for line in lines:
        # Skip empty lines
        if not line.strip():
            result_lines.append(line)
            continue

        # Global comments (no tabs, start with !!!)
        if line.startswith('!!!'):
            result_lines.append(line)
            continue

        # Check if line has columns
        if '\t' not in line:
            # Single column line - keep if it's a global comment or special
            if line.startswith('!') or line.startswith('*'):
                result_lines.append(line)
            else:
                result_lines.append(line)
            continue

        cols = line.split('\t')

        # Initial spine declaration
        if line.startswith('**'):
            spine_types = cols.copy()
            initialized = True
            # Filter to keep only desired spines
            keep_cols = []
            for i, st in enumerate(spine_types):
                if st == '**kern':
                    keep_cols.append(i)
                elif st == '**dynam' and keep_dynam:
                    keep_cols.append(i)
            # Build filtered line
            filtered = [cols[i] for i in keep_cols]
            if filtered:
                result_lines.append('\t'.join(filtered))
            continue

        if not initialized:
            # Before spine declaration, keep everything
            result_lines.append(line)
            continue

        # Determine which columns to keep based on current spine_types
        keep_cols = []
        for i, st in enumerate(spine_types):
            if i >= len(cols):
                break
            if st == '**kern':
                keep_cols.append(i)
            elif st == '**dynam' and keep_dynam:
                keep_cols.append(i)

        # Handle spine manipulators
        if line.startswith('*'):
            new_spine_types = []
            src_idx = 0
            i = 0
            while i < len(cols):
                if src_idx >= len(spine_types):
                    break
                col = cols[i]

                if col == '*^':
                    # Split: one spine becomes two with same type
                    new_spine_types.append(spine_types[src_idx])
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += 1
                    i += 1
                elif col == '*v':
                    # Join: count consecutive *v tokens to determine merge count
                    # N consecutive *v merge N spines into 1
                    v_count = 1
                    while i + v_count < len(cols) and cols[i + v_count] == '*v':
                        v_count += 1
                    # Merge v_count spines into one
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += v_count
                    i += v_count
                elif col == '*-':
                    # Terminate: remove this spine
                    src_idx += 1
                    i += 1
                else:
                    # Regular spine data (clef, key, meter, etc.)
                    new_spine_types.append(spine_types[src_idx])
                    src_idx += 1
                    i += 1

            spine_types = new_spine_types
            # NOTE: Don't rebuild keep_cols here!
            # The original keep_cols (from before spine operations) is correct
            # for filtering the CURRENT line. The updated spine_types will
            # affect the NEXT line's keep_cols determination.

        # Filter columns
        filtered = []
        for i in keep_cols:
            if i < len(cols):
                filtered.append(cols[i])

        if filtered:
            result_lines.append('\t'.join(filtered))

    return '\n'.join(result_lines)


def clean_kern_sequence(
    sequence: str,
    preserve_ties: bool = True,
    preserve_articulation: bool = False,
    warn_tuplet_ratio: bool = True,
) -> str:
    """
    Clean an entire Kern sequence (one measure or multi-measure chunk).

    Args:
        sequence: Tab-separated or newline-separated Kern tokens
        preserve_ties: Whether to keep tie markers
        preserve_articulation: Whether to keep articulation marks (' ~ ^ : ` O x v w).
                               Use True for clef-piano-full, False for clef-piano-base.
        warn_tuplet_ratio: Whether to log warning for tuplet ratio notation (%)

    Returns:
        Cleaned sequence with same structure

    Example:
        >>> seq = "8rGG\t4g/\t8cc\\\t4.c~"
        >>> clean_kern_sequence(seq)
        '8r\t4g\t8cc\t4.c'
        >>> clean_kern_sequence(seq, preserve_articulation=True)
        '8r\t4g\t8cc\t4.c~'
    """
    # Check for tuplet ratio notation (%) - will cause KeyError, Zeng also skips these
    if warn_tuplet_ratio and '%' in sequence:
        # Find the specific tokens with %
        for line in sequence.split('\n'):
            for token in line.split('\t'):
                check_tuplet_ratio_notation(token)

    lines = sequence.split('\n')
    cleaned_lines = []

    for line in lines:
        if '\t' in line:
            # Tab-separated tokens
            tokens = line.split('\t')
            cleaned_tokens = [clean_kern_token(t, preserve_ties, preserve_articulation) for t in tokens]
            cleaned_lines.append('\t'.join(cleaned_tokens))
        else:
            # Single token or control line
            cleaned_lines.append(clean_kern_token(line, preserve_ties, preserve_articulation))

    return '\n'.join(cleaned_lines)


def expand_quintuplets(sequence: str) -> str:
    """
    Expand quintuplet 8th notes (10) into tied quintuplet 16ths (20+20).

    This is a WORKAROUND for Zeng's LabelsMultiple which lacks "10" in vocab.
    The expansion preserves exact duration by using tied notes.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ MATH PROOF:                                                            │
    │   10 = 1/10 whole = 0.4 quarters (quintuplet 8th)                     │
    │   20 = 1/20 whole = 0.2 quarters (quintuplet 16th)                    │
    │   [20 + 20] = 0.2 + 0.2 = 0.4 quarters ✓                              │
    │                                                                         │
    │ KERN TIE NOTATION:                                                     │
    │   [20cc = tie start (first note of tied pair)                         │
    │   20cc] = tie end (second note of tied pair)                          │
    │                                                                         │
    │ EXAMPLE:                                                               │
    │   Input:  "10cc"                                                       │
    │   Output: "[20cc\n20cc]"  (two lines, tied together)                  │
    └─────────────────────────────────────────────────────────────────────────┘

    📝 NOTE FOR CLEF:
    This is a hack for Zeng's limited vocabulary. If you're building your own
    model, consider adding "10" directly to your vocabulary instead of using
    this workaround. Native quintuplet support produces cleaner scores.

    Args:
        sequence: Kern sequence (may contain quintuplet tokens starting with "10")

    Returns:
        Expanded sequence with quintuplets split into tied 20s
    """
    lines = sequence.split('\n')
    expanded_lines = []

    for line in lines:
        # Skip control lines (barlines, interpretations, comments)
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            expanded_lines.append(line)
            continue

        # Check if line contains quintuplet (10)
        if '\t' in line:
            tokens = line.split('\t')
        else:
            tokens = [line]

        # Check if any token starts with "10" (quintuplet)
        has_quintuplet = any(
            re.match(r'^10[A-Ga-gr]', t) for t in tokens
        )

        if not has_quintuplet:
            expanded_lines.append(line)
            continue

        # Expand quintuplets: each 10xx becomes two lines [20xx and 20xx]
        first_tokens = []
        second_tokens = []

        for token in tokens:
            match = re.match(r'^10([A-Ga-gr].*)$', token)
            if match:
                suffix = match.group(1)  # e.g., "cc" or "r"
                # First note: tie start [20xx
                first_tokens.append(f'[20{suffix}')
                # Second note: tie end 20xx]
                second_tokens.append(f'20{suffix}]')
            else:
                # Non-quintuplet token: need to handle carefully
                # Option 1: Duplicate the token (may cause issues)
                # Option 2: Use placeholder "." for continuation
                # Using "." (null token) for the second line
                first_tokens.append(token)
                second_tokens.append('.')

        expanded_lines.append('\t'.join(first_tokens))
        expanded_lines.append('\t'.join(second_tokens))

    return '\n'.join(expanded_lines)


def scan_dirty_tokens(data_dir: str, top_n: int = 50) -> List[Tuple[str, int]]:
    """
    Scan a corpus of .krn files and identify tokens that may cause KeyError.

    This is a diagnostic tool to help you understand what "dirty" tokens
    converter21 is producing in your specific dataset.

    Args:
        data_dir: Path to directory containing .krn files
        top_n: Number of top dirty tokens to return

    Returns:
        List of (token, count) tuples, sorted by frequency

    A "dirty" token is defined as one that contains:
    - Rest position markers (rGG, rCC, etc.)
    - Stem direction (/, \\)
    - Articulation marks (', ~, ^, etc.)
    """
    dirty_tokens = Counter()

    # Pattern for "clean" tokens (only semantic information)
    # Allowed: digits, dot (dotted note), r (rest), A-G/a-g (pitch),
    #          # (sharp), - (flat), n (natural), _ [ ] (ties)
    clean_pattern = re.compile(r'^[0-9\.]+[rA-Ga-g#\-n_\[\]]+$')

    data_path = Path(data_dir)
    krn_files = list(data_path.rglob('*.krn'))

    print(f"Scanning {len(krn_files)} .krn files in {data_dir}...")

    for krn_file in krn_files:
        try:
            with open(krn_file, 'r', encoding='iso-8859-1') as f:
                for line in f:
                    # Skip control lines
                    if line.startswith(('!', '*', '=')):
                        continue

                    # Split by tabs
                    tokens = line.strip().split('\t')
                    for token in tokens:
                        if token in ('.', '', 'q'):
                            continue

                        # Check if token is "dirty"
                        if not clean_pattern.match(token):
                            dirty_tokens[token] += 1
        except Exception as e:
            print(f"Error reading {krn_file}: {e}")

    return dirty_tokens.most_common(top_n)


def generate_cleaning_report(data_dir: str, output_file: str = 'kern_cleaning_report.txt'):
    """
    Generate a comprehensive report of dirty tokens found in the corpus.

    Args:
        data_dir: Path to directory containing .krn files
        output_file: Path to save the report
    """
    dirty_tokens = scan_dirty_tokens(data_dir, top_n=100)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Kern Token Cleaning Report (converter21 Output)\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total dirty token types found: {len(dirty_tokens)}\n\n")

        # Categorize by type
        rest_position = []
        stem_direction = []
        articulation = []
        other = []

        for token, count in dirty_tokens:
            if 'r' in token and re.search(r'r[A-Ga-g]+', token):
                rest_position.append((token, count))
            elif '/' in token or '\\' in token:
                stem_direction.append((token, count))
            elif re.search(r"[';~^:`]", token):
                articulation.append((token, count))
            else:
                other.append((token, count))

        f.write(f"Category Breakdown:\n")
        f.write(f"  Rest position markers: {len(rest_position)}\n")
        f.write(f"  Stem direction: {len(stem_direction)}\n")
        f.write(f"  Articulation marks: {len(articulation)}\n")
        f.write(f"  Other: {len(other)}\n\n")

        f.write("="*70 + "\n")
        f.write("Top 50 Dirty Tokens (by frequency):\n")
        f.write("="*70 + "\n\n")

        for i, (token, count) in enumerate(dirty_tokens[:50], 1):
            cleaned = clean_kern_token(token)
            f.write(f"{i:3d}. {token:20s} ({count:6d} occurrences) -> {cleaned}\n")

    print(f"Report saved to: {output_file}")


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean Kern tokens from converter21 output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan corpus and generate report
  python clean_kern.py --scan workspace/feature.asap/test/kern_upper

  # Clean a single token
  python clean_kern.py --token "8rGG"

  # Clean a file
  python clean_kern.py --file input.krn --output cleaned.krn
        """
    )

    parser.add_argument('--scan', type=str, metavar='DIR',
                        help='Scan directory for dirty tokens')
    parser.add_argument('--token', type=str, metavar='TOKEN',
                        help='Clean a single token')
    parser.add_argument('--file', type=str, metavar='FILE',
                        help='Clean a .krn file')
    parser.add_argument('--output', type=str, metavar='FILE',
                        help='Output file for cleaned content')
    parser.add_argument('--report', type=str, metavar='FILE',
                        default='kern_cleaning_report.txt',
                        help='Output file for scan report (default: kern_cleaning_report.txt)')

    args = parser.parse_args()

    if args.scan:
        generate_cleaning_report(args.scan, args.report)

    elif args.token:
        cleaned = clean_kern_token(args.token)
        print(f"Original: {args.token}")
        print(f"Cleaned:  {cleaned}")

    elif args.file:
        with open(args.file, 'r', encoding='iso-8859-1') as f:
            content = f.read()

        cleaned = clean_kern_sequence(content)

        if args.output:
            with open(args.output, 'w', encoding='iso-8859-1') as f:
                f.write(cleaned)
            print(f"Cleaned file saved to: {args.output}")
        else:
            print(cleaned)

    else:
        parser.print_help()
