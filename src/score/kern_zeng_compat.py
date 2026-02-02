"""
Zeng Vocab Compatibility Transformations
========================================

This module contains transformations to make kern sequences compatible with
Zeng's LabelsMultiple vocabulary. These are WORKAROUNDS for clef-piano-base only.

For clef-piano-full, the vocabulary should be extended instead of using these hacks.

Transformations included:
- Duration conversions (tuplets, dotted triplets)
- Accidental normalization (strip n, convert ## and --)
- Beam removal (L, J, K, k)
- Extreme pitch → clamp to piano range

Design Philosophy (from performance practice):
    Performers often "feel" n-tuplets as (n+1) notes with the last one held.
    For example, a 7-tuplet is often executed as 8 notes with the last slightly extended.

    This matches how many musicians actually practice complex tuplets:
    - Count in a familiar subdivision
    - Adjust the last note to fit the beat

    By expanding n-tuplet → (n+1) tied notes, we:
    1. Preserve exact total duration (mathematically proven)
    2. Match musical intuition
    3. Use only standard Zeng vocabulary tokens
"""

import re
from typing import Dict, List, Tuple, Optional, Set

# =============================================================================
# Zeng Vocabulary Reference
# =============================================================================
# Binary + Dotted: 1, 1., 2, 2., 4, 4., 8, 8., 16, 16., 32, 32., 64, 64.
# Triplet: 3, 6, 12, 24, 48, 96
# Extended: 20, 40, 112, 128, 176
#   NOTE: 20, 40 are quintuplet durations that converter21 CAN parse
# Breve: 0, 0., 00, 00.

ZENG_DURATIONS = {
    "1", "1.", "2", "2.", "4", "4.", "8", "8.", "16", "16.", "32", "32.", "64", "64.",
    "3", "6", "12", "24", "48", "96",
    "128", "20", "40", "176", "112",
    "00", "00.", "0", "0.",
}


# =============================================================================
# Tuplet Expansion Mappings
# =============================================================================
# Format: {original_duration: (target_duration, tuplet_size)}
# tuplet_size = how many notes form one group (triggers extra tied note)

# Quintuplet (5 in time of 4): 5 notes → 6 triplet notes
# Math: 5 × (1/10) = 6 × (1/12) = 1/2 whole ✓
QUINTUPLET_MAP = {
    '5': ('6', 5),      # quintuplet whole → triplet whole
    '10': ('12', 5),    # quintuplet half → triplet half
    # 20, 40 are already in Zeng vocab
    '80': ('96', 5),    # quintuplet 16th → triplet 16th
    '160': ('96', 5),   # quintuplet 32nd → cap at 96 (lossy but rare)
}

# Septuplet (7 in time of 4): 7 notes → 8 binary notes
# Math: 7 × (1/28) = 8 × (1/32) = 1/4 whole ✓
SEPTUPLET_MAP = {
    '7': ('8', 7),      # septuplet whole → whole
    '14': ('16', 7),    # septuplet half → half
    '28': ('32', 7),    # septuplet quarter → quarter
    '56': ('64', 7),    # septuplet 8th → 8th
    # 112 is already in Zeng vocab
}


# 9-tuplet (9 in time of 10 quintuplet): 9 notes → 10 quintuplet notes
# Math: 9 × (1/18) = 10 × (1/20) = 1/2 whole ✓
# Math: 9 × (1/36) = 10 × (1/40) = 1/4 whole ✓
# This avoids mixed 16/20 output from IP solver which breaks converter21
# (converter21 infers Tuplet 5/4 from 20s, then 16 doesn't fit the tuplet)
NINE_TUPLET_MAP = {
    '18': ('20', 9),    # 9-tuplet half → quintuplet quarter
    '36': ('40', 9),    # 9-tuplet quarter → quintuplet eighth
}

# 11-tuplet (11 in time of 8): 11 notes → 12 triplet notes
# Math: 11 × (1/44) ≈ 12 × (1/48) = 1/4 whole ✓
ELEVEN_TUPLET_MAP = {
    '11': ('12', 11),   # 11-tuplet whole → triplet whole
    '22': ('24', 11),   # 11-tuplet half → triplet half
    '44': ('48', 11),   # 11-tuplet quarter → triplet quarter
    '88': ('96', 11),   # 11-tuplet 8th → triplet 8th
}

# 15-tuplet (15 in time of 16): 15 notes → 16 binary notes
# Math: 15 × (1/60) = 16 × (1/64) = 1/4 whole ✓
FIFTEEN_TUPLET_MAP = {
    '15': ('16', 15),
    '30': ('32', 15),
    '60': ('64', 15),
}

# Dotted triplet → binary conversion (EXACT, no tie needed)
# Math: triplet × 1.5 = binary (e.g., 1/12 × 1.5 = 1/8)
DOTTED_TRIPLET_TO_BINARY = {
    '3.': '2',      # dotted triplet half = half
    '6.': '4',      # dotted triplet quarter = quarter
    '12.': '8',     # dotted triplet eighth = eighth
    '24.': '16',    # dotted triplet sixteenth = sixteenth
    '48.': '32',    # dotted triplet 32nd = 32nd
    '96.': '64',    # dotted triplet 64th = 64th
}

# Dotted tuplet expansion: n. → [n + n*2] (tied)
# Math: n. = n × 1.5 = n + n/2, and n/2 in Humdrum is n*2
# Example: 10. = 1/10 × 1.5 = 1/10 + 1/20 = [10 + 20] tied
DOTTED_TUPLET_MAP = {
    '5.': ('5', '10'),      # 5. → [5 + 10] → after tuplet conv → [6 + 12]
    '10.': ('10', '20'),    # 10. → [10 + 20] → [12 + 20]
    '14.': ('14', '28'),    # 14. → [14 + 28] → [16 + 32]
    '18.': ('18', '36'),    # 18. → [18 + 36] → [16 + 32]
    '20.': ('20', '40'),    # 20. → [20 + 40] (both in vocab)
}

# Combine all expansion maps (n → n+1 strategy)
EXPAND_MAPS = {
    **SEPTUPLET_MAP,      # 7 → 8
    **QUINTUPLET_MAP,     # 5 → 6
    **NINE_TUPLET_MAP,    # 9 → 10
    **ELEVEN_TUPLET_MAP,  # 11 → 12
    **FIFTEEN_TUPLET_MAP, # 15 → 16
}

# MERGE_MAPS removed - now handled by rhythm_quantizer with exact duration matching
# Old merge strategy lost notes (pitch information), IP preserves all pitches
MERGE_MAPS = {}  # Empty - all handled by IP now


def convert_dotted_triplets(sequence: str) -> str:
    """
    Convert dotted triplet durations to their exact binary equivalents.

    Math: triplet × 1.5 = binary
        3.  = 1/3 × 1.5 = 1/2 = 2
        6.  = 1/6 × 1.5 = 1/4 = 4
        12. = 1/12 × 1.5 = 1/8 = 8
        etc.

    This is an EXACT conversion with zero timing error.

    Args:
        sequence: Kern sequence with dotted triplet tokens

    Returns:
        Sequence with dotted triplets converted to binary durations
    """
    for dotted_triplet, binary in DOTTED_TRIPLET_TO_BINARY.items():
        # Match duration followed by pitch letter or rest
        # Use negative lookbehind (?<!\d) to prevent matching partial durations
        # e.g., prevent '6.' from matching the '6.r' part of '16.r'
        pattern = rf'(?<!\d)({re.escape(dotted_triplet)})([A-Ga-gr\[\]_])'
        replacement = rf'{binary}\2'
        sequence = re.sub(pattern, replacement, sequence)
    return sequence


def expand_dotted_tuplets(sequence: str) -> str:
    """
    First pass: Expand dotted tuplets to tied note pairs.

    n. → [n + n*2] (tied)

    Math: n. = n × 1.5 = n + n/2, and n/2 in Humdrum notation is n*2
    Example: 10. = 1/10 × 1.5 = 1/10 + 1/20 = [10c + 20c]

    The resulting n and n*2 will be converted by the second pass (tuplet expansion).

    Args:
        sequence: Kern sequence with dotted tuplet tokens

    Returns:
        Sequence with dotted tuplets expanded to tied pairs
    """
    lines = sequence.split('\n')
    result_lines = []

    for line in lines:
        # Skip control lines
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            result_lines.append(line)
            continue

        tokens = line.split('\t') if '\t' in line else [line]
        converted_tokens = []
        extra_line_tokens = []
        need_extra_line = False

        for token in tokens:
            # Check for dotted tuplets
            dotted_match = None
            for dotted_dur, (first_dur, second_dur) in DOTTED_TUPLET_MAP.items():
                if re.match(rf'^{re.escape(dotted_dur)}[A-Ga-gr\[\]_]', token):
                    dotted_match = (dotted_dur, first_dur, second_dur)
                    break

            if dotted_match:
                dotted_dur, first_dur, second_dur = dotted_match
                pitch_part = re.sub(rf'^{re.escape(dotted_dur)}', '', token)
                # Create tied pair
                converted_tokens.append(f'[{first_dur}{pitch_part}')
                extra_line_tokens.append(f'{second_dur}{pitch_part}]')
                need_extra_line = True
            else:
                converted_tokens.append(token)
                extra_line_tokens.append('.')

        result_lines.append('\t'.join(converted_tokens) if len(converted_tokens) > 1 else converted_tokens[0])
        if need_extra_line:
            result_lines.append('\t'.join(extra_line_tokens) if len(extra_line_tokens) > 1 else extra_line_tokens[0])

    return '\n'.join(result_lines)


def _scan_complete_tuplet_groups(lines: List[str]) -> Set[Tuple[int, int]]:
    """
    Pass 1: Scan for COMPLETE tuplet groups only.

    Complete group = N consecutive tuplet notes of same duration in same spine
    - Quintuplet: 5 consecutive (5, 10, 80, 160)
    - Septuplet: 7 consecutive (7, 14, 28, 56)
    - 11-tuplet: 11 consecutive (11, 22, 44, 88)
    - 15-tuplet: 15 consecutive (15, 30, 60)

    Incomplete groups (< N notes) are NOT marked for conversion.
    They will be preserved as-is for quantize_oov_tuplets() to handle.

    Returns:
        Set of (line_idx, spine_idx) for notes belonging to complete groups
    """
    complete_positions: Set[Tuple[int, int]] = set()
    spine_counts: Dict[int, Dict[str, List[int]]] = {}  # spine -> {dur -> [line_indices]}

    for line_idx, line in enumerate(lines):
        # Reset at barlines
        if line.startswith('='):
            spine_counts = {}
            continue

        # Skip other control lines
        if line.startswith(('*', '!')) or line.strip() == '':
            continue

        tokens = line.split('\t') if '\t' in line else [line]

        for spine_idx, token in enumerate(tokens):
            # Initialize
            if spine_idx not in spine_counts:
                spine_counts[spine_idx] = {}

            # Extract prefix markers
            working_token = token
            while working_token and working_token[0] in '[({':
                working_token = working_token[1:]

            # Check if tuplet note
            matched_dur = None
            matched_group_size = None

            for dur, (target, group_size) in EXPAND_MAPS.items():
                if re.match(rf'^{dur}[A-Ga-grnqQP\[\]_]', working_token):
                    matched_dur = dur
                    matched_group_size = group_size
                    break

            if matched_dur:
                # Add to spine's tuplet group
                if matched_dur not in spine_counts[spine_idx]:
                    spine_counts[spine_idx][matched_dur] = []

                spine_counts[spine_idx][matched_dur].append(line_idx)

                # Check if completed a group
                if len(spine_counts[spine_idx][matched_dur]) == matched_group_size:
                    # Mark all positions as complete
                    for idx in spine_counts[spine_idx][matched_dur]:
                        complete_positions.add((idx, spine_idx))

                    # Reset for next group
                    spine_counts[spine_idx][matched_dur] = []

            elif token != '.':
                # Non-tuplet, non-null token - reset this spine
                spine_counts[spine_idx] = {}

    return complete_positions


def expand_tuplets_to_zeng_vocab(sequence: str) -> str:
    """
    Expand tuplet notes to match Zeng's vocabulary (TWO-PASS STRATEGY).

    Pass 1: Scan and identify COMPLETE tuplet groups (N consecutive notes)
    Pass 2: Convert only complete groups, preserve incomplete groups

    This ensures timing correctness: incomplete groups won't be partially
    converted (which would break timing). They're preserved for
    quantize_oov_tuplets() to handle with IP solver.

    Complete group requirements:
    - Quintuplet: exactly 5 consecutive notes → 6 notes (5+1 tied)
    - Septuplet: exactly 7 consecutive notes → 8 notes (7+1 tied)
    - 11-tuplet: exactly 11 consecutive notes → 12 notes (11+1 tied)
    - 15-tuplet: exactly 15 consecutive notes → 16 notes (15+1 tied)

    Args:
        sequence: Kern sequence with tuplet tokens

    Returns:
        Transformed sequence with complete tuplet groups expanded
    """
    # Pre-processing: dotted tuplets
    sequence = convert_dotted_triplets(sequence)
    sequence = expand_dotted_tuplets(sequence)

    lines = sequence.split('\n')

    # Pass 1: Identify complete groups
    complete_positions = _scan_complete_tuplet_groups(lines)

    # Pass 2: Convert only complete groups
    result_lines = []
    expand_counts: Dict[int, Dict[str, int]] = {}  # spine -> {dur -> count}

    for line_idx, line in enumerate(lines):
        # Control lines
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            result_lines.append(line)
            if line.startswith('='):
                expand_counts = {}
            continue

        tokens = line.split('\t') if '\t' in line else [line]
        converted_tokens = []
        extra_line_tokens = []
        need_extra_line = False

        for spine_idx, token in enumerate(tokens):
            # Initialize
            if spine_idx not in expand_counts:
                expand_counts[spine_idx] = {}

            # Check if this position is in a complete group
            if (line_idx, spine_idx) not in complete_positions:
                # Not in complete group - keep as-is (preserve incomplete groups)
                converted_tokens.append(token)
                extra_line_tokens.append('.')
                continue

            # This is part of a complete group - convert it
            # Extract prefix markers
            prefix_markers = ''
            working_token = token
            while working_token and working_token[0] in '[({':
                prefix_markers += working_token[0]
                working_token = working_token[1:]

            # Find matching tuplet
            matched = None
            for dur, (target, group_size) in EXPAND_MAPS.items():
                if re.match(rf'^{dur}[A-Ga-grnqQP\[\]_]', working_token):
                    matched = (dur, target, group_size)
                    break

            if matched:
                dur, target, group_size = matched
                # Replace duration in all chord notes (e.g., "36ee 36eee" → "40ee 40eee")
                # \b prevents partial matches (e.g., won't match 136)
                converted = re.sub(rf'\b{dur}(?=[A-Ga-grnqQP\[\]_])', target, working_token)
                converted = prefix_markers + converted

                # Track count
                if dur not in expand_counts[spine_idx]:
                    expand_counts[spine_idx][dur] = 0
                expand_counts[spine_idx][dur] += 1

                # Add tie for group_size-th note
                if expand_counts[spine_idx][dur] == group_size:
                    need_extra_line = True
                    if '[' not in converted:
                        converted = '[' + converted
                    converted_tokens.append(converted)

                    # Create tie end (also handle chords)
                    extra_converted = re.sub(rf'\b{dur}(?=[A-Ga-grnqQP\[\]_])', target, working_token)
                    extra_line_tokens.append(extra_converted + ']')
                    expand_counts[spine_idx][dur] = 0
                else:
                    converted_tokens.append(converted)
                    extra_line_tokens.append('.')
            else:
                # Should not happen (position marked but no match)
                converted_tokens.append(token)
                extra_line_tokens.append('.')

        # Output
        result_lines.append('\t'.join(converted_tokens) if len(converted_tokens) > 1 else converted_tokens[0])

        if need_extra_line:
            result_lines.append('\t'.join(extra_line_tokens) if len(extra_line_tokens) > 1 else extra_line_tokens[0])

    return '\n'.join(result_lines)


def get_oov_durations(sequence: str) -> List[Tuple[str, int]]:
    """
    Identify OOV duration tokens in a kern sequence.

    Args:
        sequence: Kern sequence

    Returns:
        List of (duration, count) tuples for OOV durations
    """
    from collections import Counter
    oov_counter = Counter()

    for line in sequence.split('\n'):
        if line.startswith(('=', '*', '!')) or not line.strip():
            continue

        tokens = line.split('\t') if '\t' in line else [line]
        for token in tokens:
            if token == '.' or not token:
                continue

            # Skip tuplet ratio notation (X%Y) - these are counted separately
            if re.search(r'\d+%\d+', token):
                continue

            # Skip leading markers: tie ([), slur ((), phrase ({)
            match = re.match(r'^[\[({]*(\d+\.?)', token)
            if match:
                dur = match.group(1)
                if dur not in ZENG_DURATIONS:
                    oov_counter[dur] += 1

    return sorted(oov_counter.items(), key=lambda x: -x[1])


# =============================================================================
# OOV Tuplet Quantization (using rhythm_quantizer lookup table)
# =============================================================================

# Known OOV durations that need tuplet-aware quantization
# Includes:
# - Cadenza tuplets: 42, 21, 66, 58, 82, 62, 54, 46
# - 9-tuplet: 9, 18, 36, 72 (moved from MERGE_MAPS)
# - 13-tuplet: 13, 26, 52, 104 (moved from MERGE_MAPS)
# - 17-tuplet: 17, 34, 68 (moved from MERGE_MAPS)
# - 19-tuplet: 19, 38, 76 (moved from MERGE_MAPS)
# - Stray OOV: 5, 7, 10, 11, 14, 22, 23, 28 (isolated notes not in full tuplet groups)
OOV_TUPLET_DURATIONS = {
    # Cadenza tuplets
    '42', '21', '66', '58', '82', '62', '54', '46',
    # 9-tuplet (9 in time of 8)
    '9', '18', '36', '72',
    # 13-tuplet (13 in time of 8/12)
    '13', '26', '52', '104',
    # 17-tuplet (17 in time of 16)
    '17', '34', '68',
    # 19-tuplet (19 in time of 16)
    '19', '38', '76',
    # Stray OOV (isolated notes, not full tuplet groups)
    '5', '7', '10', '11', '14', '22', '23', '28',
}


def quantize_oov_tuplets(sequence: str, seed: int = None) -> str:
    """
    Quantize OOV tuplet durations using Integer Programming lookup table.

    This identifies consecutive groups of OOV durations and replaces them
    with optimal mixtures of standard vocab durations.

    Algorithm:
        1. Scan for consecutive OOV duration groups (same duration in a row)
        2. Look up optimal (long, short, count_long, count_short) from table
        3. Distribute randomly (or evenly) and replace

    Args:
        sequence: Kern sequence with OOV tuplets
        seed: Random seed for reproducible distribution (None = random)

    Returns:
        Transformed sequence with OOV durations replaced
    """
    import random

    # Import from rhythm_quantizer (avoid circular import via __init__)
    try:
        from .rhythm_quantizer import (
            get_tuplet_solution,
            get_quantized_duration,
            distribute_tuplet_random,
        )
    except ImportError:
        # Fallback for direct script execution - use importlib to avoid __init__.py
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            'rhythm_quantizer',
            __file__.replace('kern_zeng_compat.py', 'rhythm_quantizer.py')
        )
        _rq = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_rq)
        get_tuplet_solution = _rq.get_tuplet_solution
        get_quantized_duration = _rq.get_quantized_duration
        distribute_tuplet_random = _rq.distribute_tuplet_random

    if seed is not None:
        random.seed(seed)

    lines = sequence.split('\n')
    result_lines = []

    # Track OOV groups per spine
    # Each spine may have independent OOV groups
    # Format: {spine_idx: {'dur': oov_dur, 'tokens': [(line_idx, token)], 'positions': []}}
    oov_groups: Dict[int, Dict] = {}

    def flush_oov_group(spine_idx: int, pending_results: Dict[int, List[str]]):
        """Process and replace an OOV group for a spine."""
        if spine_idx not in oov_groups:
            return

        group = oov_groups[spine_idx]
        oov_dur = group['dur']
        tokens = group['tokens']
        n = len(tokens)

        if n == 0:
            del oov_groups[spine_idx]
            return

        # Look up solution
        solution = get_tuplet_solution(oov_dur, n)

        if solution:
            long_dur, short_dur, cnt_long, cnt_short = solution
            # Distribute
            distribution = distribute_tuplet_random(long_dur, short_dur, cnt_long, cnt_short)
        else:
            # No exact solution found - keep original duration
            # These will become <unk> in tokenizer (0 error policy)
            # Note: We do NOT use approximate mappings
            distribution = [oov_dur] * n

        # Apply distribution to tokens
        for i, (line_idx, original_token) in enumerate(tokens):
            new_dur = distribution[i] if i < len(distribution) else distribution[-1]
            # Replace duration in token (use lambda to avoid backreference issues)
            new_token = re.sub(
                r'^([\[({]*)' + oov_dur,
                lambda m: m.group(1) + new_dur,
                original_token
            )

            if line_idx not in pending_results:
                pending_results[line_idx] = {}
            pending_results[line_idx][spine_idx] = new_token

        del oov_groups[spine_idx]

    # First pass: identify OOV groups
    line_data = []  # [(line_idx, is_data, tokens)]

    for i, line in enumerate(lines):
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            line_data.append((i, False, line))
        else:
            tokens = line.split('\t') if '\t' in line else [line]
            line_data.append((i, True, tokens))

    # Process lines and track OOV groups
    pending_results: Dict[int, Dict[int, str]] = {}  # {line_idx: {spine_idx: new_token}}

    for line_idx, is_data, content in line_data:
        if not is_data:
            # Control line - flush all OOV groups at barlines
            if isinstance(content, str) and content.startswith('='):
                for spine_idx in list(oov_groups.keys()):
                    flush_oov_group(spine_idx, pending_results)
            continue

        tokens = content
        for spine_idx, token in enumerate(tokens):
            if token == '.' or not token:
                continue

            # Extract duration
            match = re.match(r'^[\[({]*(\d+)', token)
            if not match:
                continue

            dur = match.group(1)

            # Check if this is an OOV tuplet duration
            if dur in OOV_TUPLET_DURATIONS:
                # Check if continuing existing group or starting new one
                if spine_idx in oov_groups and oov_groups[spine_idx]['dur'] == dur:
                    # Continue group
                    oov_groups[spine_idx]['tokens'].append((line_idx, token))
                else:
                    # Flush old group if exists
                    if spine_idx in oov_groups:
                        flush_oov_group(spine_idx, pending_results)
                    # Start new group
                    oov_groups[spine_idx] = {'dur': dur, 'tokens': [(line_idx, token)]}
            else:
                # Not OOV - flush any pending group for this spine
                if spine_idx in oov_groups:
                    flush_oov_group(spine_idx, pending_results)

    # Flush remaining groups
    for spine_idx in list(oov_groups.keys()):
        flush_oov_group(spine_idx, pending_results)

    # Second pass: apply replacements
    for line_idx, is_data, content in line_data:
        if not is_data:
            result_lines.append(content)
        else:
            tokens = list(content)
            if line_idx in pending_results:
                for spine_idx, new_token in pending_results[line_idx].items():
                    if spine_idx < len(tokens):
                        tokens[spine_idx] = new_token
            result_lines.append('\t'.join(tokens))

    return '\n'.join(result_lines)


# =============================================================================
# Pitch/Accidental Compatibility (Zeng vocab: only # and -)
# =============================================================================

def strip_natural_accidentals(token: str) -> str:
    """
    Strip standalone natural signs from a token.

    Zeng's regex only allows # and - as accidentals:
        r'([a-gA-Gr]{1,4}[\\-#]*)'

    Natural signs (n) are implicit in standard notation.

    Examples:
        ffn → ff (F-natural = F)
        een → ee (E-natural = E)
        CCn → CC (Contra C-natural = C)

    Note: This should be called AFTER n#/n-/nn normalization in clean_kern.py
    """
    # Remove 'n' that appears after pitch letters (A-G or a-g)
    # but before tie markers, accidentals, or end of token
    return re.sub(r'([A-Ga-g])n(?=[^A-Ga-g]|$)', r'\1', token)


# Enharmonic conversion tables for double accidentals
# Double sharp: pitch## → next semitone (natural or sharp)
DOUBLE_SHARP_MAP = {
    # Uppercase (Great octave and below)
    'C': 'D', 'D': 'E', 'E': 'F#', 'F': 'G', 'G': 'A', 'A': 'B', 'B': 'C#',
    # Lowercase (Small octave and above)
    'c': 'd', 'd': 'e', 'e': 'f#', 'f': 'g', 'g': 'a', 'a': 'b', 'b': 'c#',
}

# Double flat: pitch-- → prev semitone (natural or flat)
DOUBLE_FLAT_MAP = {
    # Uppercase
    'C': 'B-', 'D': 'C', 'E': 'D', 'F': 'E-', 'G': 'F', 'A': 'G', 'B': 'A',
    # Lowercase
    'c': 'b-', 'd': 'c', 'e': 'd', 'f': 'e-', 'g': 'f', 'a': 'g', 'b': 'a',
}


def convert_double_accidentals(token: str) -> str:
    """
    Convert double sharps (##) and double flats (--) to enharmonic equivalents.

    Zeng's vocab only has single # and - accidentals.

    Examples:
        4f## → 4g (F double-sharp = G)
        4FF## → 4GG (Great F double-sharp = Great G)
        8b-- → 8a (B double-flat = A)
        8BB-- → 8AA (Great B double-flat = Great A)

    Note: Handles octave notation (repeated letters like CC, ccc)
    """
    # Match: optional prefix + duration + pitch letters + ## or --
    # Pattern: (prefix)(duration)(pitch_letters)(##|--)

    def replace_double_sharp_pitch(pitch_letters: str) -> str:
        """Convert pitch## to enharmonic equivalent (just the pitch part)."""
        # Get the base pitch (first letter, preserving case)
        base = pitch_letters[0]
        octave_count = len(pitch_letters)

        # Look up enharmonic
        if base.upper() in DOUBLE_SHARP_MAP:
            # Get replacement, preserving case
            if base.isupper():
                replacement = DOUBLE_SHARP_MAP[base.upper()]
            else:
                replacement = DOUBLE_SHARP_MAP[base].lower() if base in DOUBLE_SHARP_MAP else DOUBLE_SHARP_MAP[base.upper()].lower()

            # Handle octave: replicate the new pitch letter
            new_base = replacement[0]
            new_accidental = replacement[1:] if len(replacement) > 1 else ''
            new_pitch = new_base * octave_count + new_accidental
            return new_pitch
        return pitch_letters + '##'  # Unchanged

    def replace_double_flat_pitch(pitch_letters: str) -> str:
        """Convert pitch-- to enharmonic equivalent (just the pitch part)."""
        base = pitch_letters[0]
        octave_count = len(pitch_letters)

        if base.upper() in DOUBLE_FLAT_MAP:
            if base.isupper():
                replacement = DOUBLE_FLAT_MAP[base.upper()]
            else:
                replacement = DOUBLE_FLAT_MAP[base].lower() if base in DOUBLE_FLAT_MAP else DOUBLE_FLAT_MAP[base.upper()].lower()

            new_base = replacement[0]
            new_accidental = replacement[1:] if len(replacement) > 1 else ''
            new_pitch = new_base * octave_count + new_accidental
            return new_pitch
        return pitch_letters + '--'  # Unchanged

    # Double sharp: pitch## → enharmonic
    # Simplified approach: match pitch letters followed by ## anywhere in the token
    # This handles: 4f##, 4..ff##, <unk>FF##, [BBB##, 16qqf##, etc.
    token = re.sub(
        r'([A-Ga-g]+)##',
        lambda m: replace_double_sharp_pitch(m.group(1)),
        token
    )

    # Double flat: pitch-- → enharmonic
    # Same simplified approach
    token = re.sub(
        r'([A-Ga-g]+)--',
        lambda m: replace_double_flat_pitch(m.group(1)),
        token
    )

    return token


def remove_slur_phrase_markers(token: str) -> str:
    """
    Remove slur and phrase markers from a token.

    Zeng's vocab doesn't include these:
        ( ) = slur start/end (legato phrasing)
        { } = phrase start/end (musical phrase boundaries)

    For clef-piano-full, these should be preserved as they affect performance.
    """
    return re.sub(r'[(){}]', '', token)


# =============================================================================
# Breve Splitting (preserve duration)
# =============================================================================

# Breve duration mappings: original → (split_duration, count)
# 0 = breve = 2 whole notes
# 00 = longa = 4 whole notes
BREVE_SPLIT_MAP = {
    '0': ('1', 2),      # breve → 2 tied whole notes
    '0.': ('1.', 2),    # dotted breve → 2 tied dotted wholes (approximation)
    '00': ('1', 4),     # longa → 4 tied whole notes
    '00.': ('1.', 4),   # dotted longa → 4 tied dotted wholes (approximation)
}


def split_breve_token(token: str) -> Optional[List[str]]:
    """
    Split a breve/longa token into tied whole notes (or multiple rests).

    Args:
        token: A kern token that may start with breve duration (0 or 00)

    Returns:
        List of split tokens if breve found, None otherwise

    Examples:
        0c → ['[1c', '1c]']  (tied notes)
        0r → ['1r', '1r']    (two rests, no ties needed)
        00CC → ['[1CC', '1CC_', '1CC_', '1CC]']
        4c → None (not a breve)
    """
    # Check for breve durations (check longer patterns first)
    for breve_dur in ['00.', '00', '0.', '0']:
        if breve_dur not in BREVE_SPLIT_MAP:
            continue
        split_dur, count = BREVE_SPLIT_MAP[breve_dur]

        if token.startswith(breve_dur):
            # Extract pitch part (everything after duration)
            pitch_part = token[len(breve_dur):]

            # Handle rests: just repeat, no ties needed
            if pitch_part.startswith('r'):
                # Strip any position markers from rest (e.g., 0rGG → 1r)
                return [f'{split_dur}r'] * count

            # Create tied sequence for notes
            result = []
            for i in range(count):
                if i == 0:
                    # First note: tie start
                    result.append(f'[{split_dur}{pitch_part}')
                elif i == count - 1:
                    # Last note: tie end
                    result.append(f'{split_dur}{pitch_part}]')
                else:
                    # Middle notes: tie continue
                    result.append(f'{split_dur}{pitch_part}_')
            return result

    return None


def split_breves_in_sequence(sequence: str) -> str:
    """
    Split all breve/longa notes in a sequence into tied whole notes.

    This preserves the total duration while using only vocab-compatible durations.

    Args:
        sequence: Kern sequence with possible breve tokens

    Returns:
        Sequence with breves split into tied whole notes
    """
    lines = sequence.split('\n')
    result_lines = []

    for line in lines:
        # Skip control lines
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            result_lines.append(line)
            continue

        tokens = line.split('\t') if '\t' in line else [line]

        # Check if any token needs splitting
        splits_needed = []
        max_splits = 1

        for i, token in enumerate(tokens):
            split_result = split_breve_token(token)
            if split_result:
                splits_needed.append((i, split_result))
                max_splits = max(max_splits, len(split_result))
            else:
                splits_needed.append((i, None))

        if max_splits == 1:
            # No breves found
            result_lines.append(line)
        else:
            # Need to expand to multiple lines
            for line_idx in range(max_splits):
                new_tokens = []
                for token_idx, (_, split_result) in enumerate(splits_needed):
                    if split_result and line_idx < len(split_result):
                        new_tokens.append(split_result[line_idx])
                    elif split_result:
                        # Breve split exhausted, use placeholder
                        new_tokens.append('.')
                    else:
                        # Non-breve token
                        if line_idx == 0:
                            new_tokens.append(tokens[token_idx])
                        else:
                            new_tokens.append('.')

                result_lines.append('\t'.join(new_tokens))

    return '\n'.join(result_lines)


# =============================================================================
# Extreme Pitch Handling
# =============================================================================

# Zeng's regex allows 1-4 pitch letters: [a-gA-Gr]{1,4}
# Valid range: AAA (sub-contra) to bbbb (five-line)
# Piano range: AAA to cccc (88 keys, A0 to C8)
# Extreme pitches outside this range → clamp to boundary

EXTREME_LOW_PITCHES = {'AAAA', 'BBBB', 'CCCC', 'DDDD', 'EEEE', 'FFFF', 'GGGG'}
EXTREME_HIGH_PITCHES = {'aaaaa', 'bbbbb', 'ccccc', 'ddddd', 'eeeee', 'fffff', 'ggggg'}


def clamp_extreme_pitches(token: str) -> str:
    """
    Clamp extreme pitches to piano range boundaries.

    Piano range: A0 (AAA) to C8 (cccc)
    Extreme low: AAAA and below → clamp to 3 letters (e.g., AAAA- → AAA-)
    Extreme high: ccccc and above → clamp to 4 letters (e.g., ccccc → cccc)

    These are typically encoding errors from MuseSyn user uploads.
    Clamping is acoustically correct: the mel spectrogram (f_min=27.5 Hz)
    cannot resolve fundamentals below A0, so the model sees the harmonics
    which overlap with the clamped pitch region.

    Args:
        token: Kern token

    Returns:
        Token with pitch clamped to piano range
    """
    # Extract pitch letters from token
    match = re.search(r'([A-Ga-g]{4,})', token)
    if match:
        pitch_letters = match.group(1)
        base = pitch_letters[0]

        # All same letter repeated (e.g., AAAA, ccccc)
        if not all(c == base for c in pitch_letters):
            return token

        if pitch_letters.isupper() and len(pitch_letters) >= 4:
            # Extreme low: AAAA → AAA, BBBBB → BBB, etc.
            # Clamp to 3 letters (sub-contra octave = piano lowest)
            clamped = base * 3
            return token.replace(pitch_letters, clamped)

        if pitch_letters.islower() and len(pitch_letters) >= 5:
            # Extreme high: ccccc → cccc, etc.
            # Clamp to 4 letters (five-line octave = piano highest)
            clamped = base * 4
            return token.replace(pitch_letters, clamped)

    return token


# =============================================================================
# Resolve *rscale regions (must run BEFORE expand_tuplets / quantize_oov)
# =============================================================================

def _divide_token_duration(token: str, factor: int) -> str:
    """Divide the recip value(s) in a kern token by factor.

    Handles chords (space-separated notes in one token).
    """
    if not token or token == '.':
        return token

    def _div(m):
        prefix = m.group(1)
        dur_str = m.group(2)
        dotted = dur_str.endswith('.')
        dur_val = int(dur_str.rstrip('.'))
        new_dur = dur_val // factor
        if new_dur < 1:
            new_dur = 1
        return prefix + str(new_dur) + ('.' if dotted else '')

    if ' ' in token:
        return ' '.join(
            re.sub(r'^([\[({]*)(\d+\.?)', _div, p) for p in token.split(' ')
        )
    return re.sub(r'^([\[({]*)(\d+\.?)', _div, token)


def _parse_token_duration_frac(token: str) -> 'Fraction':
    """Parse kern data token duration as Fraction of quarter notes.

    Returns Fraction(0) for null tokens, grace notes, or unparseable tokens.
    """
    from fractions import Fraction
    if not token or token == '.':
        return Fraction(0)
    clean = token.lstrip('[({')
    if 'q' in clean.lower():
        return Fraction(0)
    first = clean.split()[0] if ' ' in clean else clean
    m = re.search(r'(\d+)', first)
    if not m:
        return Fraction(0)
    recip = int(m.group(1))
    if recip == 0:
        return Fraction(8)
    dur = Fraction(4, recip)
    dots = first.count('.')
    if dots > 0:
        dur = dur * (Fraction(2) - Fraction(1, 2 ** dots))
    return dur


def resolve_rscale_regions(sequence: str) -> str:
    """
    Resolve *rscale:N regions by dividing note durations by N, stripping
    *rscale markers, and reflowing line structure.

    In Humdrum, *rscale:N means "displayed durations are 1/N of actual
    durations". After resolution, each note's recip is divided by N so
    the kern value equals the actual performed duration.

    The line reflow is critical: under *rscale:N one spine advances at
    a different rate, so the original line alignment (same line = same
    onset) no longer holds after durations are changed. This function
    rebuilds the line grid based on actual onset times.

    Args:
        sequence: Kern sequence (before expand_tuplets / quantize_oov)

    Returns:
        Sequence with rscale regions resolved and line structure corrected
    """
    from fractions import Fraction

    lines = sequence.split('\n')

    # --- Phase 1: Identify rscale regions ---
    # region = (start_line, end_line, {spine_idx: factor})
    rscale_open: Dict[int, Tuple[int, int]] = {}  # spine → (start_line, factor)
    regions: List[Tuple[int, int, Dict[int, int]]] = []

    for i, line in enumerate(lines):
        if not line.startswith('*') or line.startswith('**'):
            continue
        tokens = line.split('\t')
        for si, tok in enumerate(tokens):
            m = re.match(r'^\*rscale:(\d+)', tok)
            if not m:
                continue
            factor = int(m.group(1))
            if factor > 1:
                rscale_open[si] = (i, factor)
            elif factor == 1 and si in rscale_open:
                start, fac = rscale_open.pop(si)
                regions.append((start, i, {si: fac}))

    # Unclosed regions
    for si, (start, fac) in rscale_open.items():
        regions.append((start, len(lines) - 1, {si: fac}))

    if not regions:
        return sequence

    # --- Phase 2: Filter out cue regions ---
    # Cue/ossia passages use rscale to time-compress an editorial excerpt
    # alongside the main spine. The cue passage is intentionally longer
    # than the main spine, so resolving rscale would create an unresolvable
    # timing mismatch. Since cue notes aren't in the synthesized audio,
    # the tokenizer doesn't need their actual durations.
    #
    # Two cases:
    #   a) *cue opened BEFORE *rscale (e.g., sonata13-4: cue@2255, rscale@2277)
    #   b) *cue opened AFTER *rscale within the region (e.g., sonata13-3: rscale@362, cue@363)
    #
    # Case (a): Track cumulative cue state up to each region's start
    cue_active: Dict[int, bool] = {}
    regions_by_start = {r[0]: r for r in regions}
    for i, line in enumerate(lines):
        if not line.startswith('*') or line.startswith('**'):
            continue
        tokens = line.split('\t')
        for si, tok in enumerate(tokens):
            if re.match(r'^\*cue\b', tok):
                cue_active[si] = True
            elif re.match(r'^\*Xcue\b', tok):
                cue_active.pop(si, None)
        # At region start lines, check if rscale spine is already in cue
        if i in regions_by_start:
            region = regions_by_start[i]
            for si in region[2]:
                if cue_active.get(si, False):
                    region[2][si] = -1

    # Case (b): Scan within each region for *cue that opens after *rscale
    for region in regions:
        start, end, spine_factors = region
        for i in range(start + 1, min(end + 1, len(lines))):
            ln = lines[i]
            if not ln.startswith('*') or ln.startswith('**'):
                continue
            tokens = ln.split('\t')
            for si, tok in enumerate(tokens):
                if re.match(r'^\*cue\b', tok) and si in spine_factors:
                    spine_factors[si] = -1

    # Separate into resolvable vs cue-only regions
    resolve_regions = []
    for start, end, spine_factors in regions:
        factors_clean = {si: f for si, f in spine_factors.items() if f > 0}
        if factors_clean:
            resolve_regions.append((start, end, factors_clean))
        # Cue regions (all factors == -1): leave completely untouched.
        # converter21 handles *rscale natively, so keeping the markers
        # and displayed durations preserves correct measure timing.

    # --- Phase 3: Reflow resolvable regions (reverse order) ---
    resolve_regions.sort(key=lambda r: r[0], reverse=True)

    for start, end, spine_factors in resolve_regions:
        lines = _reflow_rscale_region(lines, start, end, spine_factors)

    return '\n'.join(lines)


def _reflow_rscale_region(
    lines: List[str], start: int, end: int, spine_factors: Dict[int, int]
) -> List[str]:
    """Reflow a single rscale region: resolve durations + realign lines."""
    from fractions import Fraction

    # Determine spine count from first data line in region
    n_spines = 0
    for i in range(start, min(end + 1, len(lines))):
        ln = lines[i]
        if not ln.startswith(('*', '=', '!')) and ln.strip():
            n_spines = len(ln.split('\t'))
            break
    if n_spines == 0:
        # No data lines — just strip rscale markers
        return _strip_rscale_markers(lines, start, end)

    # Separate data lines from non-data lines within [start, end]
    prefix_nondatas = []   # non-data lines before first data line
    suffix_nondatas = []   # non-data lines after last data line
    mid_nondatas = []      # non-data lines between data lines: (after_n_data, content)
    data_lines_raw = []    # [(line_idx, tokens_list)]
    data_count = 0
    first_data_seen = False

    for i in range(start, min(end + 1, len(lines))):
        ln = lines[i]
        is_data = (not ln.startswith(('*', '=', '!')) and ln.strip() != '')

        if is_data:
            first_data_seen = True
            data_lines_raw.append((i, ln.split('\t')))
            data_count += 1
        else:
            # Strip rscale markers from interpretation lines
            content = ln
            if ln.startswith('*') and not ln.startswith('**'):
                tokens = ln.split('\t')
                new_tokens = []
                for si, tok in enumerate(tokens):
                    if re.match(r'^\*rscale:', tok):
                        new_tokens.append('*')
                    else:
                        new_tokens.append(tok)
                if all(t == '*' for t in new_tokens):
                    continue  # drop pure *rscale lines
                content = '\t'.join(new_tokens)

            if not first_data_seen:
                prefix_nondatas.append(content)
            else:
                mid_nondatas.append((data_count, content))

    # Any mid_nondatas after the last data line → suffix
    final_suffix = []
    while mid_nondatas and mid_nondatas[-1][0] >= data_count:
        final_suffix.insert(0, mid_nondatas.pop()[1])
    suffix_nondatas = final_suffix

    if not data_lines_raw:
        return _strip_rscale_markers(lines, start, end)

    # --- Compute initial spine onsets from events BEFORE the region ---
    # Walk backwards from the region start to the nearest barline,
    # then forward to compute each spine's accumulated onset at the region
    # boundary. This accounts for notes of different durations preceding
    # the region (e.g., spine 0 has a dotted half while spine 1 has a quarter).
    barline_idx = start - 1
    while barline_idx >= 0:
        if lines[barline_idx].startswith('='):
            break
        barline_idx -= 1
    barline_idx = max(0, barline_idx)

    # Accumulate from barline to region start
    spine_onset = [Fraction(0)] * n_spines
    for i in range(barline_idx, start):
        ln = lines[i]
        if ln.startswith(('*', '=', '!')) or not ln.strip():
            continue
        tokens_pre = ln.split('\t')
        for si in range(min(len(tokens_pre), n_spines)):
            tok = tokens_pre[si]
            if tok == '.' or not tok:
                continue
            dur = _parse_token_duration_frac(tok)
            # This spine has an event; its onset is current, then advances
            spine_onset[si] = spine_onset[si] + dur

    # timeline: [(onset_frac, spine_idx, resolved_token)]
    timeline: List[Tuple['Fraction', int, str]] = []

    for _, tokens in data_lines_raw:
        for si in range(min(len(tokens), n_spines)):
            tok = tokens[si]
            if tok == '.' or not tok:
                continue
            onset = spine_onset[si]

            # Resolve duration for rscale spines
            if si in spine_factors:
                resolved = _divide_token_duration(tok, spine_factors[si])
            else:
                resolved = tok

            dur = _parse_token_duration_frac(resolved)
            timeline.append((onset, si, resolved))
            spine_onset[si] = onset + dur

    # Sort by onset then spine index
    timeline.sort(key=lambda x: (x[0], x[1]))

    # --- Generate reflowed data lines ---
    new_data_lines = []
    i = 0
    while i < len(timeline):
        cur_onset = timeline[i][0]
        row = ['.'] * n_spines
        while i < len(timeline) and timeline[i][0] == cur_onset:
            _, si, tok = timeline[i]
            if si < n_spines:
                row[si] = tok
            i += 1
        new_data_lines.append('\t'.join(row))

    # --- Interleave non-data lines ---
    result = []
    result.extend(prefix_nondatas)

    data_idx = 0
    mid_idx = 0
    while data_idx < len(new_data_lines):
        # Insert mid non-data lines that belong before/at this data position
        # Scale position: original had data_count lines, new has len(new_data_lines)
        while mid_idx < len(mid_nondatas):
            orig_pos = mid_nondatas[mid_idx][0]
            # Map original position to new position proportionally
            if data_count > 0:
                mapped = orig_pos * len(new_data_lines) / data_count
            else:
                mapped = 0
            if mapped <= data_idx:
                result.append(mid_nondatas[mid_idx][1])
                mid_idx += 1
            else:
                break
        result.append(new_data_lines[data_idx])
        data_idx += 1

    # Remaining mid non-data
    while mid_idx < len(mid_nondatas):
        result.append(mid_nondatas[mid_idx][1])
        mid_idx += 1

    result.extend(suffix_nondatas)

    # Splice back into original lines
    return lines[:start] + result + lines[end + 1:]


def _strip_rscale_markers(lines: List[str], start: int, end: int) -> List[str]:
    """Strip *rscale markers in range [start, end], no data reflow needed."""
    result = list(lines)
    to_remove = []
    for i in range(start, min(end + 1, len(result))):
        ln = result[i]
        if ln.startswith('*') and not ln.startswith('**') and '*rscale:' in ln:
            tokens = ln.split('\t')
            new_tokens = ['*' if re.match(r'^\*rscale:', t) else t for t in tokens]
            if all(t == '*' for t in new_tokens):
                to_remove.append(i)
            else:
                result[i] = '\t'.join(new_tokens)
    for i in reversed(to_remove):
        del result[i]
    return result


# =============================================================================
# Tuplet Ratio Quantization (X%Y → standard duration)
# =============================================================================

def _ratio_to_fraction(x_str: str, y_str: str) -> 'Fraction':
    """Compute actual duration of X%Y as Fraction of WHOLE notes.

    Humdrum X%Y: each note's duration = Y/X whole notes.
    (RhythmQuantizer uses whole-note fractions, same as Humdrum recip.)
    """
    from fractions import Fraction
    return Fraction(int(y_str), int(x_str))


# Exact X%Y → Zeng lookup (precomputed).
# 2^n % 3 → dotted next-shorter: 4*3/(2^n) = 3*(4/2^n) = dotted (2^(n+1)).
_RATIO_EXACT_MAP: Dict[Tuple[str, str], str] = {
    ('4', '3'): '2.',     # 3q = dotted half
    ('8', '3'): '4.',     # 3/2q = dotted quarter
    ('16', '3'): '8.',    # 3/4q = dotted 8th
    ('32', '3'): '16.',   # 3/8q = dotted 16th
    ('64', '3'): '32.',   # 3/16q = dotted 32nd
    ('128', '3'): '64.',  # 3/32q = dotted 64th
}


def quantize_tuplet_ratios(sequence: str) -> str:
    """
    Quantize X%Y tuplet ratio durations and strip *tuplet/*Xtuplet markers.

    Humdrum X%Y notation: note duration = Y × (4/X) quarter notes.
    For example, 56%3 = 3 × (4/56) = 3/14 quarter.

    Strategy:
    1. Exact matches (2^n%3 → dotted): direct replacement, zero error
    2. Groups of same X%Y: IP solver finds optimal Zeng mix (zero error
       when possible, otherwise minimal error < 2ms)
    3. Strip *tuplet/*Xtuplet markers that enclosed X%Y notes

    Args:
        sequence: Kern sequence after expand_tuplets and quantize_oov_tuplets

    Returns:
        Sequence with X%Y quantized and *tuplet markers stripped
    """
    from fractions import Fraction
    import random

    try:
        from .rhythm_quantizer import RhythmQuantizer, distribute_tuplet_random
    except ImportError:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            'rhythm_quantizer',
            __file__.replace('kern_zeng_compat.py', 'rhythm_quantizer.py')
        )
        _rq = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_rq)
        RhythmQuantizer = _rq.RhythmQuantizer
        distribute_tuplet_random = _rq.distribute_tuplet_random

    lines = sequence.split('\n')

    # --- Pass 1: Identify *tuplet/*Xtuplet regions with X%Y ---
    open_tuplets: Dict = {}
    tuplet_lines_to_strip: set = set()

    for line_idx, line in enumerate(lines):
        if line.startswith('*') and not line.startswith('**'):
            tokens = line.split('\t')
            for spine_idx, tok in enumerate(tokens):
                if re.match(r'^\*tuplet', tok):
                    open_tuplets[spine_idx] = {
                        'start_line': line_idx, 'has_ratio': False
                    }
                elif re.match(r'^\*Xtuplet', tok):
                    if spine_idx in open_tuplets:
                        if open_tuplets[spine_idx]['has_ratio']:
                            tuplet_lines_to_strip.add(
                                (open_tuplets[spine_idx]['start_line'], spine_idx)
                            )
                            tuplet_lines_to_strip.add((line_idx, spine_idx))
                        del open_tuplets[spine_idx]
        elif not line.startswith(('=', '*', '!')) and line.strip():
            tokens = line.split('\t')
            for spine_idx, tok in enumerate(tokens):
                if re.search(r'\d+%\d+', tok) and spine_idx in open_tuplets:
                    open_tuplets[spine_idx]['has_ratio'] = True

    for spine_idx, info in open_tuplets.items():
        if info['has_ratio']:
            tuplet_lines_to_strip.add((info['start_line'], spine_idx))

    # Check if there are any ratios at all
    has_ratios = any(
        re.search(r'\d+%\d+', line)
        for line in lines
        if not line.startswith(('=', '*', '!')) and line.strip()
    )
    if not has_ratios:
        return sequence

    # --- Pass 2: Group consecutive X%Y tokens per spine, quantize ---
    # Collect groups: {spine_idx: {'x': str, 'y': str, 'tokens': [(line_idx, token)]}}
    rq = RhythmQuantizer()
    oov_groups: Dict[int, Dict] = {}
    # {line_idx: {spine_idx: new_token}}
    pending_results: Dict[int, Dict[int, str]] = {}
    # {line_idx: {spine_idx: extra_token}} for 2-token splits
    extra_lines: Dict[int, Dict[int, str]] = {}

    def _flush_ratio_group(spine_idx: int):
        if spine_idx not in oov_groups:
            return
        group = oov_groups.pop(spine_idx)
        x_str, y_str = group['x'], group['y']
        tokens_list = group['tokens']
        n = len(tokens_list)
        if n == 0:
            return

        key = (x_str, y_str)
        if key in _RATIO_EXACT_MAP:
            # Exact match: replace duration directly
            new_dur = _RATIO_EXACT_MAP[key]
            for line_idx, token in tokens_list:
                new_tok = re.sub(r'\d+%\d+\.?', new_dur, token)
                pending_results.setdefault(line_idx, {})[spine_idx] = new_tok
        else:
            # IP solver: compute exact total, find best Zeng mix
            per_note_dur = _ratio_to_fraction(x_str, y_str)
            total_dur = per_note_dur * n
            result = rq.quantize_tuplet(total_dur, n)

            if result:
                for i, (line_idx, token) in enumerate(tokens_list):
                    new_dur = result[i]
                    new_tok = re.sub(r'\d+%\d+\.?', new_dur, token)
                    pending_results.setdefault(line_idx, {})[spine_idx] = new_tok
            elif n == 1:
                # Single token, no IP solution: try 2-token split.
                # Find d1 + d2 from vocab that equals total_dur exactly,
                # picking the pair closest to equal split.
                # This handles long durations like 8%9 = 9/8 whole = 1 + 1/8.
                line_idx, token = tokens_list[0]
                best_pair = None
                best_imbalance = None
                for n1, v1 in rq.vocab_sorted:
                    remainder = total_dur - v1
                    if remainder <= Fraction(0):
                        continue
                    for n2, v2 in rq.vocab_sorted:
                        if v2 == remainder:
                            imb = abs(v1 - v2)
                            if best_imbalance is None or imb < best_imbalance:
                                best_pair = (n1, n2)
                                best_imbalance = imb
                            break  # found exact match for this v1
                if best_pair:
                    d1, d2 = best_pair
                    pitch_part = re.sub(r'^[\[({]*\d+%\d+\.?', '', token)
                    is_rest = pitch_part.startswith('r')
                    if is_rest:
                        # Rests don't need ties
                        new_tok1 = d1 + pitch_part
                        new_tok2 = d2 + pitch_part
                    else:
                        new_tok1 = '[' + d1 + pitch_part
                        new_tok2 = d2 + pitch_part + ']'
                    pending_results.setdefault(line_idx, {})[spine_idx] = new_tok1
                    # Mark extra line insertion needed
                    extra_lines.setdefault(line_idx, {})[spine_idx] = new_tok2
                else:
                    # Last resort: nearest single duration (lossy)
                    best_name = '176'
                    best_err = abs(per_note_dur - Fraction(4, 176))
                    for name, val in rq.vocab_sorted:
                        err = abs(per_note_dur - val)
                        if err < best_err:
                            best_err = err
                            best_name = name
                    new_tok = re.sub(r'\d+%\d+\.?', best_name, token)
                    pending_results.setdefault(line_idx, {})[spine_idx] = new_tok
            else:
                # Multi-note group, no IP solution: nearest per note (lossy)
                best_name = '176'
                best_err = abs(per_note_dur - Fraction(4, 176))
                for name, val in rq.vocab_sorted:
                    err = abs(per_note_dur - val)
                    if err < best_err:
                        best_err = err
                        best_name = name
                for line_idx, token in tokens_list:
                    new_tok = re.sub(r'\d+%\d+\.?', best_name, token)
                    pending_results.setdefault(line_idx, {})[spine_idx] = new_tok

    # Scan data lines for ratio groups
    for line_idx, line in enumerate(lines):
        if line.startswith('='):
            # Barline: flush all groups
            for si in list(oov_groups.keys()):
                _flush_ratio_group(si)
            continue
        if line.startswith(('*', '!')) or not line.strip():
            continue

        tokens = line.split('\t')
        for spine_idx, token in enumerate(tokens):
            if token == '.' or not token:
                continue

            m = re.search(r'(\d+)%(\d+)', token)
            if m:
                x_str, y_str = m.group(1), m.group(2)
                # Continue or start group
                if (spine_idx in oov_groups
                        and oov_groups[spine_idx]['x'] == x_str
                        and oov_groups[spine_idx]['y'] == y_str):
                    oov_groups[spine_idx]['tokens'].append((line_idx, token))
                else:
                    _flush_ratio_group(spine_idx)
                    oov_groups[spine_idx] = {
                        'x': x_str, 'y': y_str,
                        'tokens': [(line_idx, token)]
                    }
            else:
                # Non-ratio token: flush group
                if spine_idx in oov_groups:
                    _flush_ratio_group(spine_idx)

    # Flush remaining
    for si in list(oov_groups.keys()):
        _flush_ratio_group(si)

    # Apply pending results and insert extra lines for 2-token splits
    # Process in reverse order so line indices stay valid during insertion
    for line_idx in sorted(pending_results.keys(), reverse=True):
        line = lines[line_idx]
        tokens = line.split('\t')
        for spine_idx, new_tok in pending_results[line_idx].items():
            if spine_idx < len(tokens):
                tokens[spine_idx] = new_tok
        lines[line_idx] = '\t'.join(tokens)

    for line_idx in sorted(extra_lines.keys(), reverse=True):
        # Build extra line: '.' for all spines except the split spine
        base_line = lines[line_idx]
        base_tokens = base_line.split('\t')
        extra_tokens = ['.'] * len(base_tokens)
        for spine_idx, extra_tok in extra_lines[line_idx].items():
            if spine_idx < len(extra_tokens):
                extra_tokens[spine_idx] = extra_tok
        lines.insert(line_idx + 1, '\t'.join(extra_tokens))

    # --- Pass 3: Strip *tuplet/*Xtuplet markers ---
    cleaned_lines = []
    for line_idx, line in enumerate(lines):
        if not line.startswith('*'):
            cleaned_lines.append(line)
            continue

        tokens = line.split('\t')
        cleaned_tokens = []
        for spine_idx, t in enumerate(tokens):
            if (re.match(r'^\*X?tuplet', t)
                    and (line_idx, spine_idx) in tuplet_lines_to_strip):
                cleaned_tokens.append('*')
            else:
                cleaned_tokens.append(t)

        if all(t == '*' for t in cleaned_tokens):
            continue
        cleaned_lines.append('\t'.join(cleaned_tokens))

    return '\n'.join(cleaned_lines)


# =============================================================================
# Main Compatibility Function
# =============================================================================

def apply_zeng_pitch_compat(sequence: str) -> str:
    """
    Apply all Zeng pitch/accidental compatibility transformations.

    Order of operations:
    1. Split breves (0, 00) into tied whole notes
    2. For each token:
       a. Strip natural accidentals (n)
       b. Convert double accidentals (##, --)
       c. Remove slur/phrase markers ((), {})
       d. Clamp extreme pitches to piano range

    Args:
        sequence: Kern sequence

    Returns:
        Transformed sequence compatible with Zeng's pitch vocabulary
    """
    # 1. Split breves first (affects line structure)
    sequence = split_breves_in_sequence(sequence)

    # 2. Process each token
    lines = sequence.split('\n')
    result_lines = []

    for line in lines:
        # Skip control lines
        if line.startswith(('=', '*', '!')) or line.strip() == '':
            result_lines.append(line)
            continue

        tokens = line.split('\t') if '\t' in line else [line]
        processed_tokens = []

        for token in tokens:
            if token == '.' or not token:
                processed_tokens.append(token)
                continue

            # Handle chord tokens (space-separated notes within a single token field)
            # e.g., "16ee# 16cc##" → process each note individually
            if ' ' in token:
                notes = token.split(' ')
                processed_notes = []
                for note in notes:
                    if note == '.' or not note:
                        processed_notes.append(note)
                    else:
                        note = strip_natural_accidentals(note)
                        note = convert_double_accidentals(note)
                        note = remove_slur_phrase_markers(note)
                        note = clamp_extreme_pitches(note)
                        processed_notes.append(note)
                processed_tokens.append(' '.join(processed_notes))
            else:
                # Single note token
                token = strip_natural_accidentals(token)
                token = convert_double_accidentals(token)
                token = remove_slur_phrase_markers(token)
                token = clamp_extreme_pitches(token)
                processed_tokens.append(token)

        result_lines.append('\t'.join(processed_tokens) if len(processed_tokens) > 1 else processed_tokens[0])

    return '\n'.join(result_lines)
