"""
Kern Sanitization for Audio Generation
=======================================

Combines kern preprocessing steps needed for converter21/music21 parsing:
1. Repeat expansion (via ``src.score.expand_repeat.expand_kern_repeats``)
2. Spine timing fix (``fix_kern_spine_timing``)
3. Measure extraction (``extract_kern_measures``)

Repeat-related functions have been consolidated into ``src.score.expand_repeat``.

Usage:
    from src.score.sanitize_kern import sanitize_kern_for_audio

    with open("input.krn") as f:
        kern_content = f.read()

    sanitized = sanitize_kern_for_audio(kern_content)
    # Now safe to parse with converter21/music21
"""

import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Spine Timing Fix
# =============================================================================
# Fixes converter21's negative offset bug caused by inconsistent spine timing
# within split (*^) regions.
#
# Problem:
#     When different voices in a split region have different accumulated durations,
#     converter21 cannot calculate correct offsets and sets them to -1.0.
#
# Solution:
#     Insert rest tokens to synchronize spine positions at each "sync point"
#     (where non-null spines start new notes).
# =============================================================================

def parse_kern_duration(token: str) -> Optional[Fraction]:
    """Parse kern token duration, returns Fraction or None.

    This implementation uses re.search to find duration numbers anywhere
    in the token, which correctly handles tokens like ")4d" where the
    duration number is not at the start.

    Args:
        token: Kern token like "8c#", "16.ee-", "[4f", ")4d"

    Returns:
        Duration as Fraction of quarter notes, or None for invalid tokens
    """
    if not token or token == '.' or token.startswith(('!', '*', '=')):
        return None

    # Grace note (q) has 0 duration
    if 'q' in token.lower() and not token.startswith('='):
        return Fraction(0)

    token_clean = token.split()[0] if ' ' in token else token
    match = re.search(r'(\d+)', token_clean)
    if not match:
        return None

    recip = int(match.group(1))
    dur = Fraction(8) if recip == 0 else Fraction(4, recip)

    # Handle dots - count all dots in token (dots can appear after pitch letter)
    # e.g., "(4A." = slur + 4 + pitch A + dot, the dot is AFTER the letter
    dots = token_clean.count('.')
    if dots > 0:
        dur = dur * (Fraction(2) - Fraction(1, 2**dots))

    return dur


def duration_to_rest(dur: Fraction) -> str:
    """Convert duration (in quarter notes) to kern rest token(s).

    Args:
        dur: Duration as Fraction

    Returns:
        Kern rest token(s), may be space-separated for complex durations
    """
    dur_map = {
        Fraction(4): '1r', Fraction(3): '2.r', Fraction(2): '2r',
        Fraction(3, 2): '4.r', Fraction(1): '4r', Fraction(3, 4): '8.r',
        Fraction(1, 2): '8r', Fraction(3, 8): '16.r', Fraction(1, 4): '16r',
        Fraction(3, 16): '32.r', Fraction(1, 8): '32r', Fraction(3, 32): '64.r',
        Fraction(1, 16): '64r', Fraction(1, 32): '128r', Fraction(1, 64): '256r',
    }

    if dur in dur_map:
        return dur_map[dur]

    # Compound duration: split into multiple rests
    remaining = dur
    rests = []
    for d in sorted(dur_map.keys(), reverse=True):
        while remaining >= d > 0:
            rests.append(dur_map[d])
            remaining -= d
    return ' '.join(rests) if rests else '64r'


def fix_kern_spine_timing(kern_content: str) -> str:
    """Fix spine timing inconsistencies in kern content.

    Inserts rest tokens to synchronize spine positions within split regions,
    preventing converter21's negative offset bug.

    Only syncs spines that share the same parent lineage (came from the same
    original spine via splits). This prevents false sync when different voice
    lineages have legitimately different rhythms.

    Args:
        kern_content: Raw kern content (should be already repeat-expanded)

    Returns:
        Fixed kern content with synchronized spine timing
    """
    lines = kern_content.split('\n')
    new_lines = []
    in_split = False
    current_spine_count = 2
    spine_positions: Dict[int, Fraction] = {}
    # Track parent lineage: spines from the same original voice have the same group ID
    # Initially, each spine is its own group (0, 1 for 2-spine piece)
    spine_parent_group: Dict[int, int] = {0: 0, 1: 1}
    next_group_id = 2  # For tracking merged groups

    def get_fix_line(lagging_spines: set, target_pos: Fraction) -> str:
        """Generate a fix line for lagging spines."""
        fix_cols = []
        for j in range(current_spine_count):
            if j in lagging_spines:
                diff = target_pos - spine_positions[j]
                fix_cols.append(duration_to_rest(diff))
            else:
                fix_cols.append('.')
        return '\t'.join(fix_cols)

    for line in lines:
        stripped = line.rstrip('\n')

        # Handle spine manipulators (*^, *v) - can appear in same line!
        # e.g., "*v\t*v\t*^" or "*^\t*v\t*v\t*"
        if ('*^' in stripped or '*v' in stripped) and '\t' in stripped:
            cols = stripped.split('\t')

            # NOTE: We do NOT sync before merge operations.
            # Different voices within the same parent group can legitimately have
            # different rhythms (e.g., dotted quarter vs quarter in piano music).
            # Merge naturally takes max position, which is correct.
            # Syncing at barlines is sufficient for proper alignment.

            # Process all spine operations in one pass
            new_positions: Dict[int, Fraction] = {}
            new_parent_group: Dict[int, int] = {}
            new_idx = 0
            old_idx = 0
            while old_idx < len(cols):
                col = cols[old_idx]
                if col == '*^':
                    # Split: one spine becomes two with same position AND same parent group
                    pos = spine_positions.get(old_idx, Fraction(0))
                    grp = spine_parent_group.get(old_idx, old_idx)
                    new_positions[new_idx] = pos
                    new_positions[new_idx + 1] = pos
                    new_parent_group[new_idx] = grp
                    new_parent_group[new_idx + 1] = grp
                    new_idx += 2
                    old_idx += 1
                elif col == '*v':
                    # Merge: consecutive *v tokens become one spine
                    # Merged spine gets a new unique group ID (different lineages merged)
                    merge_pos = spine_positions.get(old_idx, Fraction(0))
                    old_idx += 1
                    while old_idx < len(cols) and cols[old_idx] == '*v':
                        merge_pos = max(merge_pos, spine_positions.get(old_idx, Fraction(0)))
                        old_idx += 1
                    new_positions[new_idx] = merge_pos
                    new_parent_group[new_idx] = next_group_id
                    new_idx += 1
                else:
                    # Regular spine token (*, *clef, *k[], etc.)
                    new_positions[new_idx] = spine_positions.get(old_idx, Fraction(0))
                    new_parent_group[new_idx] = spine_parent_group.get(old_idx, old_idx)
                    new_idx += 1
                    old_idx += 1

            current_spine_count = new_idx
            spine_positions = new_positions
            spine_parent_group = new_parent_group
            if '*v' in stripped:
                next_group_id += 1  # Increment for next potential merge

            # Update in_split status
            if '*^' in stripped:
                in_split = True
            if current_spine_count <= 2:
                in_split = False

            new_lines.append(line)
            continue

        # Barlines - sync lagging spines (only within same parent group)
        if stripped.startswith('=') and in_split:
            if spine_positions:
                # Group spines by parent lineage
                groups: Dict[int, List[int]] = {}
                for j, grp in spine_parent_group.items():
                    if grp not in groups:
                        groups[grp] = []
                    groups[grp].append(j)

                # Sync within each group
                for grp, spines in groups.items():
                    if len(spines) > 1:
                        group_positions = [spine_positions.get(j, Fraction(0)) for j in spines]
                        max_pos = max(group_positions)
                        lagging = {j for j in spines if spine_positions.get(j, Fraction(0)) < max_pos}
                        if lagging:
                            new_lines.append(get_fix_line(lagging, max_pos))
                            for j in lagging:
                                spine_positions[j] = max_pos

            new_lines.append(line)
            continue

        # Other interpretation lines, comments, empty lines
        if stripped.startswith('*') or stripped.startswith('!') or not stripped:
            new_lines.append(line)
            continue

        # Data line - ALWAYS track positions, sync only in split regions
        # Bug fix: Previously only tracked in split regions, but position divergence
        # can happen in non-split regions (e.g., quarter vs eighth in 2-spine section)
        # and carry over when split happens again.
        cols = stripped.split('\t')

        # Sync logic: only in split regions, and only within same parent group
        if in_split:
            # Find spines starting new notes (non-null with duration)
            active_spines = []
            for j, col in enumerate(cols):
                if col != '.':
                    dur = parse_kern_duration(col)
                    if dur is not None and dur > 0:
                        active_spines.append(j)

            # Group active spines by parent lineage
            if len(active_spines) > 1:
                active_by_group: Dict[int, List[int]] = {}
                for j in active_spines:
                    grp = spine_parent_group.get(j, j)
                    if grp not in active_by_group:
                        active_by_group[grp] = []
                    active_by_group[grp].append(j)

                # Sync within each group that has multiple active spines
                for grp, spines in active_by_group.items():
                    if len(spines) > 1:
                        group_positions = [spine_positions.get(j, Fraction(0)) for j in spines]
                        max_pos = max(group_positions)
                        lagging = {j for j in spines if spine_positions.get(j, Fraction(0)) < max_pos}

                        if lagging:
                            new_lines.append(get_fix_line(lagging, max_pos))
                            for j in lagging:
                                spine_positions[j] = max_pos

        # ALWAYS update spine positions (not just in split regions!)
        for j, col in enumerate(cols):
            if j not in spine_positions:
                spine_positions[j] = Fraction(0)
            if col != '.':
                dur = parse_kern_duration(col)
                if dur is not None:
                    spine_positions[j] += dur

        new_lines.append(line)

    return '\n'.join(new_lines)


def extract_kern_measures(kern_path=None, include_timing: bool = False,
                          kern_content: str = None) -> List[Dict]:
    """Extract measure line-ranges from kern file or content string.

    Splits kern content at barlines to find line ranges for each measure region.
    Used for ChunkedDataset — slicing kern_gt tokens at measure boundaries.

    NOTE: The ``measure`` field is a local index (0-based if pickup exists,
    1-based otherwise). It is NOT authoritative — the definitive measure
    number and timing come from the music21 Score via ``extract_measure_times()``.
    ChunkedDataset pairs kern_measures[i] with audio_measures[i] by **index**,
    not by measure number.

    Args:
        kern_path: Path to kern file (str or Path). Mutually exclusive with kern_content.
        include_timing: If True, include start_sec/end_sec (kern-parsed, approximate).
                        For authoritative timing, use extract_measure_times(Score).
        kern_content: Raw kern content string. If provided, kern_path is ignored.

    Returns:
        List of measure info dicts:
        [
            {"measure": 0, "line_start": 21, "line_end": 21},   # pickup (if any)
            {"measure": 1, "line_start": 23, "line_end": 26},
            ...
        ]
        line_start is the first data line of the measure (after the barline)
        line_end is the last data line before the next barline (or end of file)
    """
    measures = []
    current_measure = None
    measure_start_line = None

    # Pattern to match barlines: =N, =N-, =N:|!, ==, etc.
    # Captures the measure number if present
    barline_pattern = re.compile(r'^=(\d+)?')

    # Timing state (only used when include_timing=True)
    current_offset = Fraction(0)       # cumulative offset in quarter notes
    measure_start_offset = Fraction(0) # offset at start of current measure
    tempo_changes: List[Tuple[Fraction, float]] = [(Fraction(0), 120.0)]  # (offset, bpm)

    def offset_to_seconds(offset: Fraction) -> float:
        """Convert quarter note offset to seconds using tempo map."""
        seconds = 0.0
        prev_offset = Fraction(0)
        prev_bpm = tempo_changes[0][1]
        for t_offset, t_bpm in tempo_changes:
            if t_offset >= offset:
                break
            seconds += float(t_offset - prev_offset) * (60.0 / prev_bpm)
            prev_offset = t_offset
            prev_bpm = t_bpm
        seconds += float(offset - prev_offset) * (60.0 / prev_bpm)
        return seconds

    if kern_content is not None:
        lines = kern_content.split('\n')
        # Add newline back for consistency with readlines() format
        lines = [line + '\n' for line in lines]
    else:
        from pathlib import Path as _Path
        kern_path = _Path(kern_path)
        with open(kern_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

    # Pre-scan: find the line number of the LAST final barline (==) in
    # the file.  Only that barline terminates measure tracking; earlier ==
    # barlines (e.g. DaCapo section boundaries) are treated as regular
    # barlines so the music after them is still counted.
    last_final_barline: Optional[int] = None
    for scan_i, scan_line in enumerate(lines, start=1):
        if scan_line.strip().split('\t')[0].startswith('=='):
            last_final_barline = scan_i

    # Detect pickup measure: scan for data lines before the first barline.
    # If found, create measure 0 for the anacrusis content.
    pickup_data_start = None
    for scan_num, scan_line in enumerate(lines, start=1):
        scan_stripped = scan_line.strip()
        if not scan_stripped or scan_stripped.startswith('!') or scan_stripped.startswith('*'):
            continue
        first_tok = scan_stripped.split('\t')[0]
        if barline_pattern.match(first_tok):
            break  # Reached first barline — no pickup (or pickup already handled)
        # Found data before any barline → pickup measure exists
        if pickup_data_start is None:
            pickup_data_start = scan_num

    if pickup_data_start is not None:
        # There's content before the first barline — start tracking measure 0
        current_measure = 0
        measure_start_line = pickup_data_start
        measure_start_offset = Fraction(0)

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('!'):
            continue

        # Check for barline
        first_token = line.split('\t')[0]
        match = barline_pattern.match(first_token)

        if match:
            # Found a barline
            measure_num_str = match.group(1)

            # Close previous measure (including pickup measure 0)
            if current_measure is not None and measure_start_line is not None:
                line_end = line_num - 1  # End before this barline
                # Skip measures with no actual music data (only comments,
                # interpretations like key/meter changes, or empty lines).
                # music21 does not create Measure objects for these, so
                # including them would cause kern_measures/audio_measures mismatch.
                has_music = False
                if line_end >= measure_start_line:
                    for check_i in range(measure_start_line - 1, line_end):
                        raw = lines[check_i].strip()
                        if raw and not raw.startswith('!') and not raw.startswith('*') and not raw.startswith('='):
                            has_music = True
                            break
                if has_music:
                    entry = {
                        "measure": current_measure,
                        "line_start": measure_start_line,
                        "line_end": line_end,
                    }
                    if include_timing:
                        entry["start_sec"] = round(offset_to_seconds(measure_start_offset), 4)
                        entry["end_sec"] = round(offset_to_seconds(current_offset), 4)
                    measures.append(entry)

            # The LAST final barline (==) in the file signals end of
            # piece.  Do NOT start a new measure — any content after it
            # (e.g. tie resolution, reference comments) belongs to the
            # preceding measure, not a new one.  music21 does not create
            # a Measure for post-final-barline content.
            # NOTE: == can also appear as a section double barline
            # mid-piece (e.g. DaCapo point), so we only skip when this
            # is truly the last == in the file.
            if first_token.startswith('==') and line_num == last_final_barline:
                current_measure = None
                measure_start_line = None
                continue

            # Start new measure
            if measure_num_str:
                current_measure = int(measure_num_str)
            elif current_measure is not None:
                current_measure += 1
            else:
                current_measure = 1

            measure_start_line = line_num + 1  # Start after barline
            measure_start_offset = current_offset

        elif first_token.startswith('*'):
            # Interpretation line (metadata) - not part of measure content
            # But update measure_start_line if we haven't started the measure yet
            if measure_start_line == line_num:
                measure_start_line = line_num + 1

            # Parse tempo and time signature for timing
            if include_timing:
                if '*MM' in line:
                    tempo_match = re.search(r'\*MM=?(\d+\.?\d*)', line)
                    if tempo_match:
                        bpm = float(tempo_match.group(1))
                        tempo_changes.append((current_offset, bpm))
            continue

        else:
            # Data line - advance offset by max duration across spines
            if include_timing:
                cols = line.split('\t')
                max_dur = Fraction(0)
                for col in cols:
                    if col == '.' or not col:
                        continue
                    # Handle chords (space-separated subtokens)
                    for subtoken in col.split():
                        dur = parse_kern_duration(subtoken)
                        if dur is not None and dur > max_dur:
                            max_dur = dur
                if max_dur > 0:
                    current_offset += max_dur

    # Close the last measure — only if it contains actual music data.
    # After the final barline (==), there are typically only reference
    # comments (!!!) and spine terminators (*-), which should NOT form
    # a phantom measure.
    if current_measure is not None and measure_start_line is not None:
        has_data = False
        last_data_line = measure_start_line
        for i in range(measure_start_line - 1, len(lines)):
            raw = lines[i].strip()
            # Skip empty, comments, spine terminators, interpretation lines
            if not raw or raw.startswith('!') or raw.startswith('*'):
                continue
            # Skip barlines (shouldn't appear, but just in case)
            if raw.startswith('='):
                continue
            # Found actual music data
            has_data = True
            last_data_line = i + 1  # 1-indexed

        if has_data:
            entry = {
                "measure": current_measure,
                "line_start": measure_start_line,
                "line_end": last_data_line,
            }
            if include_timing:
                entry["start_sec"] = round(offset_to_seconds(measure_start_offset), 4)
                entry["end_sec"] = round(offset_to_seconds(current_offset), 4)
            measures.append(entry)

    return measures


# =============================================================================
# Main Entry Point
# =============================================================================


def sanitize_kern_for_audio(kern_content: str) -> str:
    """Sanitize kern content for audio generation via converter21/music21.

    Applies the full preprocessing pipeline:
    1. expand_kern_repeats - Expand Humdrum expansion labels (*>[A,B,A,...])
    2. fix_kern_spine_timing - Insert rests to synchronize spine timing

    Args:
        kern_content: Raw kern file content

    Returns:
        Sanitized kern content safe for converter21/music21 parsing
    """
    # Lazy import to avoid circular dependency:
    # expand_repeat imports extract_kern_measures from this module.
    from src.score.expand_repeat import expand_kern_repeats

    # Step 1: Expand repeats
    result = expand_kern_repeats(kern_content)

    # Step 2: Fix spine timing inconsistencies
    result = fix_kern_spine_timing(result)

    return result


# Public API
__all__ = [
    # Main entry point
    'sanitize_kern_for_audio',
    # Measure extraction
    'extract_kern_measures',
    # Spine timing fix
    'fix_kern_spine_timing',
    'parse_kern_duration',
    'duration_to_rest',
]
