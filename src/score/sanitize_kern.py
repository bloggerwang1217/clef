"""
Kern Sanitization for Audio Generation
=======================================

Combines all kern preprocessing steps needed for converter21/music21 parsing:
1. Repeat expansion (expand_kern_repeats)
2. Spine timing fix (fix_kern_spine_timing)

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


# =============================================================================
# Kern Repeat Expansion
# =============================================================================
# Expands Humdrum repeat structures (expansion labels) for MIDI/Audio generation.
#
# Humdrum uses expansion labels like:
#     *>[A,A,B,B1,B,B2,C,C1,C,C2,D,E,E,F]  # Full playback order
#     *>norep[A,B,B2,C,C2,D,E,F]            # No-repeat version
#     *>A, *>B, *>B1, etc.                  # Section markers
#
# This module expands kern content according to the expansion labels,
# producing a "through-composed" version suitable for MIDI generation.
#
# The original kern file (with repeat markers) is preserved as ground truth,
# while the expanded version is used only for audio synthesis.
# =============================================================================


def has_expansion_labels(kern_content: str) -> bool:
    """Check if kern content has Humdrum expansion labels.

    Args:
        kern_content: Raw kern file content

    Returns:
        True if expansion labels (*>[...]) are present
    """
    return bool(re.search(r'\*>\[[^\]]+\]', kern_content))


def parse_expansion_order(kern_content: str) -> Optional[List[str]]:
    """Extract the expansion order from kern content.

    Args:
        kern_content: Raw kern file content

    Returns:
        List of section names in playback order, or None if no expansion labels

    Example:
        "*>[A,A,B,B1,B,B2]" -> ["A", "A", "B", "B1", "B", "B2"]
    """
    match = re.search(r'\*>\[([^\]]+)\]', kern_content)
    if not match:
        return None
    return match.group(1).split(',')


def parse_section_ranges(kern_content: str) -> Dict[str, Tuple[int, int]]:
    """Parse section markers and their line ranges.

    Args:
        kern_content: Raw kern file content

    Returns:
        Dictionary mapping section name to (start_line, end_line) tuple.
        Lines are 0-indexed, start is inclusive, end is exclusive.

    Example:
        {"A": (15, 69), "B": (70, 104), "B1": (105, 118), ...}
    """
    lines = kern_content.split('\n')
    section_ranges: Dict[str, Tuple[int, int]] = {}
    current_section: Optional[str] = None
    current_start: int = 0

    for i, line in enumerate(lines):
        # Section marker: *>A\t*>A or *>A\t*>A\t*>A (one per spine)
        # But NOT expansion labels (*>[...]) or norep labels (*>norep[...])
        if line.startswith('*>') and '\t' in line:
            parts = line.split('\t')
            # Check all parts are section markers (not expansion/norep labels)
            is_section_marker = all(
                p.startswith('*>') and
                not p.startswith('*>[') and
                not p.startswith('*>norep')
                for p in parts
            )
            if is_section_marker:
                section = parts[0][2:]  # Extract "A" from "*>A"
                if section:
                    # Close previous section
                    if current_section is not None:
                        section_ranges[current_section] = (current_start, i)
                    # Start new section (content starts on next line)
                    current_section = section
                    current_start = i + 1

    # Close last section
    if current_section is not None:
        section_ranges[current_section] = (current_start, len(lines))

    return section_ranges


def _clean_repeat_barline(line: str) -> str:
    """Remove repeat markers from barline while preserving measure number.

    Humdrum repeat barlines look like:
        =5:|!|:  -> =5
        =:|!|:   -> =
        =10!|:   -> =10

    Args:
        line: A line that starts with '='

    Returns:
        Cleaned line with repeat markers removed
    """
    if not line.startswith('='):
        return line

    parts = line.split('\t')
    cleaned_parts = []

    for part in parts:
        if part.startswith('='):
            # Remove repeat markers: :, |, !
            # But keep the = and measure number
            cleaned = re.sub(r'[:\|!]+', '', part)
            # Ensure it still starts with =
            if not cleaned.startswith('='):
                cleaned = '=' + cleaned.lstrip('=')
            cleaned_parts.append(cleaned)
        else:
            cleaned_parts.append(part)

    return '\t'.join(cleaned_parts)


def _renumber_barlines(lines: List[str]) -> List[str]:
    """Renumber barlines sequentially after expansion.

    After expanding repeats, multiple sections may have the same measure numbers.
    This function renumbers all barlines sequentially to avoid confusion in
    music21/converter21.

    Pickup measures (=N-) are handled specially:
    - First pickup measure becomes =0
    - Subsequent barlines are numbered 1, 2, 3, ...

    Final barlines (==) are preserved as-is.

    Args:
        lines: List of kern file lines

    Returns:
        List with renumbered barlines
    """
    result = []
    measure_counter = 0
    first_barline_seen = False

    for line in lines:
        if not line.startswith('='):
            result.append(line)
            continue

        # Skip final barline
        if line.startswith('=='):
            result.append(line)
            continue

        parts = line.split('\t')

        # Determine the new measure number for this line (all spines get same number)
        # Check if this is a pickup measure by looking at the first barline part
        first_bar_part = next((p for p in parts if p.startswith('=')), None)
        is_pickup = first_bar_part and '-' in first_bar_part and not first_bar_part.startswith('==')

        if not first_barline_seen:
            first_barline_seen = True
            if is_pickup:
                new_measure_num = 0
            else:
                measure_counter += 1
                new_measure_num = measure_counter
        else:
            measure_counter += 1
            new_measure_num = measure_counter

        # Apply the same measure number to all barline spines on this line
        new_parts = []
        for part in parts:
            if part.startswith('=') and not part.startswith('=='):
                new_parts.append(f'={new_measure_num}')
            else:
                new_parts.append(part)

        result.append('\t'.join(new_parts))

    return result


def _count_spine_change(line: str) -> int:
    """Count spine change from a line containing *^ or *v.

    Args:
        line: A line that may contain spine split (*^) or merge (*v)

    Returns:
        Net change in spine count (positive for split, negative for merge)
    """
    if not line.startswith('*') or line.startswith('**'):
        return 0

    parts = line.split('\t')
    change = 0
    i = 0
    while i < len(parts):
        if parts[i] == '*^':
            change += 1  # One spine becomes two
        elif parts[i] == '*v':
            # Count consecutive *v tokens (they merge into one)
            merge_count = 0
            while i < len(parts) and parts[i] == '*v':
                merge_count += 1
                i += 1
            change -= (merge_count - 1)  # N spines become 1
            continue
        i += 1
    return change


def _get_spine_count(line: str) -> int:
    """Get the number of spines (tab-separated fields) in a line."""
    if not line or line.startswith('!'):
        return 0
    return len(line.split('\t'))


def expand_kern_repeats(kern_content: str) -> str:
    """Expand kern content according to Humdrum expansion labels.

    This function:
    1. Parses the expansion order (*>[A,A,B,...])
    2. Identifies section boundaries (*>A, *>B, etc.)
    3. Reassembles content in playback order
    4. Handles spine count mismatches between sections by inserting merge lines
    5. Removes repeat barlines to avoid music21 expandRepeats issues

    Args:
        kern_content: Raw kern file content with expansion labels

    Returns:
        Expanded kern content (through-composed, no repeats)

    Note:
        If no expansion labels are found, returns the original content
        with repeat barlines removed.
    """
    # Parse expansion order
    expansion_order = parse_expansion_order(kern_content)
    if not expansion_order:
        # No expansion labels - just remove repeat barlines
        return remove_repeat_barlines(kern_content)

    # Parse section ranges
    section_ranges = parse_section_ranges(kern_content)
    if not section_ranges:
        # No section markers found - return original with barlines cleaned
        return remove_repeat_barlines(kern_content)

    lines = kern_content.split('\n')

    # Find header end (everything before first section marker)
    first_section_start = min(start for start, _ in section_ranges.values())
    # Header ends one line before first section content (the marker line itself)
    header_end = first_section_start - 1

    # Collect header lines (exclude expansion and norep labels)
    header_lines = []
    base_spine_count = 2  # Default
    for i, line in enumerate(lines[:header_end]):
        # Skip expansion labels
        if line.startswith('*>[') or line.startswith('*>norep['):
            continue
        # Skip if all parts are expansion/norep labels
        if '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>[') or p.startswith('*>norep[') for p in parts):
                continue
        # Get base spine count from **kern line
        if line.startswith('**'):
            base_spine_count = len(line.split('\t'))
        header_lines.append(line)

    # Assemble expanded content
    expanded_lines = header_lines.copy()
    current_spine_count = base_spine_count

    for section in expansion_order:
        if section not in section_ranges:
            # Section not found, skip (shouldn't happen in well-formed files)
            continue

        start, end = section_ranges[section]

        # Get the starting spine count expected by this section
        # Look for the first data or interpretation line in the section
        section_start_spines = base_spine_count
        for line in lines[start:end]:
            if line and '\t' in line and not line.startswith('!'):
                section_start_spines = len(line.split('\t'))
                break

        # If current spine count doesn't match section start, insert merge/split
        if current_spine_count > section_start_spines:
            # Need to merge extra spines
            # Insert merge line: merge down to section_start_spines
            merge_parts = []
            remaining = current_spine_count
            target = section_start_spines
            while remaining > target:
                # Merge the last two spines into one
                merge_parts = ['*'] * (remaining - 2) + ['*v', '*v']
                expanded_lines.append('\t'.join(merge_parts))
                remaining -= 1
            current_spine_count = section_start_spines
        elif current_spine_count < section_start_spines:
            # Need to split - this is rare, usually sections start with base spines
            # Insert split line
            split_parts = ['*'] * (current_spine_count - 1) + ['*^']
            while current_spine_count < section_start_spines:
                expanded_lines.append('\t'.join(split_parts))
                current_spine_count += 1
                split_parts = ['*'] * (current_spine_count - 1) + ['*^']

        # Add section content
        for line in lines[start:end]:
            # Skip spine terminators - we'll add one at the end
            if line.strip() and all(p.strip() == '*-' for p in line.split('\t')):
                continue
            # Clean repeat barlines
            if line.startswith('='):
                line = _clean_repeat_barline(line)
            expanded_lines.append(line)

            # Track spine count changes
            if line.startswith('*') and not line.startswith('**'):
                current_spine_count += _count_spine_change(line)

    # Merge back to base spine count before terminator if needed
    while current_spine_count > base_spine_count:
        merge_parts = ['*'] * (current_spine_count - 2) + ['*v', '*v']
        expanded_lines.append('\t'.join(merge_parts))
        current_spine_count -= 1

    # Add spine terminator at the end
    expanded_lines.append('\t'.join(['*-'] * base_spine_count))

    # Renumber barlines sequentially to avoid duplicates after expansion
    # This is critical for music21/converter21 to parse the file correctly
    expanded_lines = _renumber_barlines(expanded_lines)

    return '\n'.join(expanded_lines)


def remove_repeat_barlines(kern_content: str) -> str:
    """Remove repeat barlines from kern content without expanding.

    Use this for files without expansion labels but with repeat barlines
    that cause music21 to fail.

    Args:
        kern_content: Raw kern file content

    Returns:
        Kern content with repeat barlines converted to regular barlines
    """
    lines = kern_content.split('\n')
    cleaned_lines = []

    for line in lines:
        # Skip expansion and norep labels
        if re.match(r'^\*>\[.*\]', line) or re.match(r'^\*>norep\[.*\]', line):
            continue

        # Skip section markers (*>A, *>B, etc.)
        if line.startswith('*>') and '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>') and not p.startswith('*>[') for p in parts):
                continue

        # Clean repeat barlines
        if line.startswith('='):
            line = _clean_repeat_barline(line)

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def get_expansion_info(kern_content: str) -> Dict:
    """Get information about the repeat structure for debugging.

    Args:
        kern_content: Raw kern file content

    Returns:
        Dictionary with expansion info:
        - has_expansion: bool
        - expansion_order: List[str] or None
        - sections: Dict[str, (start, end)]
        - estimated_expansion_ratio: float (expanded / original)
    """
    expansion_order = parse_expansion_order(kern_content)
    section_ranges = parse_section_ranges(kern_content)

    info = {
        'has_expansion': expansion_order is not None,
        'expansion_order': expansion_order,
        'sections': section_ranges,
        'section_count': len(section_ranges),
    }

    if expansion_order and section_ranges:
        # Estimate expansion ratio
        original_lines = sum(end - start for start, end in section_ranges.values())
        expanded_lines = sum(
            section_ranges[s][1] - section_ranges[s][0]
            for s in expansion_order
            if s in section_ranges
        )
        info['estimated_expansion_ratio'] = expanded_lines / original_lines if original_lines > 0 else 1.0
    else:
        info['estimated_expansion_ratio'] = 1.0

    return info


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
    # Step 1: Expand repeats
    result = expand_kern_repeats(kern_content)

    # Step 2: Fix spine timing inconsistencies
    result = fix_kern_spine_timing(result)

    return result


# Public API
__all__ = [
    # Main entry point
    'sanitize_kern_for_audio',
    # Repeat expansion
    'expand_kern_repeats',
    'remove_repeat_barlines',
    'has_expansion_labels',
    'get_expansion_info',
    # Spine timing fix
    'fix_kern_spine_timing',
    'parse_kern_duration',
    'duration_to_rest',
]
