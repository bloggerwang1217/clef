"""
Kern Repeat Expansion
=====================

Expands Humdrum repeat structures (expansion labels) for MIDI/Audio generation.

Humdrum uses expansion labels like:
    *>[A,A,B,B1,B,B2,C,C1,C,C2,D,E,E,F]  # Full playback order
    *>norep[A,B,B2,C,C2,D,E,F]            # No-repeat version
    *>A, *>B, *>B1, etc.                  # Section markers

This module expands kern content according to the expansion labels,
producing a "through-composed" version suitable for MIDI generation.

The original kern file (with repeat markers) is preserved as ground truth,
while the expanded version is used only for audio synthesis.

Usage:
    from src.score.kern_repeat import expand_kern_repeats, has_expansion_labels

    if has_expansion_labels(kern_content):
        expanded = expand_kern_repeats(kern_content)
        # Use expanded for MIDI generation
    else:
        # No expansion needed
        pass
"""

import re
from typing import Dict, List, Optional, Tuple


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


def expand_kern_repeats(kern_content: str) -> str:
    """Expand kern content according to Humdrum expansion labels.

    This function:
    1. Parses the expansion order (*>[A,A,B,...])
    2. Identifies section boundaries (*>A, *>B, etc.)
    3. Reassembles content in playback order
    4. Removes repeat barlines to avoid music21 expandRepeats issues

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
    for i, line in enumerate(lines[:header_end]):
        # Skip expansion labels
        if line.startswith('*>[') or line.startswith('*>norep['):
            continue
        # Skip if all parts are expansion/norep labels
        if '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>[') or p.startswith('*>norep[') for p in parts):
                continue
        header_lines.append(line)

    # Assemble expanded content
    expanded_lines = header_lines.copy()

    for section in expansion_order:
        if section not in section_ranges:
            # Section not found, skip (shouldn't happen in well-formed files)
            continue

        start, end = section_ranges[section]
        for line in lines[start:end]:
            # Skip spine terminators - we'll add one at the end
            if line.strip() and all(p.strip() == '*-' for p in line.split('\t')):
                continue
            # Clean repeat barlines
            if line.startswith('='):
                line = _clean_repeat_barline(line)
            expanded_lines.append(line)

    # Add spine terminator at the end
    # Determine number of spines from last non-empty content line
    num_spines = 1
    for line in reversed(expanded_lines):
        if '\t' in line and not line.startswith(('!', '*')):
            num_spines = len(line.split('\t'))
            break
        elif line.startswith('**'):
            num_spines = len(line.split('\t'))
            break
    expanded_lines.append('\t'.join(['*-'] * num_spines))

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


def sanitize_tuplet_durations(kern_content: str) -> str:
    """Remove dots from notes within tuplet regions to avoid DurationException.

    Problem:
        music21/converter21 cannot represent dotted notes in tuplet contexts
        because the resulting quarterLength (e.g., 2.25) cannot be expressed
        as a simple fraction. This causes DurationException during parsing.

    Solution:
        Remove the dot from note durations within tuplet regions (*tuplet to *Xtuplet).
        This is mathematically approximate but allows the kern to parse.

        Example: 4.ee- in 3/2 tuplet → 4ee- (loses some duration precision)

    Mathematical justification:
        In a 3/2 tuplet, a dotted quarter (1.5 beats) theoretically becomes:
        1.5 × (2/3) = 1.0 beats = quarter note
        So removing the dot is actually close to the correct duration.

    Args:
        kern_content: Raw kern file content

    Returns:
        Kern content with dotted notes in tuplet regions converted to undotted
    """
    lines = kern_content.split('\n')
    result_lines = []

    # Track tuplet state per spine (column)
    # Each spine can have independent tuplet regions
    num_spines = 0
    tuplet_active = []  # List[bool] per spine

    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.startswith('!'):
            result_lines.append(line)
            continue

        # Count spines from first data line
        if line.startswith('**'):
            num_spines = len(line.split('\t'))
            tuplet_active = [False] * num_spines
            result_lines.append(line)
            continue

        # Handle interpretation lines (tuplet markers)
        if line.startswith('*'):
            parts = line.split('\t')

            # Update tuplet state for each spine
            for i, part in enumerate(parts):
                if i < len(tuplet_active):
                    if part == '*tuplet':
                        tuplet_active[i] = True
                    elif part == '*Xtuplet':
                        tuplet_active[i] = False

            result_lines.append(line)
            continue

        # Handle barlines
        if line.startswith('='):
            result_lines.append(line)
            continue

        # Process data lines - remove dots from notes in tuplet regions
        parts = line.split('\t')
        new_parts = []

        for i, part in enumerate(parts):
            if i < len(tuplet_active) and tuplet_active[i]:
                # In tuplet region - remove dots from duration
                # Pattern: number followed by dot, then pitch (letter)
                # e.g., "4.ee-" → "4ee-", "8.ccL" → "8ccL"
                new_part = re.sub(
                    r'^(\d+)\.([A-Ga-gr])',
                    r'\1\2',
                    part
                )
                new_parts.append(new_part)
            else:
                new_parts.append(part)

        result_lines.append('\t'.join(new_parts))

    return '\n'.join(result_lines)
