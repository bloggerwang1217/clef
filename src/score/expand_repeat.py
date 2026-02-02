"""
Repeat Expansion for Kern and Music21 Scores
=============================================

Consolidates all repeat-related functionality:

1. **Kern-level expansion** (Humdrum expansion labels ``*>[A,A,B,...]``):
   - ``expand_kern_repeats()`` -- expand kern content to through-composed
   - ``expand_kern_repeats_with_mapping()`` -- expand + build repeat_map

2. **Music21-level expansion** (MuseSyn MusicXML):
   - ``expand_musesyn_score()`` -- expand Score + detect repeats
   - ``extract_repeat_structure()`` -- extract barlines / DaCapo / volta
   - ``build_musesyn_repeat_map()`` -- build rich repeat_map from Score

3. **Utilities**:
   - ``has_expansion_labels()``, ``parse_expansion_order()``,
     ``parse_section_ranges()``, ``remove_repeat_barlines()``,
     ``get_expansion_info()``
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import music21 as m21

from src.score.sanitize_kern import extract_kern_measures

logger = logging.getLogger(__name__)

# ============================================================================
# Kern-level utilities
# ============================================================================


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
        ``"*>[A,A,B,B1,B,B2]"`` -> ``["A", "A", "B", "B1", "B", "B2"]``
    """
    match = re.search(r'\*>\[([^\]]+)\]', kern_content)
    if not match:
        return None
    return [s.strip() for s in match.group(1).split(',')]


def parse_section_ranges(kern_content: str) -> Dict[str, Tuple[int, int]]:
    """Parse section markers and their line ranges.

    Args:
        kern_content: Raw kern file content

    Returns:
        Dictionary mapping section name to (start_line, end_line) tuple.
        Lines are 0-indexed, start is inclusive, end is exclusive.

    Example:
        ``{"A": (15, 69), "B": (70, 104), "B1": (105, 118), ...}``
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
            is_section_marker = all(
                p.startswith('*>') and
                not p.startswith('*>[') and
                not p.startswith('*>norep')
                for p in parts
            )
            if is_section_marker:
                section = parts[0][2:]  # Extract "A" from "*>A"
                if section:
                    if current_section is not None:
                        section_ranges[current_section] = (current_start, i)
                    current_section = section
                    current_start = i + 1

    if current_section is not None:
        section_ranges[current_section] = (current_start, len(lines))

    return section_ranges


def _clean_repeat_barline(line: str) -> str:
    """Remove repeat markers from barline while preserving measure number.

    Humdrum repeat barlines look like:
        ``=5:|!|:``  -> ``=5``
        ``=:|!|:``   -> ``=``
        ``=10!|:``   -> ``=10``
    """
    if not line.startswith('='):
        return line

    parts = line.split('\t')
    cleaned_parts = []

    for part in parts:
        if part.startswith('='):
            cleaned = re.sub(r'[:\|!]+', '', part)
            if not cleaned.startswith('='):
                cleaned = '=' + cleaned.lstrip('=')
            cleaned_parts.append(cleaned)
        else:
            cleaned_parts.append(part)

    return '\t'.join(cleaned_parts)


def _renumber_barlines(lines: List[str]) -> List[str]:
    """Renumber barlines sequentially after expansion.

    Pickup measures (=N-) become =0, subsequent barlines are 1, 2, 3, ...
    Final barlines (==) are preserved as-is.
    """
    result = []
    measure_counter = 0
    first_barline_seen = False

    for line in lines:
        if not line.startswith('='):
            result.append(line)
            continue

        if line.startswith('=='):
            result.append(line)
            continue

        parts = line.split('\t')

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

        new_parts = []
        for part in parts:
            if part.startswith('=') and not part.startswith('=='):
                new_parts.append(f'={new_measure_num}')
            else:
                new_parts.append(part)

        result.append('\t'.join(new_parts))

    return result


def _count_spine_change(line: str) -> int:
    """Count net spine change from a line containing *^ or *v."""
    if not line.startswith('*') or line.startswith('**'):
        return 0

    parts = line.split('\t')
    change = 0
    i = 0
    while i < len(parts):
        if parts[i] == '*^':
            change += 1
        elif parts[i] == '*v':
            merge_count = 0
            while i < len(parts) and parts[i] == '*v':
                merge_count += 1
                i += 1
            change -= (merge_count - 1)
            continue
        i += 1
    return change


def _get_spine_count(line: str) -> int:
    """Get the number of spines (tab-separated fields) in a line."""
    if not line or line.startswith('!'):
        return 0
    return len(line.split('\t'))


# ============================================================================
# Kern-level repeat expansion
# ============================================================================


def expand_kern_repeats(kern_content: str) -> str:
    """Expand kern content according to Humdrum expansion labels.

    This function:
    1. Parses the expansion order (``*>[A,A,B,...]``)
    2. Identifies section boundaries (``*>A``, ``*>B``, etc.)
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
    expansion_order = parse_expansion_order(kern_content)
    if not expansion_order:
        return remove_repeat_barlines(kern_content)

    section_ranges = parse_section_ranges(kern_content)
    if not section_ranges:
        return remove_repeat_barlines(kern_content)

    lines = kern_content.split('\n')

    # Find header end (everything before first section marker)
    first_section_start = min(start for start, _ in section_ranges.values())
    header_end = first_section_start - 1

    # Collect header lines (exclude expansion and norep labels)
    header_lines = []
    base_spine_count = 2
    for i, line in enumerate(lines[:header_end]):
        if line.startswith('*>[') or line.startswith('*>norep['):
            continue
        if '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>[') or p.startswith('*>norep[') for p in parts):
                continue
        if line.startswith('**'):
            base_spine_count = len(line.split('\t'))
        header_lines.append(line)

    # Assemble expanded content
    expanded_lines = header_lines.copy()
    current_spine_count = base_spine_count

    for section in expansion_order:
        if section not in section_ranges:
            continue

        start, end = section_ranges[section]

        section_start_spines = base_spine_count
        for line in lines[start:end]:
            if line and '\t' in line and not line.startswith('!'):
                section_start_spines = len(line.split('\t'))
                break

        if current_spine_count > section_start_spines:
            remaining = current_spine_count
            target = section_start_spines
            while remaining > target:
                merge_parts = ['*'] * (remaining - 2) + ['*v', '*v']
                expanded_lines.append('\t'.join(merge_parts))
                remaining -= 1
            current_spine_count = section_start_spines
        elif current_spine_count < section_start_spines:
            split_parts = ['*'] * (current_spine_count - 1) + ['*^']
            while current_spine_count < section_start_spines:
                expanded_lines.append('\t'.join(split_parts))
                current_spine_count += 1
                split_parts = ['*'] * (current_spine_count - 1) + ['*^']

        for line in lines[start:end]:
            if line.strip() and all(p.strip() == '*-' for p in line.split('\t')):
                continue
            if line.strip().startswith('=='):
                continue
            if line.startswith('='):
                line = _clean_repeat_barline(line)
            expanded_lines.append(line)

            if line.startswith('*') and not line.startswith('**'):
                current_spine_count += _count_spine_change(line)

    while current_spine_count > base_spine_count:
        merge_parts = ['*'] * (current_spine_count - 2) + ['*v', '*v']
        expanded_lines.append('\t'.join(merge_parts))
        current_spine_count -= 1

    expanded_lines.append('\t'.join(['=='] * base_spine_count))
    expanded_lines.append('\t'.join(['*-'] * base_spine_count))

    expanded_lines = _renumber_barlines(expanded_lines)

    return '\n'.join(expanded_lines)


def remove_repeat_barlines(kern_content: str) -> str:
    """Remove repeat barlines from kern content without expanding.

    Use this for files without expansion labels but with repeat barlines
    that cause music21 to fail.
    """
    lines = kern_content.split('\n')
    cleaned_lines = []

    for line in lines:
        if re.match(r'^\*>\[.*\]', line) or re.match(r'^\*>norep\[.*\]', line):
            continue
        if line.startswith('*>') and '\t' in line:
            parts = line.split('\t')
            if all(p.startswith('*>') and not p.startswith('*>[') for p in parts):
                continue
        if line.startswith('='):
            line = _clean_repeat_barline(line)
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def get_expansion_info(kern_content: str) -> Dict:
    """Get information about the repeat structure for debugging."""
    expansion_order = parse_expansion_order(kern_content)
    section_ranges = parse_section_ranges(kern_content)

    info: Dict[str, Any] = {
        'has_expansion': expansion_order is not None,
        'expansion_order': expansion_order,
        'sections': section_ranges,
        'section_count': len(section_ranges),
    }

    if expansion_order and section_ranges:
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


def expand_kern_repeats_with_mapping(kern_content: str) -> Tuple[str, Dict]:
    """Expand kern repeats and return measure mapping for visual aux head.

    Expansion is delegated to ``expand_kern_repeats()``.  The mapping is built
    by running ``extract_kern_measures`` on the expanded kern (single source of
    truth for measure boundaries), then annotating each entry with
    section / occurrence / repeat info from the original expansion labels.

    Args:
        kern_content: Raw kern file content with expansion labels

    Returns:
        Tuple of (expanded_kern, repeat_map_dict)
    """
    import tempfile
    from pathlib import Path

    expanded = expand_kern_repeats(kern_content)

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.krn', delete=False
    ) as tmp:
        tmp.write(expanded)
        tmp_path = Path(tmp.name)
    try:
        measures_gt = extract_kern_measures(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    n_measures = len(measures_gt)

    # Try to build rich section annotation from original content
    expansion_order = parse_expansion_order(kern_content)
    section_ranges = parse_section_ranges(kern_content)

    if expansion_order and section_ranges:
        lines = kern_content.split('\n')

        section_barline_counts: Dict[str, int] = {}
        section_original_bars: Dict[str, List[Optional[int]]] = {}
        for section_name, (start, end) in section_ranges.items():
            bar_nums: List[Optional[int]] = []
            for line in lines[start:end]:
                if line.startswith('='):
                    parts = line.split('\t')
                    match = re.match(r'=(\d+)', parts[0])
                    bar_nums.append(int(match.group(1)) if match else None)
            section_barline_counts[section_name] = len(bar_nums)
            section_original_bars[section_name] = bar_nums

        def _kern_ending_type(section_name: str) -> int:
            m = re.match(r'^([A-Za-z]+)(\d+)$', section_name)
            if m:
                base, num = m.group(1), int(m.group(2))
                other = f"{base}{2 if num == 1 else 1}"
                if other in section_ranges:
                    return num
            return 0

        section_annotations: List[Dict[str, Any]] = []
        section_occ: Counter = Counter()

        for section_name in expansion_order:
            if section_name not in section_ranges:
                continue
            occurrence = section_occ[section_name]
            section_occ[section_name] += 1
            ending_type = _kern_ending_type(section_name)

            for orig_bar in section_original_bars.get(section_name, []):
                section_annotations.append({
                    "original_measure": orig_bar,
                    "section": section_name,
                    "occurrence": occurrence,
                    "is_repeat": occurrence > 0,
                    "ending_type": ending_type,
                })

        all_original: Set[int] = set()
        for section_name in set(expansion_order):
            for b in section_original_bars.get(section_name, []):
                if b is not None:
                    all_original.add(b)

        measure_mapping = []
        for i, m_gt in enumerate(measures_gt):
            entry: Dict[str, Any] = {"expanded_measure": m_gt["measure"]}
            if i < len(section_annotations):
                entry.update(section_annotations[i])
            else:
                entry.update({
                    "original_measure": None,
                    "section": "",
                    "occurrence": 0,
                    "is_repeat": False,
                    "ending_type": 0,
                })
            measure_mapping.append(entry)

        return expanded, {
            "has_repeats": True,
            "expansion_order": expansion_order,
            "original_measure_count": len(all_original),
            "expanded_measure_count": n_measures,
            "measures": measure_mapping,
        }

    # Fallback: no expansion labels -> identity mapping
    measure_mapping = []
    for m_gt in measures_gt:
        measure_mapping.append({
            "expanded_measure": m_gt["measure"],
            "original_measure": m_gt["measure"],
            "section": "",
            "occurrence": 0,
            "is_repeat": False,
            "ending_type": 0,
        })

    return expanded, {
        "has_repeats": False,
        "expansion_order": None,
        "original_measure_count": n_measures,
        "expanded_measure_count": n_measures,
        "measures": measure_mapping,
    }


# ============================================================================
# Music21-level repeat expansion (MuseSyn)
# ============================================================================

# Navigation marker types detected from music21
_NAVIGATION_TYPES = (
    m21.repeat.DaCapo,
    m21.repeat.DalSegno,
    m21.repeat.Fine,
    m21.repeat.Segno,
    m21.repeat.Coda,
)


def expand_musesyn_score(
    score: m21.stream.Score,
) -> Tuple[m21.stream.Score, bool]:
    """Expand repeats in a MuseSyn score if present.

    Compares measure counts before/after ``expandRepeats()`` to detect
    all repeat types (barline repeats, DaCapo, DalSegno, Fine, etc.).
    Also removes StaffGroup spanners that crash converter21.

    Args:
        score: music21 Score (already sanitized).

    Returns:
        ``(score, has_repeats)`` -- the expanded score (or original if no
        repeats) and a boolean flag.
    """
    n_before = len(
        score.parts[0].getElementsByClass(m21.stream.Measure)
    )
    expanded = score.expandRepeats()
    n_after = len(
        expanded.parts[0].getElementsByClass(m21.stream.Measure)
    )
    has_repeats = n_after > n_before

    if has_repeats:
        # converter21 crashes on StaffGroup spanners after expandRepeats
        # (KeyError on PartStaff), remove them.
        for sg in list(expanded.getElementsByClass(m21.layout.StaffGroup)):
            expanded.remove(sg)
        return expanded, True

    return score, False


def extract_repeat_structure(score: m21.stream.Score) -> Dict[str, Any]:
    """Extract repeat structure from the original (unexpanded) score.

    Must be called BEFORE ``expandRepeats()``.

    Returns:
        Dictionary with:
        - repeat_barlines: [{measure, direction}, ...]
        - navigation: [{measure, type}, ...]  (DaCapo, DalSegno, Fine, Segno, Coda)
        - volta_brackets: [{measures, number}, ...]
        - orig_measure_numbers: [int, ...]
        - dacapo_measure: int or None  (measure number of DaCapo/DalSegno)
    """
    part = score.parts[0]
    measures = list(part.getElementsByClass(m21.stream.Measure))
    orig_nums = [m.number for m in measures]

    # 1. Repeat barlines
    repeat_barlines: List[Dict[str, Any]] = []
    seen_barlines: Set[Tuple[int, str]] = set()
    for m_obj in measures:
        for bar in [m_obj.leftBarline, m_obj.rightBarline]:
            if bar and isinstance(bar, m21.bar.Repeat):
                key = (m_obj.number, bar.direction)
                if key not in seen_barlines:
                    seen_barlines.add(key)
                    repeat_barlines.append({
                        "measure": m_obj.number,
                        "direction": bar.direction,
                    })

    # 2. Navigation markers (DaCapo, DalSegno, Fine, Segno, Coda)
    navigation: List[Dict[str, Any]] = []
    dacapo_measure: Optional[int] = None
    seen_nav: Set[Tuple[int, str]] = set()
    for p in score.parts:
        for el in p.flatten():
            if isinstance(el, _NAVIGATION_TYPES):
                m_num = getattr(el, "measureNumber", None)
                if m_num is None:
                    continue
                type_name = el.__class__.__name__
                key = (m_num, type_name)
                if key not in seen_nav:
                    seen_nav.add(key)
                    navigation.append({"measure": m_num, "type": type_name})
                    if type_name in ("DaCapo", "DalSegno") and dacapo_measure is None:
                        dacapo_measure = m_num

    # 3. Volta brackets (RepeatBracket spanners)
    volta_brackets: List[Dict[str, Any]] = []
    seen_volta: Set[Tuple[int, ...]] = set()
    for sp in score.spannerBundle:
        if isinstance(sp, m21.spanner.RepeatBracket):
            spanned = sp.getSpannedElements()
            m_nums = sorted(set(
                s.number for s in spanned
                if isinstance(s, m21.stream.Measure)
            ))
            if m_nums:
                key = tuple(m_nums + [sp.number])
                if key not in seen_volta:
                    seen_volta.add(key)
                    try:
                        bracket_num = int(str(sp.number).strip().rstrip("."))
                    except (ValueError, TypeError):
                        bracket_num = 0
                    volta_brackets.append({
                        "measures": m_nums,
                        "number": bracket_num,
                    })

    return {
        "repeat_barlines": repeat_barlines,
        "navigation": navigation,
        "volta_brackets": volta_brackets,
        "orig_measure_numbers": orig_nums,
        "dacapo_measure": dacapo_measure,
    }


def _get_ending_type(
    measure_num: int, volta_brackets: List[Dict[str, Any]]
) -> int:
    """Return volta bracket number (1=first ending, 2=second ending, 0=none)."""
    for vb in volta_brackets:
        if measure_num in vb["measures"]:
            return vb["number"]
    return 0


def _build_rich_mapping(
    repeat_structure: Dict[str, Any],
    expanded_score: m21.stream.Score,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a rich expanded->original measure mapping from music21 Scores.

    Handles three cases:
    1. Simple barline repeats: expanded numbers have duplicates.
    2. Pure DaCapo (no barline repeats): sequential renumbered measures.
    3. Mixed (repeats + DaCapo): first pass has duplicates, DaCapo pass
       has sequential renumbered measures.

    Returns:
        (measure_mapping, expansion_order)
    """
    orig_nums = repeat_structure["orig_measure_numbers"]
    dacapo_measure = repeat_structure["dacapo_measure"]
    volta_brackets = repeat_structure["volta_brackets"]
    orig_max = max(orig_nums) if orig_nums else 0

    exp_measures = list(
        expanded_score.parts[0].getElementsByClass(m21.stream.Measure)
    )
    exp_nums = [m.number for m in exp_measures]

    has_duplicates = len(exp_nums) != len(set(exp_nums))

    # Determine DaCapo split point in expanded sequence.
    dacapo_split: Optional[int] = None
    if dacapo_measure is not None:
        threshold = dacapo_measure
    elif not has_duplicates and len(exp_nums) > len(orig_nums):
        threshold = orig_max
    else:
        threshold = None

    if threshold is not None:
        for i, n in enumerate(exp_nums):
            if n > threshold:
                dacapo_split = i
                break

    # --- Build first pass mapping ---
    first_pass_nums = exp_nums[:dacapo_split] if dacapo_split else exp_nums
    occurrence_global: Dict[int, int] = {}
    mapping: List[Dict[str, Any]] = []
    first_pass_orig_sequence: List[int] = []

    section_idx = 0
    prev_num = -1

    for i, n in enumerate(first_pass_nums):
        if n < prev_num:
            section_idx += 1
        prev_num = n

        occ = occurrence_global.get(n, 0)
        occurrence_global[n] = occ + 1
        ending = _get_ending_type(n, volta_brackets)

        base_section = chr(ord("A") + section_idx)
        if ending > 0:
            section = f"{base_section}{ending}"
        else:
            section = base_section

        mapping.append({
            "original_measure": n,
            "section": section,
            "occurrence": occ,
            "is_repeat": occ > 0,
            "ending_type": ending,
        })
        first_pass_orig_sequence.append(n)

    # --- Build DaCapo pass mapping (if present) ---
    if dacapo_split is not None:
        dacapo_nums = exp_nums[dacapo_split:]
        dacapo_offset = dacapo_nums[0] - 1 if dacapo_nums else 0

        for i, n in enumerate(dacapo_nums):
            if i < len(first_pass_orig_sequence):
                orig_n = first_pass_orig_sequence[i]
            else:
                orig_n = n - dacapo_offset

            occ = occurrence_global.get(orig_n, 0)
            occurrence_global[orig_n] = occ + 1
            ending = _get_ending_type(orig_n, volta_brackets)

            section_idx += 1 if i == 0 else 0
            base_section = chr(ord("A") + section_idx)
            if ending > 0:
                section = f"{base_section}{ending}"
            else:
                section = base_section

            mapping.append({
                "original_measure": orig_n,
                "section": section,
                "occurrence": occ,
                "is_repeat": occ > 0,
                "ending_type": ending,
            })

    # Derive expansion_order from section sequence (deduplicate consecutive)
    expansion_order: List[str] = []
    for m in mapping:
        s = m["section"]
        if not expansion_order or expansion_order[-1] != s:
            expansion_order.append(s)

    return mapping, expansion_order


def _align_with_kern_measures(
    mapping: List[Dict[str, Any]],
    measures_gt: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Align music21-based mapping with kern-based extract_kern_measures output.

    ``measures_gt`` is the ground truth for measure count and numbering
    (from kern).  The mapping was built from music21's expanded score.
    """
    aligned: List[Dict[str, Any]] = []
    n_gt = len(measures_gt)
    n_map = len(mapping)

    for i, m_gt in enumerate(measures_gt):
        entry: Dict[str, Any] = {"expanded_measure": m_gt["measure"]}
        if i < n_map:
            entry["original_measure"] = mapping[i]["original_measure"]
            entry["section"] = mapping[i]["section"]
            entry["occurrence"] = mapping[i]["occurrence"]
            entry["is_repeat"] = mapping[i]["is_repeat"]
            entry["ending_type"] = mapping[i]["ending_type"]
        elif aligned:
            # Beyond mapping range (e.g. tie resolution after final barline).
            # Inherit from the last aligned entry.
            prev = aligned[-1]
            entry["original_measure"] = prev["original_measure"]
            entry["section"] = prev["section"]
            entry["occurrence"] = prev["occurrence"]
            entry["is_repeat"] = prev["is_repeat"]
            entry["ending_type"] = prev["ending_type"]
        else:
            entry["original_measure"] = m_gt["measure"]
            entry["section"] = ""
            entry["occurrence"] = 0
            entry["is_repeat"] = False
            entry["ending_type"] = 0

        aligned.append(entry)

    if n_map != n_gt:
        logger.warning(
            f"Measure count mismatch: music21={n_map} vs kern={n_gt}. "
            f"Alignment may be imprecise."
        )

    return aligned


def build_musesyn_repeat_map(
    repeat_structure: Dict[str, Any],
    expanded_score: m21.stream.Score,
    measures_gt: List[Dict[str, Any]],
    has_repeats: bool,
    original_measure_count: int,
) -> Dict[str, Any]:
    """Build a complete repeat_map for a MuseSyn file.

    Combines rich mapping, kern alignment, and original markers into
    the final repeat_map dictionary.

    Args:
        repeat_structure: Output of ``extract_repeat_structure()``.
        expanded_score: The score after ``expand_musesyn_score()``.
        measures_gt: Output of ``extract_kern_measures()`` on the cleaned kern.
        has_repeats: Whether the score had repeats.
        original_measure_count: Number of measures in the original score.

    Returns:
        Complete repeat_map dictionary (same format as HumSyn, plus
        ``original_markers``).
    """
    if has_repeats:
        rich_mapping, expansion_order = _build_rich_mapping(
            repeat_structure, expanded_score
        )
        aligned = _align_with_kern_measures(rich_mapping, measures_gt)
    else:
        aligned = []
        for m_gt in measures_gt:
            aligned.append({
                "expanded_measure": m_gt["measure"],
                "original_measure": m_gt["measure"],
                "section": "A",
                "occurrence": 0,
                "is_repeat": False,
                "ending_type": 0,
            })
        expansion_order = None

    original_markers = {
        "repeat_barlines": repeat_structure["repeat_barlines"],
        "navigation": repeat_structure["navigation"],
        "volta_brackets": repeat_structure["volta_brackets"],
    }

    return {
        "has_repeats": has_repeats,
        "expansion_order": expansion_order,
        "original_measure_count": original_measure_count,
        "expanded_measure_count": len(measures_gt),
        "original_markers": original_markers,
        "measures": aligned,
    }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Kern-level
    "expand_kern_repeats",
    "expand_kern_repeats_with_mapping",
    "remove_repeat_barlines",
    "has_expansion_labels",
    "parse_expansion_order",
    "parse_section_ranges",
    "get_expansion_info",
    # Music21-level (MuseSyn)
    "expand_musesyn_score",
    "extract_repeat_structure",
    "build_musesyn_repeat_map",
]
