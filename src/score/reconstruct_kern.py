#!/usr/bin/env python3
"""
Reconstruct **kern from Clef Model Predictions
===============================================

Converts factorized token predictions back to valid **kern format.

Clef model predicts factorized tokens with barlines (<bar>) and column
separators (<coc>). This module reassembles them into proper **kern notation.

Usage:
    from src.score.reconstruct_kern import reconstruct_kern_from_tokens

    tokens = ['4', 'c', '<bar>', '8', 'e', '8', 'g', '<bar>', ...]
    kern_str = reconstruct_kern_from_tokens(tokens)
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Valid duration tokens (Zeng vocab only — must match tokenizer)
VALID_DURATIONS = {
    "0", "00", "1", "2", "4", "8", "16", "32", "64", "128",
    "0.", "00.", "1.", "2.", "4.", "8.", "16.", "32.", "64.",
    "3", "6", "12", "24", "48", "96",
    "20", "40", "112", "176",
}

# Schema token detection
from src.clef.piano.tokenizer import (
    METER_NUMERATOR_TOKENS, KEY_TOKENS,
    _TOKEN_TO_KEY_SIG,
)
_NUMERATOR_SET = set(METER_NUMERATOR_TOKENS)
_KEY_SET = set(KEY_TOKENS)


def _assemble_spine_content(tokens: List[str]) -> str:
    """Assemble factorized tokens into kern spine content (notes/chords).

    Combines consecutive duration + pitch + modifier tokens into kern tokens,
    space-separated within one spine slot.

    A kern note has the structure: [pre-modifiers] duration pitch [post-modifiers]
    For chords (multiple pitches): duration pitch pitch ... (space between pitches)
    
    Pre-modifiers: [ ( { q Q P
    Post-modifiers: ] ) } ;
    """
    PRE_MODIFIERS = {"[", "(", "{", "q", "Q", "P"}
    POST_MODIFIERS = {"]", ")", "}", ";", "_"}

    result = []
    current_note = []
    has_pitch = False  # whether current_note already has a pitch
    has_duration = False  # whether current_note has a duration yet

    for token in tokens:
        if token in VALID_DURATIONS:
            # New duration: flush current_note UNLESS it only has pre-modifiers
            if current_note and has_pitch:
                result.append("".join(current_note))
                current_note = [token]
                has_pitch = False
                has_duration = True
            elif current_note and all(t in PRE_MODIFIERS for t in current_note):
                # Pre-modifiers waiting for duration — keep them
                current_note.append(token)
                has_duration = True
            else:
                if current_note:
                    result.append("".join(current_note))
                current_note = [token]
                has_pitch = False
                has_duration = True
        elif token == 'r' or re.match(r'^[A-Ga-g]+[#\-]*$', token):
            # This is a pitch token
            if has_pitch and has_duration:
                # We already have a pitch for this duration (chord note)
                # Space-separate it from the previous pitch
                result.append("".join(current_note))
                current_note = [token]
                has_pitch = True
                # Keep has_duration = True since this is still the same duration
            else:
                # First pitch for this duration, add to current_note
                current_note.append(token)
                has_pitch = True
        elif token == '.':
            if current_note:
                result.append("".join(current_note))
                current_note = []
                has_pitch = False
                has_duration = False
            result.append(".")
        elif token in PRE_MODIFIERS:
            # Pre-modifier: if we already have a pitch, flush first (new note)
            if has_pitch:
                result.append("".join(current_note))
                current_note = []
                has_pitch = False
                has_duration = False
            current_note.append(token)
        elif token in POST_MODIFIERS:
            current_note.append(token)
        else:
            current_note.append(token)

    if current_note:
        result.append("".join(current_note))

    return " ".join(result)


def extract_kern_metadata(kern_content: str) -> Dict[str, str]:
    """Extract essential metadata from a kern file for reconstruction.

    Extracts interpretation lines that converter21 needs for correct parsing:
    time signature, key signature, clef, instrument, staff assignment.
    Also extracts the starting measure number for proper barline numbering.

    Args:
        kern_content: Original kern file content

    Returns:
        Dict with keys like 'time_sig', 'key_sig', 'clef', 'first_measure_num', etc.
        Each value is a full tab-separated interpretation line (or a number string for first_measure_num).
    """
    metadata = {}
    found_first_measure = False
    
    for line in kern_content.split('\n'):
        stripped = line.strip()
        
        # Extract the first measure number (format: =0, =1, =1-, etc.)
        if not found_first_measure and stripped.startswith('='):
            first_field = stripped.split('\t')[0]
            # Extract just the number part (=0, =1, =1-, etc.)
            match = re.match(r'^=([0-9]+)', first_field)
            if match:
                metadata['first_measure_num'] = match.group(1)
                found_first_measure = True
        
        if not stripped.startswith('*'):
            continue
        # Skip spine structure and terminators
        if stripped.startswith('**') or stripped.startswith('*-'):
            continue
        if stripped.startswith('*^') or stripped.startswith('*v'):
            continue

        first_field = stripped.split('\t')[0]
        if first_field.startswith('*M') and '/' in first_field and 'time_sig' not in metadata:
            metadata['time_sig'] = line
        elif first_field.startswith('*k[') and 'key_sig' not in metadata:
            metadata['key_sig'] = line
        elif first_field.startswith('*clef') and 'clef' not in metadata:
            metadata['clef'] = line
        elif first_field.startswith('*staff') and 'staff' not in metadata:
            metadata['staff'] = line
        elif first_field.startswith('*I') and 'instrument' not in metadata:
            metadata['instrument'] = line
        elif first_field.startswith('*MM') and 'tempo' not in metadata:
            metadata['tempo'] = line

    return metadata


def reconstruct_kern_from_tokens(
    tokens: List[str],
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Reconstruct **kern from factorized tokens.

    Structural tokens:
        <bar>  — measure boundary (barline in all spines)
        <coc>  — column separator (tab between spines within one kern line)
        <nl>   — line separator (newline between kern data lines)

    Token sequence for a 2-spine measure:
        [s1_notes] <coc> [s2_notes] <nl> [s1_notes] <coc> [s2_notes] <bar> ...

    Args:
        tokens: List of factorized tokens
        metadata: Optional dict from extract_kern_metadata(). If provided,
            interpretation lines (time sig, clef, etc.) are injected after
            the **kern header.

    Returns:
        Reconstructed **kern string
    """
    SPINE_TOKENS = {'<split>', '<merge>', '<*>'}
    SPINE_MAP = {'<split>': '*^', '<merge>': '*v', '<*>': '*'}

    # Filter out padding/sos/eos, extract schema tokens, and track changes.
    # Schema changes between bars become interpretation lines in the output.
    filtered = []
    initial_time_sig = None   # first time sig seen (for header)
    initial_key_sig = None    # first key sig seen (for header)
    current_time_sig = None   # current state
    current_key_sig = None
    # Map from bar index (count of <bar> tokens) to schema changes
    bar_schema_changes = {}   # bar_idx -> {'time_sig': ..., 'key_sig': ...}
    bar_count = 0
    consuming_schema = False
    schema_expect_denom = False
    pending_time_sig = None
    pending_key_sig = None
    schema_time_sig_num = None

    def _flush_schema():
        """Record schema changes at current bar.

        Initial schema (bar_count=0) goes into the header via initial_time_sig/
        initial_key_sig. Schema changes at bar_count>0 become interpretation
        lines before the barline. This ensures files without pre-barline
        time/key sigs (e.g. Joplin pickup bars) reconstruct correctly.
        """
        nonlocal pending_time_sig, pending_key_sig, current_time_sig, current_key_sig
        nonlocal initial_time_sig, initial_key_sig
        changes = {}
        if pending_time_sig:
            if pending_time_sig != current_time_sig:
                # Only set initial from schema that appears before the first bar
                if bar_count == 0 and initial_time_sig is None:
                    initial_time_sig = pending_time_sig
                else:
                    changes['time_sig'] = pending_time_sig
                current_time_sig = pending_time_sig
            elif bar_count == 0 and initial_time_sig is None:
                initial_time_sig = pending_time_sig
                current_time_sig = pending_time_sig
        if pending_key_sig:
            if pending_key_sig != current_key_sig:
                if bar_count == 0 and initial_key_sig is None:
                    initial_key_sig = pending_key_sig
                else:
                    changes['key_sig'] = pending_key_sig
                current_key_sig = pending_key_sig
            elif bar_count == 0 and initial_key_sig is None:
                initial_key_sig = pending_key_sig
                current_key_sig = pending_key_sig
        if changes:
            bar_schema_changes[bar_count] = changes
        pending_time_sig = None
        pending_key_sig = None

    for t in tokens:
        if t in ['<pad>', '<sos>', '<eos>']:
            continue

        # Schema consumption: after <bar> or at start
        if consuming_schema:
            if t in _NUMERATOR_SET:
                schema_expect_denom = True
                schema_time_sig_num = t[1:-1]
                continue
            if schema_expect_denom and t in VALID_DURATIONS:
                schema_expect_denom = False
                pending_time_sig = f'*M{schema_time_sig_num}/{t}'
                continue
            if t in _KEY_SET:
                pending_key_sig = _TOKEN_TO_KEY_SIG.get(t, '*k[]')
                continue
            # Not a schema token — flush pending schema and stop consuming
            _flush_schema()
            consuming_schema = False
            schema_expect_denom = False

        if t in _NUMERATOR_SET:
            consuming_schema = True
            schema_expect_denom = True
            schema_time_sig_num = t[1:-1]
            continue
        if t in _KEY_SET:
            consuming_schema = True
            pending_key_sig = _TOKEN_TO_KEY_SIG.get(t, '*k[]')
            continue

        if t == '<bar>':
            if consuming_schema:
                _flush_schema()
            consuming_schema = True
            schema_expect_denom = False
            bar_count += 1

        filtered.append(t)

    # Flush any remaining schema at end
    if consuming_schema:
        _flush_schema()

    # Inject extracted schema into metadata
    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)
    if initial_time_sig and 'time_sig' not in metadata:
        metadata['time_sig'] = initial_time_sig
    if initial_key_sig and 'key_sig' not in metadata:
        metadata['key_sig'] = initial_key_sig

    # Build kern lines. Track current spine count dynamically
    # since <split>/<merge> change it.
    kern_lines = []       # list of (line_str, is_bar_placeholder)
    current_spines = []   # spine contents for current kern line
    current_spine = []    # tokens for current spine
    in_spine_op = False   # are we accumulating a spine operation line?

    def flush_data_line():
        """Flush current data spines into a kern line."""
        nonlocal current_spines, current_spine
        if current_spine or current_spines:
            current_spines.append(_assemble_spine_content(current_spine))
            current_spine = []
            line = "\t".join(current_spines)
            if line.strip():
                kern_lines.append(line)
            current_spines = []

    def flush_spine_op_line():
        """Flush spine operation tokens into an interpretation line."""
        nonlocal current_spines, current_spine, in_spine_op
        if current_spine or current_spines:
            # current_spine holds the last spine op token(s)
            current_spines.append(
                SPINE_MAP.get(current_spine[0], '*') if current_spine else '*'
            )
            current_spine = []
            line = "\t".join(current_spines)
            kern_lines.append(line)
            current_spines = []
        in_spine_op = False

    for token in filtered:
        if token == '<bar>':
            if in_spine_op:
                flush_spine_op_line()
            else:
                flush_data_line()
            kern_lines.append('<BAR_PLACEHOLDER>')
            continue

        if token == '<nl>':
            if in_spine_op:
                flush_spine_op_line()
            else:
                flush_data_line()
            continue

        if token in SPINE_TOKENS:
            if not in_spine_op:
                # Entering a spine operation line — flush any pending data
                flush_data_line()
                in_spine_op = True
            else:
                # Already in spine op — this token is for the current spine
                pass

            # Accumulate: the token IS the spine content
            if current_spine:
                # We had a previous spine op token → flush as a spine
                current_spines.append(SPINE_MAP[current_spine[0]])
                current_spine = []
            current_spine = [token]
            continue

        if token == '<coc>':
            if in_spine_op:
                # <coc> between spine op tokens
                current_spines.append(SPINE_MAP[current_spine[0]])
                current_spine = []
            else:
                current_spines.append(_assemble_spine_content(current_spine))
                current_spine = []
            continue

        # Regular data token
        if in_spine_op:
            # We were in a spine op but got a data token — flush the op first
            flush_spine_op_line()
        current_spine.append(token)

    # Flush remaining
    if in_spine_op:
        flush_spine_op_line()
    else:
        flush_data_line()

    # Determine initial n_spines by finding the first data line BEFORE any
    # spine operation. A split/merge changes the count, so data lines after
    # a split reflect the post-split count, not the initial count.
    n_spines = None
    for line in kern_lines:
        if line == '<BAR_PLACEHOLDER>':
            if n_spines is None:
                # No data line before first bar — look at the barline itself
                # (can't determine from bar, keep searching)
                continue
            break
        if line.startswith('*'):
            # Hit a spine operation before finding a data line —
            # the initial count equals the number of fields in this line
            # (each field is one input spine: *, *^, or *v)
            if n_spines is None:
                n_spines = len(line.split('\t'))
            break
        # Data line
        n_tabs = line.count('\t')
        if n_tabs > 0 and n_spines is None:
            n_spines = n_tabs + 1
    if n_spines is None:
        n_spines = 1

    # Second pass: replace barline placeholders with correct spine count.
    # Track spine count through split/merge operations.
    # Inject interpretation lines when schema changes between bars.
    # Use first_measure_num from metadata if available (e.g. =0 for pickup bars).
    measure_num = int(metadata.get('first_measure_num', '0')) if metadata else 0

    current_n = n_spines
    bar_idx = 0  # counts bars to look up schema changes
    output_lines = []
    for line in kern_lines:
        if line == '<BAR_PLACEHOLDER>':
            bar_idx += 1
            # Inject schema change interpretation lines BEFORE the barline.
            # In kern format, interpretation lines like *k[b-] / *M3/4 appear
            # before the barline they apply to, so the tokenizer reads them
            # before processing the barline (updating state before schema emission).
            if bar_idx in bar_schema_changes:
                changes = bar_schema_changes[bar_idx]
                if 'time_sig' in changes:
                    ts = changes['time_sig']
                    output_lines.append("\t".join([ts] * current_n))
                if 'key_sig' in changes:
                    ks = changes['key_sig']
                    output_lines.append("\t".join([ks] * current_n))
            bar = f"={measure_num}"
            output_lines.append("\t".join([bar] * current_n))
            measure_num += 1
        else:
            output_lines.append(line)
            # Update spine count based on split/merge
            if line.startswith('*'):
                fields = line.split('\t')
                n_out = 0
                i = 0
                while i < len(fields):
                    if fields[i] == '*^':
                        n_out += 2  # split: one spine becomes two
                        i += 1
                    elif fields[i] == '*v':
                        # Count consecutive *v's — they merge together
                        while i < len(fields) and fields[i] == '*v':
                            i += 1
                        n_out += 1  # merged into one
                    else:
                        n_out += 1  # unchanged
                        i += 1
                current_n = n_out

    # Build header
    header_lines = ["\t".join(["**kern"] * n_spines)]

    if metadata:
        for key in ['staff', 'instrument', 'clef', 'key_sig', 'time_sig', 'tempo']:
            if key in metadata:
                header_lines.append(metadata[key])

    footer = "\t".join(["*-"] * n_spines)
    header = "\n".join(header_lines)
    body = "\n".join(output_lines)
    return f"{header}\n{body}\n{footer}\n"


def reconstruct_kern_from_token_ids(
    token_ids: List[int],
    tokenizer,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Reconstruct **kern from token IDs.

    Args:
        token_ids: List of token IDs from model prediction
        tokenizer: KernTokenizer instance
        metadata: Optional dict from extract_kern_metadata()

    Returns:
        Reconstructed **kern string
    """
    tokens = []
    for tid in token_ids:
        token = tokenizer.id_to_token.get(tid, f"<UNK:{tid}>")
        if token in ["<pad>", "<sos>", "<eos>"]:
            continue
        tokens.append(token)

    return reconstruct_kern_from_tokens(tokens, metadata=metadata)


if __name__ == "__main__":
    # Two-spine example with <nl> line separators
    test_tokens = [
        '4', 'r', '<coc>', '4', 'c',
        '<bar>',
        '4', 'C', '<coc>', '8', 'e', '8', 'g',
        '<nl>',
        '.', '<coc>', '8', 'd', '8', 'f',
        '<nl>',
        '8', 'r', '<coc>', '4', 'r',
        '<bar>',
        '2', 'C', '<coc>', '2', 'c',
    ]

    print("Test tokens:", test_tokens)
    print("\nReconstructed kern:")
    print(reconstruct_kern_from_tokens(test_tokens))
