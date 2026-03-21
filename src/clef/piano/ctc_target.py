"""CTC target sequence builder for hierarchical decoding.

Builds the CTC target sequence from kern text:
    <bar> <dur1> <dur2> ... <dur_k> <bar> <dur1> ... <bar>

Each <dur_i> corresponds to one time step (<nl> row), and is the minimum
duration across all spines that have a new onset at that time step.
Sustain tokens (.) are skipped. Rests (4r, 8r, ...) are included.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from .tokenizer import (
    DURATION_TOKENS, SPECIAL_TOKENS, VOCAB,
    KernTokenizer,
)


# Beat values for each duration string (in quarter notes).
# Used to find the minimum duration (shortest = smallest beat value).
_DUR_BEATS: dict[str, float] = {
    "00": 16.0, "0": 8.0,
    "1": 4.0, "2": 2.0, "4": 1.0, "8": 0.5, "16": 0.25,
    "32": 0.125, "64": 0.0625, "128": 0.03125,
    "00.": 24.0, "0.": 12.0,
    "1.": 6.0, "2.": 3.0, "4.": 1.5, "8.": 0.75,
    "16.": 0.375, "32.": 0.1875, "64.": 0.09375,
    "3": 4/3, "6": 2/3, "12": 1/3, "24": 1/6, "48": 1/12, "96": 1/24,
    "20": 0.2, "40": 0.1, "112": 1/28, "176": 1/44,
    "8q": 0.5, "16q": 0.25,
}

# Regex to extract duration prefix from a kern note/rest token.
# Handles optional tie-start bracket: [4E -> 4, 2.a- -> 2., 8r -> 8
_DUR_RE = re.compile(r'^\[?(\d+\.?(?:q)?)')


def _extract_dur(token: str) -> Optional[str]:
    """Extract duration string from a kern token. Returns None for sustain (.)."""
    token = token.strip()
    if not token or token == '.' or token == '_':
        return None
    m = _DUR_RE.match(token)
    if m:
        dur = m.group(1)
        if dur in _DUR_BEATS:
            return dur
    return None


def _row_min_dur(row_tokens: List[str]) -> Optional[str]:
    """Given the kern tokens of one row, return the minimum duration string.

    Splits chords (space-separated within a spine) and takes the minimum
    duration across all spines/notes that have a new onset.
    Returns None if all tokens are sustain (no new onset this row).
    """
    best_dur: Optional[str] = None
    best_beats: float = float('inf')

    for tok in row_tokens:
        # Each tab-separated spine may contain chord notes (space-separated)
        for note in tok.split():
            dur = _extract_dur(note)
            if dur is None:
                continue
            beats = _DUR_BEATS.get(dur, float('inf'))
            if beats < best_beats:
                best_beats = beats
                best_dur = dur

    return best_dur


def build_ctc_target(kern_text: str, tokenizer: KernTokenizer) -> List[int]:
    """Build CTC target token ID sequence from kern text.

    Format: <bar> dur1 dur2 ... durk <bar> dur1 ... <bar>

    Args:
        kern_text: Raw **kern file content.
        tokenizer: KernTokenizer instance (for vocab lookup).

    Returns:
        List of token IDs. Duration token IDs are looked up from tokenizer.vocab.
        <bar> token ID is SPECIAL_TOKENS["<bar>"].
    """
    bar_id = SPECIAL_TOKENS["<bar>"]
    vocab = tokenizer.vocab

    # Map duration string -> rest token ID (e.g. "8" -> vocab["8r"])
    dur_to_id: dict[str, int] = {}
    for dur in DURATION_TOKENS:
        rest_tok = dur + "r"
        if rest_tok in vocab:
            dur_to_id[dur] = vocab[rest_tok]

    result: List[int] = []
    in_music = False       # True after first barline
    pending_bar = False    # True: next data row should flush <bar> first

    for line in kern_text.splitlines():
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('!'):
            continue

        # Meta/interpretation lines: skip (schema not in CTC target)
        if stripped.startswith('*'):
            continue

        # Barline
        if stripped.startswith('='):
            if pending_bar:
                result.append(bar_id)
            pending_bar = True
            in_music = True
            continue

        if not in_music:
            continue

        # Data row: flush pending <bar> first
        if pending_bar:
            result.append(bar_id)
            pending_bar = False

        # Split by tab into spine tokens and find minimum duration
        spine_tokens = stripped.split('\t')
        min_dur = _row_min_dur(spine_tokens)

        if min_dur is None:
            continue  # all sustain, skip

        if min_dur in dur_to_id:
            result.append(dur_to_id[min_dur])
        # else: OOV duration, skip (shouldn't happen with valid kern)

    # Do NOT flush a trailing pending bar (e.g. final == double barline).
    # It has no measure content — the piece ends at the last duration token.

    return result


def ctc_target_to_str(target_ids: List[int], tokenizer: KernTokenizer) -> str:
    """Convert CTC target ID sequence back to readable string."""
    id_to_tok = {v: k for k, v in tokenizer.vocab.items()}
    return ' '.join(id_to_tok.get(tid, f'?{tid}') for tid in target_ids)


def build_ctc_vocab(vocab: Dict[str, int]) -> Tuple[Dict[int, int], int]:
    """Build compact CTC vocab from full tokenizer vocab.

    CTC ID 0 = blank.
    CTC IDs 1..N = <bar> + standalone duration tokens (rhythm skeleton only, no schema).

    Returns:
        full_to_ctc: mapping from full vocab ID -> compact CTC ID
        ctc_vocab_size: total size (N+1, including blank)
    """
    ctc_full_ids = set()
    ctc_full_ids.add(SPECIAL_TOKENS["<bar>"])
    for dur in DURATION_TOKENS:
        if dur in vocab:
            ctc_full_ids.add(vocab[dur])
    sorted_ids = sorted(ctc_full_ids)
    full_to_ctc = {full_id: ctc_id + 1 for ctc_id, full_id in enumerate(sorted_ids)}
    return full_to_ctc, len(sorted_ids) + 1  # +1 for blank


def labels_to_ctc_targets(
    labels: torch.Tensor,           # [B, S] full vocab IDs (padded with 0)
    full_to_ctc: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract CTC targets from label sequences by filtering to CTC vocab tokens.

    Returns:
        targets: [sum_of_lengths]  1-D tensor of compact CTC IDs
        lengths: [B]               number of CTC tokens per sequence
    """
    ctc_set = set(full_to_ctc.keys())
    targets_list: List[int] = []
    lengths: List[int] = []
    for b in range(labels.size(0)):
        ctc_seq = [full_to_ctc[t] for t in labels[b].tolist() if t in ctc_set]
        targets_list.extend(ctc_seq)
        lengths.append(len(ctc_seq))
    return (
        torch.tensor(targets_list, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


# Module-level constants (computed once at import from global VOCAB)
CTC_FULL_TO_COMPACT, CTC_VOCAB_SIZE = build_ctc_vocab(VOCAB)
CTC_BAR_IDX: int = CTC_FULL_TO_COMPACT[SPECIAL_TOKENS["<bar>"]]

# All non-blank CTC channel indices (= all of 1..CTC_VOCAB_SIZE-1).
# Schema removed from CTC vocab, so every non-blank channel is rhythmic.
CTC_RHYTHMIC_INDICES: List[int] = list(range(1, CTC_VOCAB_SIZE))
