"""
Kern Tokenizer - Compound Word Encoding for **kern format
==========================================================

Converts **kern tokens to compound representation where each note event
is a single atomic token (duration+pitch combined), following the
Compound Word Transformer (Hsiao et al. 2021) design.

Example:
    "4c#" -> ["4c#"]         (quarter note C# — single compound token)
    "8.G"  -> ["8.G"]        (dotted eighth G)
    "16r"  -> ["16r"]        (sixteenth rest)
    "[4E"  -> ["[", "4E"]    (tie start modifier + compound note)

Embedding Design (CWT-style):
    embed("4c#") = pitch_embed("c#") + dur_embed("4")
    embed(".")   = other_embed(".")   (struct token)
    embed("<bar>") = other_embed("<bar>")  (struct token)

    Weight-tied output logits:
    logit("dur_i pitch_j") = h · (pitch_embed[j] + dur_embed[i])^T
                           = h·pitch_embed[j]^T + h·dur_embed[i]^T

Vocabulary Design:
    - Special tokens (IDs 0-10): <pad>, <sos>, <eos>, <coc>, <bar>, etc.
    - Note tokens (IDs 11..11+N_NOTE-1): all dur×pitch compound tokens
    - Schema tokens: <2>~<17> (numerator), <key:0>~<key:7#>/<key:1b>~<key:7b>
    - Duration tokens (standalone): schema denominators (2, 4, 8, ...)
    - Structural/modifier tokens: ., [, ], q, Q, P, etc.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


# =============================================================================
# Vocabulary Definition
# =============================================================================

# Special tokens
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<coc>": 3,      # Change of Column (multi-track separator)
    "<bar>": 4,      # Bar line
    "<cont>": 5,     # Chunk boundary (piece continues in next chunk)
    "<nl>": 6,       # Newline (line separator within measure)
    "<split>": 7,    # Spine split (*^)
    "<merge>": 8,    # Spine merge (*v)
    "<*>": 9,        # Null interpretation (* in split/merge lines)
    "<prev>": 10,    # Non-first chunk start (piece started in a previous chunk)
}

# Schema tokens: time signature numerators (beats per measure)
# Denominator reuses existing duration tokens (2, 4, 8, 16, 32)
METER_NUMERATOR_TOKENS = [
    "<2>", "<3>", "<4>", "<5>", "<6>", "<7>",
    "<8>", "<9>", "<10>", "<12>", "<17>",
]

# Schema tokens: key signatures (circle of fifths)
KEY_TOKENS = [
    "<key:0>",                                          # C major / A minor
    "<key:1#>", "<key:2#>", "<key:3#>", "<key:4#>",    # sharps
    "<key:5#>", "<key:6#>", "<key:7#>",
    "<key:1b>", "<key:2b>", "<key:3b>", "<key:4b>",    # flats
    "<key:5b>", "<key:6b>", "<key:7b>",
]

# Mapping from kern key signature to token
_KEY_SIG_TO_TOKEN = {
    '*k[]': '<key:0>',
    '*k[f#]': '<key:1#>', '*k[f#c#]': '<key:2#>', '*k[f#c#g#]': '<key:3#>',
    '*k[f#c#g#d#]': '<key:4#>', '*k[f#c#g#d#a#]': '<key:5#>',
    '*k[f#c#g#d#a#e#]': '<key:6#>', '*k[f#c#g#d#a#e#b#]': '<key:7#>',
    '*k[b-]': '<key:1b>', '*k[b-e-]': '<key:2b>', '*k[b-e-a-]': '<key:3b>',
    '*k[b-e-a-d-]': '<key:4b>', '*k[b-e-a-d-g-]': '<key:5b>',
    '*k[b-e-a-d-g-c-]': '<key:6b>', '*k[b-e-a-d-g-c-f-]': '<key:7b>',
}

# Reverse mapping: token to kern key signature
_TOKEN_TO_KEY_SIG = {v: k for k, v in _KEY_SIG_TO_TOKEN.items()}


def normalize_key_sig(key_sig: str) -> str:
    """Normalize non-standard key signatures (e.g. Chopin with explicit naturals).

    Strips natural signs and maps to standard circle-of-fifths form.
    Examples:
        '*k[bnenf#c#]' -> '*k[f#c#]'  (D major with explicit naturals)
        '*k[cancel]'   -> '*k[]'       (cancellation marker)
    """
    if key_sig in _KEY_SIG_TO_TOKEN:
        return key_sig
    if key_sig == '*k[cancel]':
        return '*k[]'
    # Strip natural signs (n, n-) and reconstruct
    inside = key_sig[3:-1]  # extract content between *k[ and ]
    # Keep only sharps and flats
    accidentals = []
    i = 0
    while i < len(inside):
        ch = inside[i]
        if ch in 'abcdefg':
            # Check next char
            if i + 1 < len(inside) and inside[i + 1] == '#':
                accidentals.append(f'{ch}#')
                i += 2
            elif i + 1 < len(inside) and inside[i + 1] == '-':
                accidentals.append(f'{ch}-')
                i += 2
            else:
                # Natural (no accidental) — skip
                # Also handle 'n' or 'n-' after letter
                if i + 1 < len(inside) and inside[i + 1] == 'n':
                    i += 2
                    if i < len(inside) and inside[i] == '-':
                        i += 1  # skip 'n-'
                else:
                    i += 1
        else:
            i += 1
    return f'*k[{"".join(accidentals)}]'

# Duration tokens (Zeng vocab only — OOV durations handled by rhythm_quantizer.py)
DURATION_TOKENS = [
    # Binary
    "0", "00",                          # breve, longa
    "1", "2", "4", "8", "16", "32", "64", "128",
    # Dotted
    "0.", "00.",
    "1.", "2.", "4.", "8.", "16.", "32.", "64.",
    # Triplets
    "3", "6", "12", "24", "48", "96",
    # Extended (Zeng vocab: quintuplet, septuplet, 11-tuplet)
    "20", "40", "112", "176",
]

# Grace note durations — only included in vocab when include_grace_notes=True.
# All grace durations are normalised to one of these two canonical values.
GRACE_DURATION_TOKENS = ["8q", "16q"]

# Pitch tokens - Humdrum kern notation
# Octave notation: CC=C2, C=C3, c=C4, cc=C5, ccc=C6, cccc=C7
PITCH_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
ACCIDENTALS = ["-", "", "#"]  # flat, none, sharp (Zeng-compatible, no n/##/--)

def generate_pitch_tokens() -> List[str]:
    """Generate all valid pitch tokens in Humdrum kern notation."""
    pitches = []

    # Sub-contra octave (C0): CCCC, DDDD, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter}{letter}{letter}{letter}{acc}")

    # Contra octave (C1): CCC, DDD, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter}{letter}{letter}{acc}")

    # Great octave (C2): CC, DD, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter}{letter}{acc}")

    # Small octave (C3): C, D, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter}{acc}")

    # One-line octave (C4): c, d, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter.lower()}{acc}")

    # Two-line octave (C5): cc, dd, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter.lower()}{letter.lower()}{acc}")

    # Three-line octave (C6): ccc, ddd, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter.lower()}{letter.lower()}{letter.lower()}{acc}")

    # Four-line octave (C7): cccc, dddd, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter.lower()}{letter.lower()}{letter.lower()}{letter.lower()}{acc}")

    # Five-line octave (C8): ccccc, ddddd, etc.
    for letter in PITCH_LETTERS:
        for acc in ACCIDENTALS:
            pitches.append(f"{letter.lower()}{letter.lower()}{letter.lower()}{letter.lower()}{letter.lower()}{acc}")

    # Rest
    pitches.append("r")

    return pitches

PITCH_TOKENS = generate_pitch_tokens()


# =============================================================================
# Compound Note Tokens (CWT-style)
# =============================================================================

def generate_note_tokens() -> List[str]:
    """Generate compound note tokens: all (duration, pitch) combinations.

    Ordering: dur[i] × pitch[j] → index i*N_PITCH + j.
    This ordering is required by CWTEmbedding's weight-tied output.
    """
    tokens = []
    for dur in DURATION_TOKENS:
        for pitch in PITCH_TOKENS:
            tokens.append(dur + pitch)
    return tokens


NOTE_TOKENS: List[str] = generate_note_tokens()
NOTE_START_ID: int = max(SPECIAL_TOKENS.values()) + 1  # = 11 (right after SPECIAL)
N_DUR:   int = len(DURATION_TOKENS)   # number of duration types
N_PITCH: int = len(PITCH_TOKENS)      # number of pitch types (including rest)
N_NOTE:  int = N_DUR * N_PITCH        # total compound note tokens

# Lookup: compound note string → (dur_index, pitch_index) within their tables
_PITCH_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(PITCH_TOKENS)}
_DUR_TO_IDX:   Dict[str, int] = {d: i for i, d in enumerate(DURATION_TOKENS)}
# Sorted durations by length descending for unambiguous prefix parsing
_DUR_SORTED: List[str] = sorted(DURATION_TOKENS, key=len, reverse=True)


def _parse_note_token(token: str) -> Tuple[str, str]:
    """Parse compound note token into (duration, pitch) strings.

    Returns:
        (dur_str, pitch_str) such that dur_str + pitch_str == token.

    Raises:
        ValueError if token is not a valid compound note.
    """
    for dur in _DUR_SORTED:
        if token.startswith(dur):
            pitch = token[len(dur):]
            if pitch in _PITCH_TO_IDX:
                return dur, pitch
    raise ValueError(f"Cannot parse compound note token: {token!r}")


# Kern pitch to MIDI mapping
_KERN_BASE_SEMITONES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def kern_pitch_to_midi(token: str) -> int:
    """Convert a kern pitch token to MIDI number.

    Uppercase = octave 3 and below, lowercase = octave 4 and above.
    Examples: CCCC=12(C0), CCC=24(C1), CC=36(C2), C=48(C3),
              c=60(C4), cc=72(C5), ccc=84(C6), cccc=96(C7), ccccc=108(C8)
    Accidentals: # = +1, - = -1.
    Returns -1 for rest (r) or unrecognized tokens.
    """
    if not token or token == 'r':
        return -1

    # Strip accidentals from the end
    accidental = 0
    core = token
    while core.endswith('#'):
        accidental += 1
        core = core[:-1]
    while core.endswith('-'):
        accidental -= 1
        core = core[:-1]

    if not core:
        return -1

    letter = core[0].upper()
    if letter not in _KERN_BASE_SEMITONES:
        return -1

    base = _KERN_BASE_SEMITONES[letter]

    if core[0].isupper():
        # Uppercase: C=48(oct3), CC=36(oct2), CCC=24(oct1), CCCC=12(oct0)
        repeat = len(core)
        midi = (5 - repeat) * 12 + base
    else:
        # Lowercase: c=60(oct4), cc=72(oct5), ccc=84(oct6), cccc=96(oct7)
        repeat = len(core)
        midi = (4 + repeat) * 12 + base

    return midi + accidental

# Modifier tokens
MODIFIER_TOKENS = [
    # Ties
    "[",    # Tie start
    "]",    # Tie end
    "_",    # Tie continue (not commonly used in kern)

    # Grace notes: q/Q/P are now incorporated into compound note duration ("8q"/"16q"),
    # so they are NOT emitted as standalone modifier tokens.

    # Phrase/slur markers (often in kern)
    "(",    # Phrase/slur start
    ")",    # Phrase/slur end
    "{",    # Phrase start
    "}",    # Phrase end

    # Beam markers (informational, may keep or strip)
    "L",    # Beam start
    "J",    # Beam end

    # Fermata
    ";",
]

# Structural tokens
STRUCTURAL_TOKENS = [
    ".",    # Placeholder (null token)
    "\t",   # Column separator
    "\n",   # Line separator (implicit in sequence)
]


def build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary mapping (compound note token scheme).

    Vocab ordering (required by CWTEmbedding weight-tied output):
        0  .. 10           : SPECIAL tokens (fixed IDs)
        11 .. 11+N_NOTE-1  : NOTE tokens (compound dur+pitch, dur-major order)
        11+N_NOTE ..       : other tokens (standalone dur, meter, key, struct, modifier)

    Returns:
        Tuple of (token_to_id, id_to_token) dictionaries
    """
    token_to_id = {}

    # 1. Special tokens (fixed IDs 0-10)
    for token, special_idx in SPECIAL_TOKENS.items():
        token_to_id[token] = special_idx
    idx = NOTE_START_ID  # = 11

    # 2. Note tokens (compound) — ordered dur[i]*N_PITCH + pitch[j]
    for token in NOTE_TOKENS:
        token_to_id[token] = idx
        idx += 1

    # 3. Standalone duration tokens (used as schema denominators: *M4/4 → "4")
    for token in DURATION_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # 4. Schema tokens (meter numerators + key signatures)
    for token in METER_NUMERATOR_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1
    for token in KEY_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # 5. Structural tokens (., \t, \n)
    for token in STRUCTURAL_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # 6. Modifier tokens ([, ], q, Q, P, etc.)
    for token in MODIFIER_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    id_to_token = {v: k for k, v in token_to_id.items()}
    return token_to_id, id_to_token


# Build global vocab
VOCAB, ID_TO_TOKEN = build_vocab()
VOCAB_SIZE = len(VOCAB)


def build_cwt_info(vocab: Dict[str, int]):
    """Build CWTEmbedding dispatch tensors from a vocabulary.

    Returns a dict with:
        n_dur, n_pitch, n_note, n_other, n_special, note_start_id : int
        is_note        : BoolTensor  [vocab_size] — True for note tokens
        note_dur_idx   : LongTensor  [vocab_size] — dur table index (0..N_DUR-1)
        note_pitch_idx : LongTensor  [vocab_size] — pitch table index (0..N_PITCH-1)
        other_idx      : LongTensor  [vocab_size] — other_embed index for non-note tokens

    other_embed has n_other = vocab_size - N_NOTE entries, laid out as:
        indices 0..n_special-1 → SPECIAL tokens (vocab IDs 0..n_special-1)
        indices n_special..    → non-special non-note tokens (in vocab-ID order)
    """
    import torch as _torch
    vocab_size = max(vocab.values()) + 1
    n_special = len(SPECIAL_TOKENS)  # = NOTE_START_ID = 11

    is_note        = _torch.zeros(vocab_size, dtype=_torch.bool)
    note_dur_idx   = _torch.zeros(vocab_size, dtype=_torch.long)
    note_pitch_idx = _torch.zeros(vocab_size, dtype=_torch.long)
    other_idx      = _torch.zeros(vocab_size, dtype=_torch.long)

    note_set = set(NOTE_TOKENS)
    for token, token_id in vocab.items():
        if token in note_set:
            is_note[token_id] = True
            dur, pitch = _parse_note_token(token)
            note_dur_idx[token_id]   = _DUR_TO_IDX[dur]
            note_pitch_idx[token_id] = _PITCH_TO_IDX[pitch]

    # Build other_idx: contiguous indices for all non-note tokens
    # Layout: SPECIAL first (IDs 0..n_special-1), then the rest in vocab-ID order
    other_tokens_sorted = sorted(
        [(tid, tok) for tok, tid in vocab.items() if not is_note[tid]],
        key=lambda x: x[0]
    )
    for local_idx, (token_id, _) in enumerate(other_tokens_sorted):
        other_idx[token_id] = local_idx
    n_other = len(other_tokens_sorted)

    return {
        'n_dur':        N_DUR,
        'n_pitch':      N_PITCH,
        'n_note':       N_NOTE,
        'n_other':      n_other,
        'n_special':    n_special,
        'note_start_id': NOTE_START_ID,
        'is_note':        is_note,
        'note_dur_idx':   note_dur_idx,
        'note_pitch_idx': note_pitch_idx,
        'other_idx':      other_idx,
    }


# Pre-computed CWT info for the default vocabulary
CWT_INFO = build_cwt_info(VOCAB)


# =============================================================================
# Tokenizer Class
# =============================================================================

class KernTokenizer:
    """Tokenizer for **kern format with compound word encoding.

    Converts kern tokens like "4c#" into compound form ["4c#"].

    Example:
        >>> tokenizer = KernTokenizer()
        >>> tokenizer.tokenize_kern_token("4c#")
        ['4c#']
        >>> tokenizer.encode("4c# 8d")
        [token_ids...]

    Example:
        >>> tokenizer = KernTokenizer()
        >>> tokenizer.tokenize_token("4c#")
        ['4', 'c#']
        >>> tokenizer.encode("4c# 8d")
        [token_ids...]
        >>> tokenizer.decode([...])
        "4c# 8d"
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None,
                 include_grace_notes: bool = False):
        """Initialize tokenizer.

        Args:
            vocab: Optional custom vocabulary. Uses default if None.
            include_grace_notes: If True, encode grace notes as compound tokens
                ("8q"/"16q" + pitch). If False (default), skip grace notes entirely —
                consistent with Zeng and a2s-transformer baselines.
        """
        self.vocab = vocab or VOCAB
        self.include_grace_notes = include_grace_notes
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Compile regex patterns
        self._duration_pattern = re.compile(
            r'^(\d+\.?)'  # Duration: digits optionally followed by dot
        )
        # Valid kern pitch: same letter repeated + optional accidentals.
        # Uses backreference to ensure all octave letters are identical.
        self._single_pitch_re = re.compile(
            r'(([A-G])\2*|([a-g])\3*)[#\-n]*|r'
        )
        self._tie_start_pattern = re.compile(r'^[\[({]+')
        self._tie_end_pattern = re.compile(r'[\])}]+$')
        self._grace_pattern = re.compile(r'[qQP]')
        self._beam_pattern = re.compile(r'[LJKk]+$')

        # Build token-type sets for validation and reconstruction
        self._valid_durations = set(DURATION_TOKENS)
        self._valid_pitches = set(PITCH_TOKENS)
        self._valid_notes = set(NOTE_TOKENS)  # compound note tokens

    @staticmethod
    def _clean_pitch(pitch: str) -> str:
        """Clean a pitch token: strip naturals, resolve conflicting accidentals.

        Zeng-compatible vocabulary does not include natural accidentals (n).
        Conflicting sharps+flats (data errors) are resolved by keeping the first.
        """
        if pitch == 'r':
            return pitch
        pitch = pitch.replace('n', '')
        if not pitch:
            return ''
        # Resolve conflicting accidentals (e.g. ccc#- -> ccc#)
        if '#' in pitch and '-' in pitch:
            sharp_pos = pitch.index('#')
            flat_pos = pitch.index('-')
            if sharp_pos < flat_pos:
                pitch = pitch.replace('-', '')
            else:
                pitch = pitch.replace('#', '')
        return pitch

    def _split_kern_token(self, token: str) -> List[str]:
        """Split concatenated kern notes that each have their own duration.

        Handles data quality issues where notes lack space separation:
            '4r4G-'  -> ['4r', '4G-']     (rest + note concatenated)
            '16c#0'  -> ['16c#']           (stray '0' discarded)
            '24ffd#' -> ['24ffd#']         (pitch splitting handled in tokenize_kern_token)
        """
        clean = self._beam_pattern.sub('', token)
        # Split at note boundaries: after pitch/accidental, before a digit that starts a
        # NEW note (next segment has digit + pitch letter).
        # This avoids splitting pitch-then-duration tokens like 'dd8q' → ['dd', '8q'].
        parts = re.split(r'(?<=[A-Ga-gr#\-n])(?=\d+\.?[A-Ga-gr])', clean)
        # Strip trailing stray digits after a pitch/accidental (data quality: '16c#0' → '16c#').
        parts = [re.sub(r'(?<=[A-Ga-gr#\-n])\d+\.?$', '', p) for p in parts]
        # Keep only parts that contain a pitch character (discard stray numbers)
        result = [p for p in parts if re.search(r'[A-Ga-gr]', p)]
        return result if result else [clean]

    def tokenize_kern_token(self, token: str, default_duration: Optional[str] = None) -> List[str]:
        """Tokenize a single kern token into compound form.

        Each note event becomes one atomic compound token (duration+pitch).
        Prefix modifiers ([, q, Q, P) are emitted as separate preceding tokens.
        Trailing tie markers (], _, )) are emitted as separate following tokens.

        Handles malformed kern data: concatenated chord notes (e.g. '24ffd#'),
        natural accidentals (e.g. 'bbn'), and conflicting accidentals (e.g. 'ccc#-').

        Args:
            token: A single kern token like "4c#", "[8.G", "16r"

        Returns:
            Compound tokens: ["4c#"], ["[", "8.G"], ["4c#", "]"], etc.
        """
        if not token or token == ".":
            return ["."]

        result = []
        remaining = token

        # 1. Extract leading tie/phrase markers ([, (, {)
        tie_start_match = self._tie_start_pattern.match(remaining)
        if tie_start_match:
            for char in tie_start_match.group():
                result.append(char)
            remaining = remaining[tie_start_match.end():]

        # 2. Strip beam markers (L, J, K, k) — never included in output
        remaining = self._beam_pattern.sub('', remaining)

        # 3. Extract grace note marker (q, Q, P).
        # Grace notes are encoded as compound note tokens with duration "8q" or "16q"
        # (all grace durations normalised to one of these two).
        #
        # Special cases in training data:
        #   '8qqe'  — double-q (long appoggiatura); only the first q is consumed here,
        #             the second q stays in `remaining` but is ignored by pitch/dur parsing.
        #   '12qfff#' — triplet-quarter grace (12 < 16) → maps to "8q".
        grace_match = self._grace_pattern.search(remaining)
        is_grace = grace_match is not None
        if is_grace and not self.include_grace_notes:
            return []
        if grace_match:
            remaining = remaining[:grace_match.start()] + remaining[grace_match.end():]

        # 4. Extract trailing tie markers (], ), })
        trailing_markers = []
        tie_end_match = self._tie_end_pattern.search(remaining)
        if tie_end_match:
            for char in tie_end_match.group():
                trailing_markers.append(char)
            remaining = remaining[:tie_end_match.start()]

        # 4b. Remove stray digits embedded between a pitch letter and its accidental.
        # Kern pitch tokens never contain digits; numbers are duration-only.
        # Data quality: 'A0-' → 'A-', '8A0-' → '8A-'.
        remaining = re.sub(r'([A-Ga-g])\d+([#\-n])', r'\1\2', remaining)

        # 5. Extract ALL valid pitches (handles concatenated chord notes)
        pitches = []
        for pm in self._single_pitch_re.finditer(remaining):
            pitch = self._clean_pitch(pm.group(0))
            if pitch:
                pitches.append(pitch)
        remaining = self._single_pitch_re.sub('', remaining)

        # 6. Extract duration; fall back to chord-inherited duration if absent.
        dur_match = re.search(r'(\d+\.?)', remaining)
        duration = dur_match.group(1) if dur_match else default_duration

        # 6b. Normalise grace note duration to "8q" or "16q".
        # All grace durations map to one of two canonical values:
        #   numeric part >= 16 (16th or shorter) → "16q"
        #   numeric part <  16 (8th or longer)   → "8q"
        #   durationless grace                   → "8q"
        if is_grace:
            if duration is not None:
                num = int(re.match(r'\d+', duration).group())
                duration = "16q" if num >= 16 else "8q"
            else:
                duration = "8q"

        # 7. Emit compound note tokens (one per pitch, duration repeated for chords)
        for pitch in pitches:
            note = (duration or "") + pitch
            if note in self._valid_notes:
                result.append(note)
            else:
                # Fallback: emit standalone duration + pitch (OOV compound)
                if duration:
                    result.append(duration)
                result.append(pitch)

        # 8. If no pitches but have duration (schema denominator or bare duration)
        if not pitches and duration:
            result.append(duration)

        result.extend(trailing_markers)

        if not result:
            raise ValueError(f"Cannot tokenize kern token: {token!r}")

        return result

    def tokenize_line(self, line: str) -> List[str]:
        """Tokenize a line of kern data.

        Handles tab-separated spines and special lines (comments, interpretations).
        Time signature and key signature interpretation lines update internal state
        (self._current_time_sig, self._current_key_sig) but don't emit tokens
        directly — schema tokens are injected after <bar> by tokenize().

        Args:
            line: A line from kern file

        Returns:
            List of tokens (factorized)
        """
        line = line.strip()

        # Skip empty lines
        if not line:
            return []

        # Skip comments
        if line.startswith("!"):
            return []

        # Handle interpretation lines
        if line.startswith("*"):
            fields = line.split("\t")

            # Check for time signature (*M4/4, *M3/8, etc.)
            for field in fields:
                if field.startswith("*M") and '/' in field:
                    self._current_time_sig = field  # e.g. '*M4/4'
                    break

            # Check for key signature (*k[f#c#], *k[], etc.)
            for field in fields:
                if field.startswith("*k["):
                    self._current_key_sig = normalize_key_sig(field)
                    break

            # Handle spine split/merge
            has_split_merge = any(f in ("*^", "*v") for f in fields)
            if not has_split_merge:
                return []

            # Tokenize spine split/merge: *^ -> <split>, *v -> <merge>, * -> <*>
            result = []
            for i, field in enumerate(fields):
                if i > 0:
                    result.append("<coc>")
                if field == "*^":
                    result.append("<split>")
                elif field == "*v":
                    result.append("<merge>")
                else:
                    result.append("<*>")
            return result

        # Handle bar lines
        if line.startswith("="):
            return ["<bar>"]

        result = []

        # Split by tabs (spines)
        spines = line.split("\t")

        for i, spine in enumerate(spines):
            # Add column separator between spines
            if i > 0:
                result.append("<coc>")

            # Handle chord (space-separated notes in same spine)
            # Kern convention: subsequent chord notes may omit duration (inherits from first note).
            notes_raw = spine.split()
            chord_duration: Optional[str] = None  # inherited duration for this spine token

            for note in notes_raw:
                # Split concatenated notes (data quality fix: e.g. '4r4G-' -> ['4r', '4G-'])
                split_notes = self._split_kern_token(note)

                # Pre-scan: find any explicit duration in the split parts for forward/backward
                # propagation within this concatenated token (handles 'A4e' -> ['A', '4e']
                # where the duration only appears in a later part).
                local_duration = chord_duration
                for sn in split_notes:
                    stripped = re.sub(r'^[\[({qQP]+', '', sn)
                    dm = re.search(r'\d+\.?', stripped)
                    if dm:
                        local_duration = dm.group(0)
                        break

                for sn in split_notes:
                    # Update chord_duration from this note if it carries an explicit duration.
                    stripped = re.sub(r'^[\[({qQP]+', '', sn)
                    dur_m = re.search(r'\d+\.?', stripped)
                    if dur_m:
                        chord_duration = dur_m.group(0)
                    tokens = self.tokenize_kern_token(sn, default_duration=local_duration)
                    result.extend(tokens)

        return result

    def _make_schema_tokens(self) -> List[str]:
        """Build schema token sequence from current time_sig and key_sig state.

        Returns tokens like: ['<4>', '4', '<key:0>']
        for *M4/4 and *k[].
        """
        tokens = []

        # Time signature: extract numerator and denominator
        if self._current_time_sig:
            # Parse *M4/4 -> numerator=4, denominator=4
            m = re.match(r'\*M(\d+)/(\d+)', self._current_time_sig)
            if m:
                numerator = m.group(1)
                denominator = m.group(2)
                num_token = f'<{numerator}>'
                if num_token in VOCAB:
                    tokens.append(num_token)
                    tokens.append(denominator)  # reuse duration token

        # Key signature
        if self._current_key_sig:
            key_token = _KEY_SIG_TO_TOKEN.get(self._current_key_sig)
            if key_token:
                tokens.append(key_token)

        return tokens

    def tokenize(self, kern_content: str) -> List[str]:
        """Tokenize entire kern content.

        Schema-first: after each <bar>, injects schema tokens (time sig + key sig).
        Initial schema is emitted right after <sos>.
        <nl> separates consecutive data lines within a measure.

        Args:
            kern_content: Full kern file content or sequence

        Returns:
            List of all tokens (factorized)
        """
        # Reset schema state for this tokenization
        self._current_time_sig = None
        self._current_key_sig = None

        # First pass: scan for initial time_sig and key_sig before first barline
        # (interpretation lines appear before first measure)
        lines = kern_content.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("="):
                break  # stop at first barline
            if stripped.startswith("*"):
                fields = stripped.split("\t")
                for field in fields:
                    if field.startswith("*M") and '/' in field and not self._current_time_sig:
                        self._current_time_sig = field
                    if field.startswith("*k[") and not self._current_key_sig:
                        self._current_key_sig = normalize_key_sig(field)

        result = ["<sos>"]
        # No initial schema here: the leading <bar> (present in all chunks
        # and full pieces) triggers schema injection via _make_schema_tokens(),
        # so the first measure's schema always appears after the first <bar>.

        prev_was_data = False  # Was the previous emitted line a data line?

        for line in lines:
            line_tokens = self.tokenize_line(line)
            if not line_tokens:
                continue

            is_bar = (line_tokens == ["<bar>"])

            # Insert <nl> between consecutive data lines within a measure
            if prev_was_data and not is_bar:
                result.append("<nl>")

            result.extend(line_tokens)

            # After <bar>, inject schema tokens
            if is_bar:
                result.extend(self._make_schema_tokens())

            prev_was_data = not is_bar

        result.append("<eos>")
        return result

    def encode(self, kern_content: str, strict: bool = True) -> List[int]:
        """Encode kern content to token IDs.

        Args:
            kern_content: Kern file content or sequence
            strict: If True (default), raise ValueError on OOV tokens.
                    If False, raise ValueError but caller can catch it.

        Returns:
            List of token IDs

        Raises:
            ValueError: If any token is not in vocabulary
        """
        tokens = self.tokenize(kern_content)
        ids = []
        for t in tokens:
            if t not in self.vocab:
                raise ValueError(f"Token {t!r} not in vocabulary")
            ids.append(self.vocab[t])
        return ids

    def is_duration_id(self, token_id: int) -> bool:
        """Check if a token ID is a standalone duration token (schema denominator)."""
        token = self.id_to_token.get(token_id, "")
        return token in self._valid_durations

    def is_note_id(self, token_id: int) -> bool:
        """Check if a token ID is a compound note token (dur+pitch)."""
        token = self.id_to_token.get(token_id, "")
        return token in self._valid_notes

    def is_pitch_id(self, token_id: int) -> bool:
        """Check if a token ID represents a pitch/note event.

        Returns True for compound note tokens (the new primary note representation).
        """
        return self.is_note_id(token_id)

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to kern format.

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens like <sos>, <eos>

        Returns:
            Reconstructed kern string
        """
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, f"<UNK:{tid}>")
            if skip_special and token in ["<pad>", "<sos>", "<eos>", "<cont>", "<prev>"]:
                continue
            tokens.append(token)

        # Reconstruct kern format
        return self._reconstruct_kern(tokens)

    def _reconstruct_kern(self, tokens: List[str]) -> str:
        """Reconstruct kern string from factorized tokens.

        This is the inverse of tokenization - combines duration and pitch
        back into single kern tokens. Schema tokens after <bar> are consumed
        to track state but not emitted as note content.
        """
        # Schema token sets for detection
        _numerator_set = set(METER_NUMERATOR_TOKENS)
        _key_set = set(KEY_TOKENS)

        result = []
        current_note = []
        # State machine: after <bar>, consume schema tokens before note data
        consuming_schema = False
        schema_expect_denom = False  # after numerator, expect denominator

        for token in tokens:
            # Schema consumption: after <bar> (or at start), eat schema tokens
            if consuming_schema:
                if token in _numerator_set:
                    schema_expect_denom = True
                    continue
                if schema_expect_denom and token in self._valid_durations:
                    schema_expect_denom = False
                    continue
                if token in _key_set:
                    continue
                # Not a schema token — stop consuming, process normally
                consuming_schema = False
                schema_expect_denom = False

            if token in _numerator_set or token in _key_set:
                # Schema token outside of post-bar context (e.g. after <sos>)
                consuming_schema = True
                if token in _numerator_set:
                    schema_expect_denom = True
                continue

            if token == "<nl>":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\n")
            elif token == "<coc>":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\t")
            elif token == "<bar>":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\n=\n")
                consuming_schema = True
                schema_expect_denom = False
            elif token == ".":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append(".")
            elif token in self._valid_notes:
                # Compound note token: flush any accumulated prefix, emit note
                current_note.append(token)
                result.append("".join(current_note))
                current_note = []
            elif token in self._valid_durations:
                # Standalone duration (schema denominator — shouldn't normally reach here)
                current_note.append(token)
            elif token in ["[", "(", "{"]:
                # Prefix modifiers: accumulate before the compound note
                current_note.append(token)
            elif token in ["]", ")", "}", "_", ";"]:
                # Trailing modifiers: append to last emitted kern token
                if result:
                    result[-1] = result[-1] + token
                else:
                    current_note.append(token)
            elif token in ("<split>", "<merge>", "<*>"):
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                if token == "<split>":
                    result.append("*^")
                elif token == "<merge>":
                    result.append("*v")
                else:
                    result.append("*")
            else:
                current_note.append(token)

        if current_note:
            result.append("".join(current_note))

        # Smart join: space between note tokens, no space around structural chars
        STRUCTURAL = ("\t", "\n", "\n=\n")
        output = []
        for i, item in enumerate(result):
            if i > 0 and item not in STRUCTURAL and result[i - 1] not in STRUCTURAL:
                output.append(" ")
            output.append(item)
        return "".join(output)

    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def get_special_token_ids(self) -> Dict[str, int]:
        """Return special token IDs."""
        return {k: self.vocab[k] for k in SPECIAL_TOKENS.keys()}


# =============================================================================
# Utility Functions
# =============================================================================

def analyze_kern_file(filepath: Path, tokenizer: Optional[KernTokenizer] = None) -> Dict:
    """Analyze a kern file and return statistics.

    Args:
        filepath: Path to kern file
        tokenizer: Optional tokenizer instance

    Returns:
        Dictionary with token counts, unknown tokens, etc.
    """
    if tokenizer is None:
        tokenizer = KernTokenizer()

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    tokens = tokenizer.tokenize(content)
    token_ids = tokenizer.encode(content)

    # Token frequency
    from collections import Counter
    token_freq = Counter(tokens)

    return {
        "filepath": str(filepath),
        "total_tokens": len(tokens),
        "unique_tokens": len(token_freq),
        "top_tokens": token_freq.most_common(20),
    }


# =============================================================================
# Main: Demo and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Kern Tokenizer Demo")
    print("=" * 60)

    tokenizer = KernTokenizer()
    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")

    # Test cases
    test_tokens = [
        "4c#",
        "8.G",
        "16r",
        "[4E",
        "4d]",
        "8qf#",
        "24ggLL",
        "2.BB-",
    ]

    print("\n--- Token Factorization ---")
    for token in test_tokens:
        factorized = tokenizer.tokenize_kern_token(token)
        print(f"  {token:15} -> {factorized}")

    # Test full line
    print("\n--- Line Tokenization ---")
    test_line = "4E 4G 4B- 4c\t4r"
    tokens = tokenizer.tokenize_line(test_line)
    print(f"  Input: {test_line}")
    print(f"  Tokens: {tokens}")

    # Test encode/decode
    print("\n--- Encode/Decode ---")
    test_content = """4c 4e 4g
8d 8f
"""
    tokens = tokenizer.tokenize(test_content)
    print(f"  Tokens: {tokens}")

    ids = tokenizer.encode(test_content)
    print(f"  IDs: {ids[:20]}...")

    decoded = tokenizer.decode(ids)
    print(f"  Decoded: {decoded}")
