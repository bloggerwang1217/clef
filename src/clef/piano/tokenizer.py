"""
Kern Tokenizer - Factorized Encoding for **kern format
=======================================================

Converts **kern tokens to factorized representation where duration and pitch
are separate tokens. This follows Zeng et al.'s approach for vocabulary efficiency.

Example:
    "4c#" -> ["4", "c#"]      (quarter note C#)
    "8.G"  -> ["8.", "G"]     (dotted eighth G)
    "16r"  -> ["16", "r"]     (sixteenth rest)
    "[4E"  -> ["[", "4", "E"] (tie start + quarter E)

Schema-first design (Mamba decoder):
    After each <bar>, schema tokens are injected:
    <bar> <4> 4 <key:3b> 8E- <coc> 8B- ...
           ^  ^    ^
    numerator | key signature
         denominator (reuses existing duration token)

Vocabulary Design:
    - Special tokens: <pad>, <sos>, <eos>, <coc>, <bar>, etc.
    - Schema tokens: <2>~<17> (numerator), <key:0>~<key:7#>/<key:1b>~<key:7b>
    - Duration tokens: 1, 2, 4, 8, 16, 32, 64, 128, dotted, triplets
    - Pitch tokens: CC, C, c, cc, ccc (octave notation) with accidentals
    - Modifiers: ties ([, ], _), grace notes (q, Q, P)

Benefits of Factorized Encoding:
    - Small vocab (~512) vs single-token (~9000+)
    - Compositional generalization
    - Rare combinations don't need fallback tokens
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
    "<b>": 5,        # Beat boundary
    "<continue>": 6, # Chunk boundary (piece continues in next chunk)
    "<nl>": 7,       # Newline (line separator within measure)
    "<split>": 8,    # Spine split (*^)
    "<merge>": 9,    # Spine merge (*v)
    "<*>": 10,       # Null interpretation (* in split/merge lines)
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

# Modifier tokens
MODIFIER_TOKENS = [
    # Ties
    "[",    # Tie start
    "]",    # Tie end
    "_",    # Tie continue (not commonly used in kern)

    # Grace notes
    "q",    # Short appoggiatura
    "Q",    # Long appoggiatura
    "P",    # Acciaccatura (grace note)

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
    """Build vocabulary mapping.

    Returns:
        Tuple of (token_to_id, id_to_token) dictionaries
    """
    token_to_id = {}
    idx = 0

    # Special tokens first
    for token, special_idx in SPECIAL_TOKENS.items():
        token_to_id[token] = special_idx
        idx = max(idx, special_idx + 1)

    # Schema tokens (meter numerators + key signatures)
    for token in METER_NUMERATOR_TOKENS:
        token_to_id[token] = idx
        idx += 1
    for token in KEY_TOKENS:
        token_to_id[token] = idx
        idx += 1

    # Structural tokens
    for token in STRUCTURAL_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # Duration tokens
    for token in DURATION_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # Pitch tokens
    for token in PITCH_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # Modifier tokens
    for token in MODIFIER_TOKENS:
        if token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    # Reverse mapping
    id_to_token = {v: k for k, v in token_to_id.items()}

    return token_to_id, id_to_token


# Build global vocab
VOCAB, ID_TO_TOKEN = build_vocab()
VOCAB_SIZE = len(VOCAB)


# =============================================================================
# Tokenizer Class
# =============================================================================

class KernTokenizer:
    """Tokenizer for **kern format with factorized encoding.

    Converts kern tokens like "4c#" into factorized form ["4", "c#"].

    Example:
        >>> tokenizer = KernTokenizer()
        >>> tokenizer.tokenize_token("4c#")
        ['4', 'c#']
        >>> tokenizer.encode("4c# 8d")
        [token_ids...]
        >>> tokenizer.decode([...])
        "4c# 8d"
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """Initialize tokenizer.

        Args:
            vocab: Optional custom vocabulary. Uses default if None.
        """
        self.vocab = vocab or VOCAB
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

        # Build duration and pitch sets for validation
        self._valid_durations = set(DURATION_TOKENS)
        self._valid_pitches = set(PITCH_TOKENS)

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
        # Split at boundaries: after pitch/accidental/rest, before digit (new duration)
        parts = re.split(r'(?<=[A-Ga-gr#\-n])(?=\d)', clean)
        # Keep only parts that contain a pitch character (discard stray numbers)
        result = [p for p in parts if re.search(r'[A-Ga-gr]', p)]
        return result if result else [clean]

    def tokenize_kern_token(self, token: str) -> List[str]:
        """Tokenize a single kern token into factorized components.

        Handles malformed kern data: concatenated chord notes (e.g. '24ffd#'),
        natural accidentals (e.g. 'bbn'), and conflicting accidentals (e.g. 'ccc#-').

        Args:
            token: A single kern token like "4c#", "[8.G", "16r"

        Returns:
            List of factorized tokens: ["4", "c#"], ["[", "8.", "G"], etc.
        """
        if not token or token == ".":
            return ["."]

        result = []
        remaining = token

        # 1. Extract leading tie/phrase markers
        tie_start_match = self._tie_start_pattern.match(remaining)
        if tie_start_match:
            for char in tie_start_match.group():
                result.append(char)
            remaining = remaining[tie_start_match.end():]

        # 2. Strip beam markers early (L, J, K, k) - never included in output
        remaining = self._beam_pattern.sub('', remaining)

        # 3. Extract grace note marker (q, Q, P) from anywhere
        grace_match = self._grace_pattern.search(remaining)
        grace_token = None
        if grace_match:
            grace_token = grace_match.group()
            remaining = remaining[:grace_match.start()] + remaining[grace_match.end():]

        # 4. Extract ALL valid pitches using backreference regex.
        #    This handles concatenated chord notes like 'ffd#' -> ['ff', 'd#']
        #    where the greedy [A-Ga-g]+ would incorrectly match 'ffd#' as one pitch.
        pitches = []
        for pm in self._single_pitch_re.finditer(remaining):
            pitch = self._clean_pitch(pm.group(0))
            if pitch:
                pitches.append(pitch)
        # Remove all pitch material from remaining to isolate duration
        remaining = self._single_pitch_re.sub('', remaining)

        # 5. Extract duration (from what's left — may be at start or after
        #    removing pitch/grace from non-standard orderings like "a8q")
        dur_match = re.search(r'(\d+\.?)', remaining)
        duration = None
        if dur_match:
            duration = dur_match.group(1)
            remaining = remaining[:dur_match.start()] + remaining[dur_match.end():]

        # 6. Emit tokens: [duration] [grace] pitch for each pitch found.
        #    For concatenated chords (multiple pitches), duration is repeated.
        for i, pitch in enumerate(pitches):
            if i == 0:
                if duration:
                    result.append(duration)
                if grace_token:
                    result.append(grace_token)
            else:
                # Chord note from concatenated data: repeat duration
                if duration:
                    result.append(duration)
            result.append(pitch)

        # 7. If no pitches found but we have a duration, emit just the duration
        if not pitches and duration:
            result.append(duration)

        # 8. Extract trailing tie markers from what's left
        tie_end_match = self._tie_end_pattern.search(remaining)
        if tie_end_match:
            for char in tie_end_match.group():
                result.append(char)

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
            notes_raw = spine.split()

            for note in notes_raw:
                # Split concatenated notes (data quality fix: e.g. '4r4G-' -> ['4r', '4G-'])
                split_notes = self._split_kern_token(note)
                for sn in split_notes:
                    tokens = self.tokenize_kern_token(sn)
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
        # Emit initial schema right after <sos>
        result.extend(self._make_schema_tokens())

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
            if skip_special and token in ["<pad>", "<sos>", "<eos>", "<continue>"]:
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
            elif token in self._valid_durations:
                if current_note:
                    result.append("".join(current_note))
                current_note = [token]
            elif token in self._valid_pitches:
                current_note.append(token)
            elif token in ["[", "]", "(", ")", "{", "}", "q", "Q", "P", ";"]:
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
