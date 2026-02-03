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

Vocabulary Design:
    - Special tokens: <pad>, <sos>, <eos>, <unk>, <coc>, etc.
    - Duration tokens: 1, 2, 4, 8, 16, 32, 64, 128, dotted, triplets
    - Pitch tokens: CC, C, c, cc, ccc (octave notation) with accidentals
    - Modifiers: ties ([, ], _), grace notes (q, Q, P)

Benefits of Factorized Encoding:
    - Small vocab (~512) vs single-token (~9000+)
    - Compositional generalization
    - Rare combinations don't become <unk>
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
    "<unk>": 3,
    "<coc>": 4,      # Change of Column (multi-track separator)
    "<bar>": 5,      # Bar line
    "<b>": 6,        # Beat boundary
    "<continue>": 7, # Chunk boundary (piece continues in next chunk)
    "<nl>": 8,       # Newline (line separator within measure)
    "<split>": 9,    # Spine split (*^)
    "<merge>": 10,   # Spine merge (*v)
    "<*>": 11,       # Null interpretation (* in split/merge lines)
}

# Duration tokens (Zeng vocab compatible)
DURATION_TOKENS = [
    # Binary
    "0", "00",                          # breve, longa
    "1", "2", "4", "8", "16", "32", "64", "128",
    # Dotted
    "0.", "00.",
    "1.", "2.", "4.", "8.", "16.", "32.", "64.",
    # Triplets
    "3", "6", "12", "24", "48", "96",
    # Extended (quintuplets, septuplets, etc.)
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
        self._pitch_pattern = re.compile(
            r'([A-Ga-g]+[#\-n]*|r)'  # Pitch: letters with optional accidentals (#, -, n), or rest
        )
        self._tie_start_pattern = re.compile(r'^[\[({]+')
        self._tie_end_pattern = re.compile(r'[\])}]+$')
        self._grace_pattern = re.compile(r'[qQP]')
        self._beam_pattern = re.compile(r'[LJKk]+$')

        # Build duration and pitch sets for validation
        self._valid_durations = set(DURATION_TOKENS)
        self._valid_pitches = set(PITCH_TOKENS)

    def tokenize_kern_token(self, token: str) -> List[str]:
        """Tokenize a single kern token into factorized components.

        Args:
            token: A single kern token like "4c#", "[8.G", "16r"

        Returns:
            List of factorized tokens: ["4", "c#"], ["[", "8.", "G"], etc.
        """
        if not token or token == ".":
            return ["."]

        # Handle <unk> tokens
        if token.startswith("<unk>"):
            # Extract pitch part after <unk>
            pitch_part = token[5:]  # Remove "<unk>"
            if pitch_part:
                # Try to extract pitch
                pitch_match = self._pitch_pattern.search(pitch_part)
                if pitch_match:
                    pitch = pitch_match.group(1)
                    return ["<unk>", pitch]
            return ["<unk>"]

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

        # 4. Extract pitch (from anywhere in remaining)
        pitch_token = None
        pitch_match = self._pitch_pattern.search(remaining)
        if pitch_match:
            pitch_token = pitch_match.group(1)
            remaining = remaining[:pitch_match.start()] + remaining[pitch_match.end():]

        # 5. Extract duration (from what's left — may be at start or after
        #    removing pitch/grace from non-standard orderings like "a8q")
        dur_match = re.search(r'(\d+\.?)', remaining)
        if dur_match:
            duration = dur_match.group(1)
            result.append(duration)
            remaining = remaining[:dur_match.start()] + remaining[dur_match.end():]

        # 6. Emit grace note marker (after duration, before pitch)
        if grace_token:
            result.append(grace_token)

        # 7. Emit pitch
        if pitch_token:
            result.append(pitch_token)

        # 8. Extract trailing tie markers from what's left
        tie_end_match = self._tie_end_pattern.search(remaining)
        if tie_end_match:
            for char in tie_end_match.group():
                result.append(char)
            remaining = remaining[:tie_end_match.start()]

        # If nothing was extracted, return as unknown
        if not result:
            return ["<unk>"]

        return result

    def tokenize_line(self, line: str) -> List[str]:
        """Tokenize a line of kern data.

        Handles tab-separated spines and special lines (comments, interpretations).

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

        # Handle spine split/merge interpretation lines
        if line.startswith("*"):
            fields = line.split("\t")
            has_split_merge = any(f in ("*^", "*v") for f in fields)
            if not has_split_merge:
                # Regular interpretation (clef, key sig, etc.) — skip
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
            notes = spine.split()

            for j, note in enumerate(notes):
                # Tokenize each note
                tokens = self.tokenize_kern_token(note)
                result.extend(tokens)

        return result

    def tokenize(self, kern_content: str) -> List[str]:
        """Tokenize entire kern content.

        Inserts <nl> between consecutive data lines within a measure.
        <bar> tokens already imply a line break, so <nl> is NOT inserted
        before or after <bar>.

        Args:
            kern_content: Full kern file content or sequence

        Returns:
            List of all tokens (factorized)
        """
        result = ["<sos>"]
        prev_was_data = False  # Was the previous emitted line a data line?

        for line in kern_content.split("\n"):
            line_tokens = self.tokenize_line(line)
            if not line_tokens:
                continue

            is_bar = (line_tokens == ["<bar>"])

            # Insert <nl> between consecutive data lines within a measure
            if prev_was_data and not is_bar:
                result.append("<nl>")

            result.extend(line_tokens)
            prev_was_data = not is_bar

        result.append("<eos>")
        return result

    def encode(self, kern_content: str) -> List[int]:
        """Encode kern content to token IDs.

        Args:
            kern_content: Kern file content or sequence

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(kern_content)
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]

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
            token = self.id_to_token.get(tid, "<unk>")
            if skip_special and token in ["<pad>", "<sos>", "<eos>", "<continue>"]:
                continue
            tokens.append(token)

        # Reconstruct kern format
        return self._reconstruct_kern(tokens)

    def _reconstruct_kern(self, tokens: List[str]) -> str:
        """Reconstruct kern string from factorized tokens.

        This is the inverse of tokenization - combines duration and pitch
        back into single kern tokens. Handles <nl> as line separator.
        """
        result = []
        current_note = []

        for token in tokens:
            if token == "<nl>":
                # Line separator: flush and insert newline
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\n")
            elif token == "<coc>":
                # End current note, add tab
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\t")
            elif token == "<bar>":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append("\n=\n")
            elif token == ".":
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                result.append(".")
            elif token in self._valid_durations:
                # Start new note with duration
                if current_note:
                    result.append("".join(current_note))
                current_note = [token]
            elif token in self._valid_pitches:
                # Add pitch to current note
                current_note.append(token)
            elif token in ["[", "]", "(", ")", "{", "}", "q", "Q", "P", ";"]:
                # Modifiers
                current_note.append(token)
            elif token in ("<split>", "<merge>", "<*>"):
                # Spine structure tokens — emit as kern interpretation
                if current_note:
                    result.append("".join(current_note))
                    current_note = []
                if token == "<split>":
                    result.append("*^")
                elif token == "<merge>":
                    result.append("*v")
                else:
                    result.append("*")
            elif token == "<unk>":
                if current_note:
                    result.append("".join(current_note))
                current_note = ["<unk>"]
            else:
                # Unknown token
                current_note.append(token)

        # Don't forget last note
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

    # Count unknowns
    unk_id = tokenizer.vocab["<unk>"]
    unk_count = token_ids.count(unk_id)

    # Token frequency
    from collections import Counter
    token_freq = Counter(tokens)

    return {
        "filepath": str(filepath),
        "total_tokens": len(tokens),
        "unique_tokens": len(token_freq),
        "unknown_count": unk_count,
        "unknown_rate": unk_count / len(tokens) if tokens else 0,
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
        "<unk>een",
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
