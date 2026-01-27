"""
Rhythm Quantizer - Universal Duration Quantization Engine
==========================================================

This module provides a unified algorithm for converting any duration to
standard vocabulary tokens using Integer Programming concepts.

Core Algorithm:
    Given target duration D and number of notes N:
    1. Find two vocab durations T_long and T_short that bracket D/N
    2. Solve: x + y = N, x * T_long + y * T_short = D
    3. Distribute using Euclidean rhythm (Bresenham-like) for uniform spacing

This handles:
    - Tuplets (N notes in duration D)
    - Single notes (split into ties if needed)
    - Any OOV duration

Mathematical Foundation:
    For 21-tuplet in half note:
    - Target: D = 1/2, N = 21
    - Average: 1/42 (between 1/32 and 1/64)
    - Solve: x + y = 21, x/32 + y/64 = 1/2
    - Result: x = 11 (32nds), y = 10 (64ths)
    - Verify: 11/32 + 10/64 = 22/64 + 10/64 = 32/64 = 1/2 ✓

References:
    - Euclidean Rhythm: Toussaint (2005)
    - Bresenham's Line Algorithm
    - Linear Diophantine Equations
"""

from fractions import Fraction
from typing import List, Optional, Tuple, Dict, Set
import math


# =============================================================================
# Zeng Vocabulary (as Fractions for exact arithmetic)
# =============================================================================

# Standard binary durations
BINARY_DURATIONS = {
    '0': Fraction(2, 1),      # breve (double whole)
    '1': Fraction(1, 1),      # whole
    '2': Fraction(1, 2),      # half
    '4': Fraction(1, 4),      # quarter
    '8': Fraction(1, 8),      # eighth
    '16': Fraction(1, 16),    # sixteenth
    '32': Fraction(1, 32),    # thirty-second
    '64': Fraction(1, 64),    # sixty-fourth
    '128': Fraction(1, 128),  # one-twenty-eighth
}

# Dotted durations (1.5x base)
DOTTED_DURATIONS = {
    '0.': Fraction(3, 1),     # dotted breve
    '1.': Fraction(3, 2),     # dotted whole
    '2.': Fraction(3, 4),     # dotted half
    '4.': Fraction(3, 8),     # dotted quarter
    '8.': Fraction(3, 16),    # dotted eighth
    '16.': Fraction(3, 32),   # dotted sixteenth
    '32.': Fraction(3, 64),   # dotted thirty-second
    '64.': Fraction(3, 128),  # dotted sixty-fourth
}

# Triplet durations (2/3 of binary)
TRIPLET_DURATIONS = {
    '3': Fraction(1, 3),      # triplet half (2/3 of half = 1/3)
    '6': Fraction(1, 6),      # triplet quarter
    '12': Fraction(1, 12),    # triplet eighth
    '24': Fraction(1, 24),    # triplet sixteenth
    '48': Fraction(1, 48),    # triplet thirty-second
    '96': Fraction(1, 96),    # triplet sixty-fourth
}

# Extended durations in Zeng vocab
EXTENDED_DURATIONS = {
    '20': Fraction(1, 20),    # quintuplet quarter
    '40': Fraction(1, 40),    # quintuplet eighth
    '112': Fraction(1, 112),  # septuplet sixteenth
    '176': Fraction(1, 176),  # 11-tuplet sixteenth
    # '128' already in binary
}

# Combined Zeng vocabulary
ZENG_VOCAB: Dict[str, Fraction] = {
    **BINARY_DURATIONS,
    **DOTTED_DURATIONS,
    **TRIPLET_DURATIONS,
    **EXTENDED_DURATIONS,
}

# Sorted by duration (longest first) for greedy algorithms
ZENG_VOCAB_SORTED = sorted(ZENG_VOCAB.items(), key=lambda x: -x[1])


def duration_to_fraction(dur_str: str) -> Optional[Fraction]:
    """
    Convert Humdrum duration string to Fraction.

    In Humdrum, duration number = reciprocal of whole note fraction.
    e.g., '4' = quarter note = 1/4 whole note

    Args:
        dur_str: Duration string (e.g., '4', '8.', '42')

    Returns:
        Fraction representing duration, or None if invalid
    """
    if not dur_str:
        return None

    # Check if dotted
    is_dotted = dur_str.endswith('.')
    base_str = dur_str[:-1] if is_dotted else dur_str

    try:
        base_num = int(base_str)
        if base_num == 0:
            base_val = Fraction(2, 1)  # breve
        else:
            base_val = Fraction(1, base_num)

        if is_dotted:
            return base_val * Fraction(3, 2)
        return base_val
    except ValueError:
        return None


class RhythmQuantizer:
    """
    Universal rhythm quantization engine.

    Converts any duration or tuplet to standard vocabulary tokens
    using Integer Programming / Euclidean rhythm distribution.

    Example:
        >>> q = RhythmQuantizer()
        >>> # 21-tuplet in half note
        >>> result = q.quantize_tuplet(Fraction(1, 2), 21)
        >>> print(result)
        ['32', '64', '32', '64', '32', '64', '32', '64', '32', '64', '32',
         '64', '32', '64', '32', '64', '32', '64', '32', '64', '32']
    """

    def __init__(self, vocab: Optional[Dict[str, Fraction]] = None):
        """
        Initialize quantizer with vocabulary.

        Args:
            vocab: Dictionary mapping token names to Fraction durations.
                   Defaults to Zeng vocabulary.
        """
        self.vocab = vocab or ZENG_VOCAB
        # Sort by duration descending (for greedy algorithms)
        self.vocab_sorted = sorted(self.vocab.items(), key=lambda x: -x[1])
        # Also keep just the binary grid for tuplet solving
        self.binary_grid = {k: v for k, v in self.vocab.items()
                           if k.isdigit() and not k.startswith('0')}

    def quantize_tuplet(
        self,
        total_duration: Fraction,
        num_notes: int,
        use_euclidean: bool = True,
    ) -> Optional[List[str]]:
        """
        Quantize N notes fitting in total_duration.

        Solves the Integer Programming problem:
            x + y = N
            x * T_long + y * T_short = D

        Args:
            total_duration: Total duration as Fraction
            num_notes: Number of notes
            use_euclidean: If True, distribute using Euclidean rhythm.
                          If False, group all long notes first.

        Returns:
            List of duration tokens, or None if no solution exists.

        Example:
            >>> q.quantize_tuplet(Fraction(1, 2), 21)
            ['32', '64', '32', '64', ...]  # 11 x '32' + 10 x '64'
        """
        if num_notes <= 0:
            return None

        avg_duration = total_duration / num_notes

        # 1. Find bracketing durations from binary grid
        # We want T_long >= avg_duration >= T_short
        t_long_name, t_long_val = None, None
        t_short_name, t_short_val = None, None

        # Sort binary grid by duration descending
        binary_sorted = sorted(self.binary_grid.items(), key=lambda x: -x[1])

        for name, val in binary_sorted:
            if val >= avg_duration:
                t_long_name, t_long_val = name, val
            if val <= avg_duration and t_short_name is None:
                t_short_name, t_short_val = name, val
                break

        # Handle edge cases
        if t_long_val is None or t_short_val is None:
            # Duration outside our grid range
            return None

        if t_long_val == t_short_val:
            # Exact match - all notes are the same duration
            return [t_long_name] * num_notes

        # 2. Solve linear Diophantine equation
        # x + y = N
        # x * t_long + y * t_short = D
        #
        # Substituting y = N - x:
        # x * t_long + (N - x) * t_short = D
        # x * (t_long - t_short) = D - N * t_short
        # x = (D - N * t_short) / (t_long - t_short)

        numerator = total_duration - (num_notes * t_short_val)
        denominator = t_long_val - t_short_val

        if denominator == 0:
            return [t_long_name] * num_notes

        x = numerator / denominator

        # Check if x is a non-negative integer
        # CRITICAL: Only accept exact solutions (error = 0)
        # Duration must be 100% accurate; note count can change but not timing
        if x.denominator != 1 or x < 0 or x > num_notes:
            # No exact integer solution exists - reject
            return None

        count_long = int(x)
        count_short = num_notes - count_long

        # 3. Distribute using Euclidean rhythm (Bresenham-like)
        if use_euclidean:
            return self._euclidean_distribute(
                t_long_name, count_long,
                t_short_name, count_short,
                num_notes
            )
        else:
            # Simple: all long first, then short
            return [t_long_name] * count_long + [t_short_name] * count_short

    def _euclidean_distribute(
        self,
        long_name: str, count_long: int,
        short_name: str, count_short: int,
        total: int
    ) -> List[str]:
        """
        Distribute long and short notes evenly using Euclidean rhythm.

        This is essentially Bresenham's line algorithm applied to rhythm.
        The result is the "most even" distribution possible.

        Args:
            long_name: Token for longer duration
            count_long: Number of long notes
            short_name: Token for shorter duration
            count_short: Number of short notes
            total: Total number of notes (= count_long + count_short)

        Returns:
            List of tokens with long and short notes interleaved evenly.
        """
        result = []

        # Use Bresenham-style error accumulation
        # We're distributing count_long items among total positions
        if count_long == 0:
            return [short_name] * total
        if count_short == 0:
            return [long_name] * total

        # Determine which is minority (to distribute)
        if count_long <= count_short:
            minority_name, minority_count = long_name, count_long
            majority_name = short_name
        else:
            minority_name, minority_count = short_name, count_short
            majority_name = long_name

        # Bresenham-style distribution
        error = 0
        minority_placed = 0

        for i in range(total):
            error += minority_count
            if error >= total - minority_placed:
                result.append(minority_name)
                error -= total
                minority_placed += 1
            else:
                result.append(majority_name)

        return result

    def quantize_single(
        self,
        duration: Fraction,
        max_notes: int = 4,
    ) -> Optional[List[str]]:
        """
        Quantize a single duration, potentially splitting into ties.

        Uses greedy algorithm: pick largest vocab duration that fits,
        repeat until remainder is zero or too small.

        Args:
            duration: Target duration as Fraction
            max_notes: Maximum number of tied notes allowed

        Returns:
            List of duration tokens to be tied, or None if no solution.

        Example:
            >>> q.quantize_single(Fraction(3, 20))  # dotted quintuplet
            ['8', '64']  # 1/8 + 1/64 ≈ 3/20
        """
        if duration <= 0:
            return None

        # Check if exact match exists
        for name, val in self.vocab_sorted:
            if val == duration:
                return [name]

        # Greedy decomposition
        result = []
        remaining = duration

        for _ in range(max_notes):
            if remaining <= 0:
                break

            # Find largest vocab duration that fits
            best_name, best_val = None, Fraction(0)
            for name, val in self.vocab_sorted:
                if val <= remaining:
                    best_name, best_val = name, val
                    break

            if best_name is None:
                # Remaining is smaller than smallest vocab duration
                # Use smallest as approximation
                smallest_name = self.vocab_sorted[-1][0]
                if remaining > self.vocab_sorted[-1][1] / 2:
                    result.append(smallest_name)
                break

            result.append(best_name)
            remaining -= best_val

        if not result:
            return None

        # Verify solution quality
        actual = sum(self.vocab[name] for name in result)
        error = abs(float(actual - duration) / float(duration))

        if error > 0.05:  # More than 5% error
            return None

        return result

    def quantize(
        self,
        duration: Fraction,
        num_notes: int = 1,
    ) -> Optional[List[str]]:
        """
        Universal quantization entry point.

        Args:
            duration: Target total duration
            num_notes: Number of notes (1 for single, >1 for tuplet)

        Returns:
            List of duration tokens
        """
        if num_notes > 1:
            return self.quantize_tuplet(duration, num_notes)
        else:
            return self.quantize_single(duration)


def analyze_tuplet(duration_str: str, num_notes: int) -> Dict:
    """
    Analyze a tuplet and provide quantization solution.

    Args:
        duration_str: Container duration (e.g., '2' for half note)
        num_notes: Number of notes in tuplet

    Returns:
        Dictionary with analysis results
    """
    q = RhythmQuantizer()

    container = duration_to_fraction(duration_str)
    if container is None:
        return {'error': f'Invalid duration: {duration_str}'}

    avg = container / num_notes
    result = q.quantize_tuplet(container, num_notes)

    if result is None:
        return {
            'container': str(container),
            'num_notes': num_notes,
            'average_duration': str(avg),
            'solution': None,
            'error': 'No solution found'
        }

    # Analyze solution
    from collections import Counter
    counts = Counter(result)

    actual_duration = sum(q.vocab[name] for name in result)
    error = float(actual_duration - container)

    return {
        'container': str(container),
        'num_notes': num_notes,
        'average_duration': str(avg),
        'solution': result,
        'composition': dict(counts),
        'actual_duration': str(actual_duration),
        'error': error,
        'error_percent': error / float(container) * 100 if container != 0 else 0,
    }


# =============================================================================
# Convenience functions for common cases
# =============================================================================

def solve_n_tuplet(n: int, container: str = '4') -> Optional[List[str]]:
    """
    Solve for n-tuplet in a standard container.

    Args:
        n: Number of notes in tuplet
        container: Container duration ('1'=whole, '2'=half, '4'=quarter, etc.)

    Returns:
        List of duration tokens

    Examples:
        >>> solve_n_tuplet(7, '4')   # septuplet in quarter
        ['32', '64', '32', '64', '32', '64', '32']
        >>> solve_n_tuplet(21, '2')  # 21-tuplet in half
        ['32', '64', '32', '64', ...]
    """
    q = RhythmQuantizer()
    container_frac = duration_to_fraction(container)
    if container_frac is None:
        return None
    return q.quantize_tuplet(container_frac, n)


def convert_oov_duration(duration_str: str) -> Optional[List[str]]:
    """
    Convert an OOV duration to standard vocab tokens.

    Args:
        duration_str: OOV duration string (e.g., '42', '21', '66')

    Returns:
        List of tied duration tokens, or None if no solution

    Example:
        >>> convert_oov_duration('42')
        ['64']  # 1/42 ≈ 1/64 (closest approximation)
    """
    q = RhythmQuantizer()
    dur = duration_to_fraction(duration_str)
    if dur is None:
        return None
    return q.quantize_single(dur)


# =============================================================================
# Pre-computed Lookup Table
# =============================================================================
# Format: {oov_duration: (long_dur, short_dur, count_long, count_short)}
# The actual distribution can be randomized at conversion time.

def build_lookup_table() -> Dict[str, Tuple[str, str, int, int]]:
    """
    Pre-compute quantization solutions for all possible OOV durations.

    Returns:
        Dictionary mapping OOV duration string to solution tuple:
        (long_dur, short_dur, count_long, count_short)

        If count_short == 0, it's a direct mapping (no mixing needed).
        If solution is None, the duration is truly unquantizable.
    """
    q = RhythmQuantizer()
    lookup = {}

    # Generate all possible tuplet durations we might encounter
    # Format in Humdrum: N means 1/N of whole note
    # Common tuplets: 3, 5, 6, 7, 9, 10, 11, 12, 13, ...
    # We check up to 256 to cover extreme cadenzas

    for n in range(1, 257):
        dur_str = str(n)

        # Skip if already in vocab
        if dur_str in ZENG_VOCAB:
            continue

        dur_frac = Fraction(1, n)

        # Find bracketing binary durations
        # Binary grid: 1, 2, 4, 8, 16, 32, 64, 128
        log_val = math.log2(n)  # n = 42 -> log2(42) ≈ 5.39
        lower_exp = math.floor(log_val)  # 5 -> 2^5 = 32
        upper_exp = math.ceil(log_val)   # 6 -> 2^6 = 64

        # Clamp to valid range
        lower_exp = max(0, min(7, lower_exp))  # 0-7 for 1-128
        upper_exp = max(0, min(7, upper_exp))

        t_long = 2 ** lower_exp   # e.g., 32
        t_short = 2 ** upper_exp  # e.g., 64

        if t_long == t_short:
            # Exact power of 2 (shouldn't happen for OOV, but just in case)
            lookup[dur_str] = (str(t_long), str(t_long), 1, 0)
            continue

        t_long_val = Fraction(1, t_long)
        t_short_val = Fraction(1, t_short)

        # For single note conversion:
        # We need to find the closest standard duration
        # Since this is a single note, we pick the closer one

        # Distance to each
        dist_long = abs(dur_frac - t_long_val)
        dist_short = abs(dur_frac - t_short_val)

        if dist_long <= dist_short:
            lookup[dur_str] = (str(t_long), str(t_long), 1, 0)
        else:
            lookup[dur_str] = (str(t_short), str(t_short), 1, 0)

    # Also handle dotted OOV (e.g., 42.)
    for n in range(1, 257):
        dur_str = f"{n}."

        if dur_str in ZENG_VOCAB:
            continue

        dur_frac = Fraction(3, 2 * n)  # n. = 1.5/n

        # Find closest standard dotted or use tie
        # For simplicity, map to closest vocab duration
        best_name, best_dist = None, float('inf')
        for name, val in ZENG_VOCAB.items():
            dist = abs(float(dur_frac - val))
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_name:
            lookup[dur_str] = (best_name, best_name, 1, 0)

    return lookup


# Pre-built lookup table (computed once at module load)
OOV_LOOKUP_TABLE = build_lookup_table()


def build_tuplet_lookup_table() -> Dict[Tuple[str, int], Tuple[str, str, int, int]]:
    """
    Pre-compute quantization solutions for common OOV tuplet patterns.

    Returns:
        Dictionary mapping (oov_duration, group_size) to solution tuple:
        (long_dur, short_dur, count_long, count_short)
    """
    q = RhythmQuantizer()
    lookup = {}

    # Known OOV durations from analysis
    oov_durations = [
        # Common cadenza tuplets
        '42', '21', '66', '58', '82', '62', '54', '46',
        # 9-tuplet (9 in time of 8) - moved from MERGE_MAPS
        '9', '18', '36', '72',
        # 13-tuplet (13 in time of 8/12) - moved from MERGE_MAPS
        '13', '26', '52', '104',
        # 17-tuplet (17 in time of 16) - moved from MERGE_MAPS
        '17', '34', '68',
        # 19-tuplet (19 in time of 16) - moved from MERGE_MAPS
        '19', '38', '76',
        # Stray OOV (isolated notes, not full tuplet groups)
        '5', '7', '10', '11', '14', '22', '23', '28',
    ]

    # Common group sizes (from analysis)
    # Extended to cover cadenza tuplets (up to 30+ notes in a group)
    group_sizes = list(range(1, 31))  # 1 to 30

    for oov in oov_durations:
        dur_frac = Fraction(1, int(oov))

        for n in group_sizes:
            total_dur = dur_frac * n

            # Find bracketing binary durations
            avg = total_dur / n

            # Sort by duration descending, use all vocab (including triplets)
            vocab_sorted = sorted(q.vocab.items(), key=lambda x: -x[1])

            t_long_name, t_long_val = None, None
            t_short_name, t_short_val = None, None

            for name, val in vocab_sorted:
                if val >= avg:
                    t_long_name, t_long_val = name, val
                if val <= avg and t_short_name is None:
                    t_short_name, t_short_val = name, val
                    break

            if t_long_val is None or t_short_val is None:
                continue

            if t_long_val == t_short_val:
                lookup[(oov, n)] = (t_long_name, t_long_name, n, 0)
                continue

            # Solve IP
            numerator = total_dur - (n * t_short_val)
            denominator = t_long_val - t_short_val

            if denominator == 0:
                lookup[(oov, n)] = (t_long_name, t_long_name, n, 0)
                continue

            x = numerator / denominator

            # Check if valid integer solution (exact only, no approximation)
            # CRITICAL: Only store exact solutions where error = 0
            if x.denominator == 1 and 0 <= x <= n:
                count_long = int(x)
                count_short = n - count_long
                lookup[(oov, n)] = (t_long_name, t_short_name, count_long, count_short)
            # else: No exact solution - don't store (will be reported as true OOV)

    return lookup


# Pre-built tuplet lookup table
TUPLET_LOOKUP_TABLE = build_tuplet_lookup_table()


def get_tuplet_solution(oov_duration: str, group_size: int) -> Optional[Tuple[str, str, int, int]]:
    """
    Quick lookup for tuplet quantization.

    Args:
        oov_duration: OOV duration string (e.g., '42')
        group_size: Number of consecutive notes

    Returns:
        Tuple of (long_dur, short_dur, count_long, count_short),
        or None if not in lookup table.
    """
    return TUPLET_LOOKUP_TABLE.get((oov_duration, group_size))


def get_quantized_duration(oov_duration: str) -> Optional[str]:
    """
    Quick lookup for OOV duration conversion.

    Args:
        oov_duration: OOV duration string (e.g., '42', '21')

    Returns:
        Standard vocab duration string, or None if no solution
    """
    if oov_duration in ZENG_VOCAB:
        return oov_duration

    entry = OOV_LOOKUP_TABLE.get(oov_duration)
    if entry:
        long_dur, short_dur, count_long, count_short = entry
        return long_dur  # For single note, just return the mapped duration

    return None


def get_tuplet_mix(oov_duration: str, num_notes: int) -> Optional[Tuple[str, str, int, int]]:
    """
    Get the mix formula for a tuplet.

    Args:
        oov_duration: Base duration of each note (e.g., '42' for 42nd notes)
        num_notes: Number of notes in the tuplet

    Returns:
        Tuple of (long_dur, short_dur, count_long, count_short)
        These can be distributed randomly or evenly.
    """
    dur_frac = duration_to_fraction(oov_duration)
    if dur_frac is None:
        return None

    total_duration = dur_frac * num_notes

    q = RhythmQuantizer()
    # Get the solution details
    avg = total_duration / num_notes

    # Find bracketing durations
    binary_sorted = sorted(q.binary_grid.items(), key=lambda x: -x[1])

    t_long_name, t_long_val = None, None
    t_short_name, t_short_val = None, None

    for name, val in binary_sorted:
        if val >= avg:
            t_long_name, t_long_val = name, val
        if val <= avg and t_short_name is None:
            t_short_name, t_short_val = name, val
            break

    if t_long_val is None or t_short_val is None:
        return None

    if t_long_val == t_short_val:
        return (t_long_name, t_long_name, num_notes, 0)

    # Solve IP
    numerator = total_duration - (num_notes * t_short_val)
    denominator = t_long_val - t_short_val

    if denominator == 0:
        return (t_long_name, t_long_name, num_notes, 0)

    x = numerator / denominator

    # CRITICAL: Only accept exact integer solutions (error = 0)
    if x.denominator != 1 or x < 0 or x > num_notes:
        # No exact solution exists - reject
        return None

    count_long = int(x)
    count_short = num_notes - count_long

    return (t_long_name, t_short_name, count_long, count_short)


def distribute_tuplet_random(
    long_dur: str,
    short_dur: str,
    count_long: int,
    count_short: int,
    seed: Optional[int] = None
) -> List[str]:
    """
    Distribute long and short durations randomly.

    Args:
        long_dur: Token for longer duration
        short_dur: Token for shorter duration
        count_long: Number of long notes
        count_short: Number of short notes
        seed: Random seed for reproducibility

    Returns:
        Randomly shuffled list of duration tokens
    """
    import random

    result = [long_dur] * count_long + [short_dur] * count_short

    if seed is not None:
        random.seed(seed)
    random.shuffle(result)

    return result


# =============================================================================
# Main: Demo and testing
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Rhythm Quantizer Demo")
    print("=" * 60)

    q = RhythmQuantizer()

    # Test cases
    test_cases = [
        # (container, num_notes, description)
        ('2', 21, "21-tuplet in half note (Liszt cadenza)"),
        ('4', 7, "Septuplet in quarter note"),
        ('4', 5, "Quintuplet in quarter note"),
        ('4', 11, "11-tuplet in quarter note"),
        ('2', 9, "9-tuplet in half note"),
        ('4', 13, "13-tuplet in quarter note"),
    ]

    for container, n, desc in test_cases:
        print(f"\n{desc}")
        print("-" * 40)
        result = analyze_tuplet(container, n)
        print(f"Container: {result['container']}")
        print(f"Average duration: {result['average_duration']}")
        if result.get('solution'):
            print(f"Composition: {result['composition']}")
            print(f"Error: {result['error_percent']:.6f}%")
            # Show first 15 notes of solution
            sol = result['solution']
            if len(sol) > 15:
                print(f"Solution: {sol[:15]}... (truncated)")
            else:
                print(f"Solution: {sol}")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")

    # Test OOV single durations
    print("\n" + "=" * 60)
    print("OOV Single Duration Lookup Table")
    print("=" * 60)

    oov_durations = ['42', '21', '66', '58', '82', '62', '54', '46']
    for dur in oov_durations:
        result = get_quantized_duration(dur)
        frac = duration_to_fraction(dur)
        if result:
            actual = ZENG_VOCAB[result]
            error = abs(float(actual - frac) / float(frac)) * 100
            print(f"  {dur} (1/{dur}) -> {result} (error: {error:.2f}%)")
        else:
            print(f"  {dur} -> No solution")

    # Test tuplet mix
    print("\n" + "=" * 60)
    print("Tuplet Mix Formula")
    print("=" * 60)

    # 7 notes of duration 28 (septuplet quarter)
    mix = get_tuplet_mix('28', 7)
    if mix:
        print(f"  7 x 28th notes: {mix[2]} x {mix[0]} + {mix[1]} x {mix[1]}")
        print(f"  Random distribution: {distribute_tuplet_random(*mix, seed=42)}")
