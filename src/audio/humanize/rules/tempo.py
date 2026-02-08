"""
Tempo marking interpretation.

Converts tempo markings (Allegro, Andante, etc.) to BPM values.
NOT a Rule - this sets the BASE tempo before rules are applied.
"""

from typing import Optional, Tuple, Dict
import numpy as np


# Standard tempo ranges (BPM) from classical music tradition
TEMPO_MARKINGS: Dict[str, Tuple[float, float]] = {
    # Very slow
    'grave': (20, 40),
    'largo': (40, 60),
    'lento': (45, 60),
    'larghetto': (60, 66),
    'adagio': (66, 76),

    # Slow
    'andante': (76, 108),
    'andantino': (80, 108),

    # Moderate
    'moderato': (108, 120),
    'allegretto': (112, 120),

    # Fast
    'allegro': (120, 168),
    'vivace': (168, 176),
    'presto': (168, 200),
    'prestissimo': (200, 240),
}


class TempoInterpreter:
    """
    Interpret tempo markings from score.

    NOT a Rule (no k value) - this sets the BASE tempo.
    """

    def __init__(self, default_bpm: float = 108.0):
        """
        Initialize interpreter.

        Args:
            default_bpm: Default tempo if no marking found (BasisMixer default)
        """
        self.default_bpm = default_bpm

    def get_base_tempo(
        self,
        marking: Optional[str],
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Get base BPM from tempo marking.

        Args:
            marking: Tempo marking string (e.g., "Allegro", "♩= 120")
            rng: Random generator for sampling within range (optional)

        Returns:
            BPM value
        """
        if marking is None:
            return self.default_bpm

        marking_lower = marking.lower().strip()

        # Check for explicit BPM (e.g., "♩= 120" or "120")
        if '=' in marking_lower:
            try:
                bpm = float(marking_lower.split('=')[1].strip())
                return bpm
            except (ValueError, IndexError):
                pass

        # Try to parse as number
        try:
            bpm = float(marking_lower)
            if 20 <= bpm <= 300:  # Reasonable range
                return bpm
        except ValueError:
            pass

        # Check known markings
        for name, (low, high) in TEMPO_MARKINGS.items():
            if name in marking_lower:
                if rng is not None:
                    return rng.uniform(low, high)
                return (low + high) / 2

        return self.default_bpm

    def get_tempo_from_score(self, part) -> float:
        """
        Extract tempo marking from partitura Part.

        Args:
            part: partitura Part object

        Returns:
            BPM value
        """
        try:
            import partitura

            # Look for explicit tempo markings
            for tempo in part.iter_all(partitura.score.Tempo):
                if hasattr(tempo, 'bpm') and tempo.bpm is not None:
                    return float(tempo.bpm)

            # Look for tempo directions (text)
            for direction in part.iter_all(partitura.score.Direction):
                if hasattr(direction, 'text') and direction.text:
                    bpm = self.get_base_tempo(direction.text)
                    if bpm != self.default_bpm:
                        return bpm

        except Exception:
            pass

        return self.default_bpm

    def get_tempo_from_musicxml(self, musicxml_path: str) -> float:
        """
        Extract tempo from MusicXML file by direct XML parsing.

        Much faster than music21 - only parses metronome markings.

        Args:
            musicxml_path: Path to .xml or .mxl file

        Returns:
            Tempo in BPM, or default_bpm if not found
        """
        import xml.etree.ElementTree as ET
        from zipfile import ZipFile
        from pathlib import Path

        path = Path(musicxml_path)

        try:
            # Read XML
            if path.suffix == '.mxl':
                # Compressed MusicXML
                with ZipFile(path) as z:
                    # Find the main XML file (not META-INF)
                    for name in z.namelist():
                        if name.endswith('.xml') and not name.startswith('META-INF'):
                            root = ET.fromstring(z.read(name))
                            break
                    else:
                        return self.default_bpm
            else:
                # Uncompressed XML
                root = ET.parse(path).getroot()

            # Search for metronome marking
            # Structure: <direction> → <direction-type> → <metronome> → <per-minute>
            for direction in root.iter('direction'):
                for dt in direction.findall('direction-type'):
                    metronome = dt.find('metronome')
                    if metronome is not None:
                        per_minute = metronome.find('per-minute')
                        if per_minute is not None and per_minute.text:
                            try:
                                bpm = float(per_minute.text)

                                # Convert to quarter-note BPM based on beat-unit
                                # MusicXML per-minute is in the beat-unit's BPM,
                                # but our engine uses quarter-note BPM throughout
                                beat_unit = metronome.find('beat-unit')
                                unit_to_quarters = {
                                    'whole': 4.0,
                                    'half': 2.0,
                                    'quarter': 1.0,
                                    'eighth': 0.5,
                                    '16th': 0.25,
                                    '32nd': 0.125,
                                }
                                if beat_unit is not None and beat_unit.text:
                                    quarter_mult = unit_to_quarters.get(
                                        beat_unit.text, 1.0
                                    )
                                else:
                                    quarter_mult = 1.0  # Assume quarter

                                # Check for dotted note (e.g., dotted quarter = 1.5x)
                                beat_unit_dot = metronome.find('beat-unit-dot')
                                if beat_unit_dot is not None:
                                    quarter_mult *= 1.5

                                bpm *= quarter_mult
                                return bpm
                            except ValueError:
                                continue

            # Also check <sound tempo="120"/> attribute
            for sound in root.iter('sound'):
                tempo_attr = sound.get('tempo')
                if tempo_attr:
                    try:
                        return float(tempo_attr)
                    except ValueError:
                        continue

            return self.default_bpm

        except Exception as e:
            # Don't fail loudly - just return default
            return self.default_bpm
