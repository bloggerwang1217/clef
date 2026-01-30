"""
Articulation rules: staccato/legato duration adjustments.

Staccato: shorten note duration
Legato: overlap notes slightly
"""

from typing import Dict, Any
from .base import Rule


class StaccatoRule(Rule):
    """Staccato: shorten note duration."""

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Shorten staccato notes."""
        if not self.enabled:
            return 1.0

        is_staccato = features.get('staccato', False)

        if is_staccato:
            # 50% shorter at k=1
            return 1.0 - self.k * 0.5

        return 1.0


class LegatoRule(Rule):
    """Legato: overlap notes slightly."""

    def __init__(self, config, overlap_ms: float = 30.0):
        """
        Initialize legato rule.

        Args:
            config: RuleConfig with k value
            overlap_ms: Overlap duration (milliseconds), default 30
        """
        super().__init__(config)
        self.overlap_ms = overlap_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """No velocity effect."""
        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Extend notes in legato passages."""
        if not self.enabled:
            return 1.0

        in_slur = features.get('in_slur', False)

        if in_slur:
            # Extend by overlap amount
            base_dur = features.get('duration', 0.5)
            if base_dur > 0:
                overlap_ratio = (self.k * self.overlap_ms / 1000) / base_dur
                return 1.0 + overlap_ratio

        return 1.0


class TenutoRule(Rule):
    """
    Tenuto: hold full value, slightly lengthen.

    Tenuto marking (-) indicates the note should be held for its full value,
    and often slightly lengthened beyond the written duration.
    """

    def __init__(self, config, extension_ratio: float = 0.05):
        """
        Initialize tenuto rule.

        Args:
            config: RuleConfig with k value
            extension_ratio: Duration extension ratio (default 5% = 0.05)
        """
        super().__init__(config)
        self.extension_ratio = extension_ratio

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Tenuto can have slight velocity boost (emphasis)."""
        if not self.enabled:
            return 0.0

        # Read from note.articulations (injected by engine)
        is_tenuto = features.get('tenuto', False)

        if is_tenuto:
            # Slight emphasis: +1dB
            return self.k * 1.0

        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Extend tenuto notes slightly."""
        if not self.enabled:
            return 1.0

        # Read from note.articulations (injected by engine)
        is_tenuto = features.get('tenuto', False)

        if is_tenuto:
            # Extend by 5% at k=1
            return 1.0 + self.k * self.extension_ratio

        return 1.0


class AccentRule(Rule):
    """
    Accent (>): increase velocity and optionally delay slightly.

    Accent marking indicates emphasis through both dynamics and timing.
    """

    def __init__(self, config, velocity_boost_dB: float = 3.0, delay_ms: float = 0.0):
        """
        Initialize accent rule.

        Args:
            config: RuleConfig with k value
            velocity_boost_dB: Velocity increase (dB), default 3.0
            delay_ms: Optional agogic delay (milliseconds), default 0 (no delay)
        """
        super().__init__(config)
        self.velocity_boost_dB = velocity_boost_dB
        self.delay_ms = delay_ms

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Increase velocity for accented notes."""
        if not self.enabled:
            return 0.0

        # Read from note.articulations (injected by engine)
        is_accent = features.get('accent', False)

        if is_accent:
            return self.k * self.velocity_boost_dB

        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """Optional agogic delay for accented notes."""
        if not self.enabled or self.delay_ms == 0.0:
            return 0.0

        # Read from note.articulations (injected by engine)
        is_accent = features.get('accent', False)

        if is_accent:
            return self.k * self.delay_ms / 1000

        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """No duration effect."""
        return 1.0


class MarcatoRule(Rule):
    """
    Marcato (^): strong accent, stronger than regular accent.

    Marcato marking indicates very strong emphasis, typically with:
    - Significant velocity increase
    - Possible slight shortening (detached character)
    """

    def __init__(self, config, velocity_boost_dB: float = 5.0, shortening_ratio: float = 0.05):
        """
        Initialize marcato rule.

        Args:
            config: RuleConfig with k value
            velocity_boost_dB: Velocity increase (dB), default 5.0 (stronger than accent)
            shortening_ratio: Duration shortening ratio (default 5% = 0.05)
        """
        super().__init__(config)
        self.velocity_boost_dB = velocity_boost_dB
        self.shortening_ratio = shortening_ratio

    def apply_velocity(self, note: Any, features: Dict[str, Any]) -> float:
        """Strong velocity boost for marcato notes."""
        if not self.enabled:
            return 0.0

        # Read from note.articulations (injected by engine)
        is_marcato = features.get('marcato', False)

        if is_marcato:
            return self.k * self.velocity_boost_dB

        return 0.0

    def apply_timing(self, note: Any, features: Dict[str, Any]) -> float:
        """No timing effect."""
        return 0.0

    def apply_duration(self, note: Any, features: Dict[str, Any]) -> float:
        """Slightly shorten marcato notes (detached character)."""
        if not self.enabled:
            return 1.0

        # Read from note.articulations (injected by engine)
        is_marcato = features.get('marcato', False)

        if is_marcato:
            # Shorten by 5% at k=1
            return 1.0 - self.k * self.shortening_ratio

        return 1.0
