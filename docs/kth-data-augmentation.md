# clef-piano-full MIDI Humanization 實作計畫

## 背景

clef 是一個 audio-to-score (A2S) 研究專案，目標是 ISMIR 2026 / ICLR 2027。

**核心哲學**：「**Enlightenment**」而非「Noise」— 這是在加入音樂知識，不是加入隨機噪音。把這件事當作「**混音**」來看待，把 DAW 搬到 Python 上。

**專案現況**：
- **clef-piano-base**（已完成）：Zeng baseline 比較用，移除 `**dynam` spine，uniform velocity (90)
- **clef-piano-full**（待實作）：保留 `**dynam` spine (--preset clef-piano-full parameter in humsyn_preprocessor.py)，需要 rule-based humanization

**重要原則1**：兩個 pipeline 完全分離，不會互相影響。clef-piano-base 的設定不會被修改。
**重要原則2**：實作時請確保按照 KTH rule book 的預設數值，請確認實作實作數值與 rule book 一致
---

## 三階段架構

### Stage 1: Score → MIDI (音樂性編碼) — 本次重點

用 **Partitura** 解析樂譜，實作 **KTH 規則系統**：
- Dynamics 標記 → Velocity 映射
- HIGH LOUD: pitch 越高 velocity 越強
- PHRASE ARCH: 樂句中間強，兩端弱
- METRICAL ACCENT: 下拍重音
- Micro-timing jitter (±10-15ms)
- Chord asynchrony (melody lead 20-50ms)
- Final ritardando (平方根函數)
- Auto pedal (syncopated pedaling)
- **所有參數可 randomize** 產生多樣性

### Stage 2: MIDI → Audio (音色渲染) — 精簡

- 沿用現有 FluidSynth pipeline
- 4 個 SoundFont (TimGM6mb, FluidR3_GM, UprightPianoKW, SalamanderGrandPiano)
- 保持訊號乾淨

### Stage 3: Audio → Audio (真實世界模擬) — 暫緩

- Piano solo 可以少做（ASAP 夠乾淨）
- 留給 clef-tutti 認真做

---

## 檔案結構與職責

```
src/audio/humanize/                # 新增目錄
├── __init__.py                    # 公開 API exports
├── config.py                      # RuleConfig + HumanizationConfig
├── metadata.py                    # HumanizationMetadata for reproducibility
├── analysis/                      # 分析工具
│   ├── __init__.py
│   └── non_chord_tone.py          # ⭐ NCT heuristic detection
├── rules/                         # KTH 規則實作
│   ├── __init__.py
│   ├── base.py                    # Rule 抽象基底類別
│   │
│   │  # === Velocity 規則 ===
│   ├── high_loud.py               # Pitch → velocity
│   ├── phrase_arch.py             # Phrase position → velocity + tempo
│   ├── duration_contrast.py       # Duration → velocity + duration
│   ├── melodic_charge.py          # ⭐ 非和弦音/導音 → velocity (需和聲分析)
│   │
│   │  # === Timing 規則 ===
│   ├── rubato.py                  # ⭐ Phrase/beat level tempo variation
│   ├── final_ritard.py            # End section slowdown
│   ├── timing.py                  # Micro-timing jitter + chord async
│   ├── fermata.py                 # ⭐ Fermata duration + pause
│   ├── dynamics_tempo.py          # ⭐ Cresc→accel, Agogic accent
│   ├── articulation_tempo.py      # ⭐ Tenuto/legato → timing
│   ├── punctuation.py             # ⭐ 樂句間 micropause (氣口)
│   ├── leap.py                    # ⭐ 大跳 timing/duration 調整
│   ├── repetition.py              # ⭐ 重複音 micropause
│   │
│   │  # === Articulation 規則 ===
│   ├── articulation.py            # Staccato/legato duration
│   ├── ornaments.py               # ⭐ Grace notes, trills, mordents
│   │
│   │  # === Special ===
│   ├── pedal.py                   # Auto pedaling (CC64)
│   ├── tempo.py                   # ⭐ Tempo marking → BPM
│   └── safety.py                  # ⭐ Social-duration-care, Global normalization
│
├── engine.py                      # HumanizationEngine 主類別
├── convert.py                     # dB ↔ velocity 轉換工具
└── presets.py                     # 風格預設 (romantic, classical)
```

**共 26 個規則/元件**：
| 類別 | 規則數 | 規則名稱 |
|------|--------|----------|
| Velocity | 4 | HighLoud, PhraseArch(vel), DurationContrast(vel), **MelodicCharge** |
| Timing | 12 | PhraseRubato, BeatRubato, FinalRitard, MicroTiming, ChordAsync, Fermata, CrescendoTempo, AgogicAccent, ArticulationTempo, **Punctuation**, **Leap**, **Repetition** |
| Articulation | 5 | Staccato, Legato, GraceNote, Trill, Mordent |
| Safety | 2 | **SocialDurationCare**, **GlobalNormalizer** |
| Special | 3 | AutoPedal, TempoInterpreter, **CLI** |

### Gemini / Claude 回饋確認

| 建議 | 狀態 | 對應規則 |
|------|------|----------|
| Melodic Charge (旋律張力) | ✅ 已加入 | `melodic_charge.py` |
| Punctuation (氣口) | ✅ 已加入 | `punctuation.py` |
| Sound Level Envelope (防爆) | ✅ 已加入 | `safety.py` (GlobalNormalizer) |
| Social-duration-care | ✅ 已加入 | `safety.py` |
| Leap handling | ✅ 已加入 | `leap.py` |
| Repetition handling | ✅ 已加入 | `repetition.py` |
| Inégales | ⚪ 延後 | 古典鋼琴不需要 |
| Intonation rules | ⚪ 延後 | 留給 clef-solo/tutti |

### Velocity-Tempo-Duration 耦合關係圖

```
Score Feature          Velocity Effect       Tempo Effect         Duration Effect
────────────────────────────────────────────────────────────────────────────────
crescendo        →     ↑ velocity       +    ↑ tempo (accel)      —
diminuendo       →     ↓ velocity       +    ↓ tempo (rit)        —
sf / accent      →     ↑↑ velocity      +    agogic delay         —
phrase peak      →     ↑ velocity       +    ↑ tempo              —
phrase end       →     ↓ velocity       +    ↓ tempo         +    ↓ duration (氣口)
non-chord tone   →     ↑ velocity       +    —                +    ↑ duration
leading tone     →     ↑ velocity       +    —                    —
tenuto           →     —                +    slight delay    +    ↑ duration
staccato         →     —                +    —               +    ↓↓ duration
large leap up    →     —                +    micropause      +    ↓ duration (首音)
large leap down  →     —                +    micropause      +    ↑ duration (首音)
repeated note    →     —                +    micropause      +    ↓ duration
```

---

## 各檔案詳細規格

### `src/audio/humanize/__init__.py`

```python
"""
KTH-style MIDI humanization with k-value system.

Usage:
    from src.audio.humanize import HumanizationEngine, HumanizationConfig

    config = HumanizationConfig().randomize(seed=42)
    engine = HumanizationEngine(config)
    engine.humanize('score.krn', 'output.mid')
"""
from .config import RuleConfig, HumanizationConfig
from .engine import HumanizationEngine
from .presets import ROMANTIC, CLASSICAL, BALANCED

__all__ = [
    'RuleConfig',
    'HumanizationConfig',
    'HumanizationEngine',
    'ROMANTIC', 'CLASSICAL', 'BALANCED',
]
```

---

### `src/audio/humanize/config.py`

**職責**：定義 k 值系統的設定結構

**包含**：
- `RuleConfig` dataclass — 單一規則的 k 值與範圍
- `HumanizationConfig` dataclass — 完整設定，包含所有規則
- `randomize()` 方法 — 產生隨機化設定副本
- `to_dict()` 方法 — 匯出設定供 logging

**依賴**：`numpy`（RNG）

---

### `src/audio/humanize/convert.py`

**職責**：dB ↔ MIDI velocity 轉換

**包含**：
```python
def velocity_to_dB(velocity: int, reference: int = 64) -> float:
    """Convert MIDI velocity to dB (0 dB = reference velocity)."""

def dB_to_velocity(dB: float, reference: int = 64) -> int:
    """Convert dB back to MIDI velocity, clamped to 1-127."""

def dynamics_to_velocity(marking: str, dynamics_map: dict) -> int:
    """Convert dynamic marking (p, f, etc.) to velocity."""
```

**依賴**：`numpy`

**參考**：KTH PDF 的 dB-velocity 曲線

---

### `src/audio/humanize/rules/base.py`

**職責**：規則抽象基底類別

**包含**：
```python
from abc import ABC, abstractmethod

class Rule(ABC):
    """Base class for KTH-style humanization rules."""

    def __init__(self, config: RuleConfig):
        self.config = config

    @property
    def k(self) -> float:
        return self.config.k

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @abstractmethod
    def apply_velocity(self, note, features: dict) -> float:
        """Return velocity delta in dB."""
        pass

    @abstractmethod
    def apply_timing(self, note, features: dict) -> float:
        """Return timing delta in seconds."""
        pass

    @abstractmethod
    def apply_duration(self, note, features: dict) -> float:
        """Return duration multiplier."""
        pass
```

---

### `src/audio/humanize/rules/high_loud.py`

**職責**：實作 High-loud 規則（pitch 越高越大聲）

**公式**：`dB_delta = k × 0.5 × (pitch - 60)`

**包含**：
```python
class HighLoudRule(Rule):
    """High-loud: higher pitches are played louder."""

    SEMITONE_COEFFICIENT = 0.5  # dB per semitone

    def apply_velocity(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0
        semitones_above_c4 = note.pitch - 60
        return self.k * self.SEMITONE_COEFFICIENT * semitones_above_c4

    def apply_timing(self, note, features: dict) -> float:
        return 0.0  # No timing effect

    def apply_duration(self, note, features: dict) -> float:
        return 1.0  # No duration effect
```

---

### `src/audio/humanize/rules/phrase_arch.py`

**職責**：實作 Phrase-arch 規則（樂句弧線）

**公式**：`dB_delta = k × 6 × arch_function(position)`

**包含**：
```python
class PhraseArchRule(Rule):
    """Phrase-arch: louder in middle of phrase, softer at boundaries."""

    MAX_EFFECT_DB = 6.0

    def __init__(self, config: RuleConfig, peak_position: float = 0.6):
        super().__init__(config)
        self.peak_position = peak_position

    def apply_velocity(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0
        phrase_pos = features.get('phrase_position')  # 0-1
        if phrase_pos is None:
            return 0.0
        # Parabolic arch centered at peak_position
        arch = 1 - ((phrase_pos - self.peak_position) / self.peak_position) ** 2
        return self.k * self.MAX_EFFECT_DB * arch
```

**依賴**：需要 partitura 的 `slur_basis` 來偵測樂句邊界

---

### `src/audio/humanize/rules/duration_contrast.py`

**職責**：實作 Duration-contrast 規則（長音更長更大聲）

**包含**：
```python
class DurationContrastRule(Rule):
    """Duration-contrast: longer notes louder and stretched."""

    def apply_velocity(self, note, features: dict) -> float:
        # Relative duration vs local average
        rel_dur = features.get('relative_duration', 1.0)
        return self.k * 3.0 * np.log2(rel_dur)

    def apply_duration(self, note, features: dict) -> float:
        rel_dur = features.get('relative_duration', 1.0)
        return 1.0 + self.k * 0.1 * (rel_dur - 1.0)
```

---

### `src/audio/humanize/rules/melodic_charge.py` ⭐ 新增 (Gemini 建議)

**職責**：實作 Melodic-charge 規則（非和弦音強調）

**KTH 定義**：*Emphasis on notes remote from current chord/key.*

**實作方式**：使用 **Heuristic-based NCT detection**（不需要完整和聲分析）

**為什麼重要**：
- 不只是「高音大聲」(High-loud)
- 而是「**不和諧**或**具導向性**」的音大聲
- 這是讓演奏有「音樂性」而非「機械性」的關鍵

**包含**：
```python
class MelodicChargeRule(Rule):
    """
    Melodic-charge: emphasis on non-chord tones.

    Uses heuristic detection based on:
    1. Metric position (weak beat = likely NCT)
    2. Duration (short = likely NCT)
    3. Melodic motion (step-step patterns)
    4. Dissonance with concurrent notes
    """

    def __init__(self, config: RuleConfig, nct_boost_dB: float = 2.0):
        super().__init__(config)
        self.nct_boost_dB = nct_boost_dB
        self.detector = NonChordToneDetector()

    def apply_velocity(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0

        analysis = self.detector.analyze_note(
            note_idx=features['note_idx'],
            note_array=features['note_array'],
            features=features
        )
        # NCT confidence (0-1) → dB boost
        return self.k * analysis.melodic_charge

    def apply_timing(self, note, features: dict) -> float:
        """Appoggiaturas get agogic accent (slight delay)."""
        if not self.enabled:
            return 0.0

        analysis = self.detector.analyze_note(...)
        if analysis.nct_type == NCTType.APPOGGIATURA:
            return self.k * 0.02  # 20ms delay
        return 0.0
```

**HumanizationConfig 設定**：
```python
melodic_charge: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=0.8, k_range=(0.3, 1.2)  # Conservative: heuristic-based
))
nct_boost_dB: float = 2.0
```

---

### `src/audio/humanize/analysis/non_chord_tone.py` ⭐ 新增

**職責**：Heuristic-based 非和弦音偵測（不需要和聲分析）

**核心原理**：非和弦音分類基於**接近**和**離開**方式：

| 類型 | 接近 | 離開 | 位置 |
|------|------|------|------|
| Passing Tone | Step | Step (同向) | 弱拍 |
| Neighbor Tone | Step | Step (反向回原) | 弱拍 |
| Appoggiatura | Leap | Step (反向) | **強拍** |
| Escape Tone | Step | Leap (反向) | 弱拍 |

**包含**：
```python
from dataclasses import dataclass
from enum import Enum

class NCTType(Enum):
    CHORD_TONE = "chord_tone"
    PASSING_TONE = "passing_tone"
    NEIGHBOR_TONE = "neighbor_tone"
    APPOGGIATURA = "appoggiatura"
    ESCAPE_TONE = "escape_tone"
    UNKNOWN = "unknown"

@dataclass
class NCTAnalysis:
    nct_type: NCTType
    confidence: float  # 0.0 - 1.0
    melodic_charge: float  # dB boost


class NonChordToneDetector:
    """
    Heuristic-based NCT detection.
    ~75% accuracy, fast for batch processing.
    """

    def analyze_note(self, note_idx, note_array, features) -> NCTAnalysis:
        score = 0.0
        detected_type = NCTType.UNKNOWN

        note = note_array[note_idx]
        prev = note_array[note_idx - 1] if note_idx > 0 else None
        next = note_array[note_idx + 1] if note_idx < len(note_array) - 1 else None

        # Heuristic 1: 節拍位置
        if features.get('beat_strength', 0.5) < 0.3:
            score += 0.2

        # Heuristic 2: 時值
        if features.get('duration_ratio', 1.0) < 0.5:
            score += 0.15

        # Heuristic 3: 旋律運動
        if prev is not None and next is not None:
            motion = self._analyze_motion(prev, note, next)
            detected_type = motion['type']
            score += motion['score']

        # Heuristic 4: 不協和音程
        concurrent = self._get_concurrent_pitches(note, note_array)
        score += self._compute_dissonance(note, concurrent)

        confidence = min(score, 1.0)
        melodic_charge = confidence * 2.0  # 0-2 dB

        if detected_type == NCTType.UNKNOWN:
            detected_type = NCTType.PASSING_TONE if score > 0.5 else NCTType.CHORD_TONE

        return NCTAnalysis(detected_type, confidence, melodic_charge)

    def _analyze_motion(self, prev, curr, next) -> dict:
        interval_in = curr['pitch'] - prev['pitch']
        interval_out = next['pitch'] - curr['pitch']
        is_step_in, is_step_out = abs(interval_in) <= 2, abs(interval_out) <= 2
        same_dir = interval_in * interval_out > 0
        opp_dir = interval_in * interval_out < 0

        if is_step_in and is_step_out and same_dir:
            return {'type': NCTType.PASSING_TONE, 'score': 0.4}
        if is_step_in and is_step_out and opp_dir:
            return {'type': NCTType.NEIGHBOR_TONE, 'score': 0.4}
        if not is_step_in and is_step_out and opp_dir:
            return {'type': NCTType.APPOGGIATURA, 'score': 0.35}
        if is_step_in and not is_step_out and opp_dir:
            return {'type': NCTType.ESCAPE_TONE, 'score': 0.3}
        return {'type': NCTType.UNKNOWN, 'score': 0.0}

    def _compute_dissonance(self, note, concurrent_pitches):
        note_pc = note['pitch'] % 12
        dissonant = {1, 2, 6, 10, 11}  # m2, M2, tritone, m7, M7
        score = sum(0.1 for pc in concurrent_pitches
                    if pc != note_pc and min(abs(note_pc - pc), 12 - abs(note_pc - pc)) in dissonant)
        return min(score, 0.25)
```

**準確度比較**：

| 方法 | 準確度 | 速度 | 推薦 |
|------|--------|------|------|
| 純節拍+時值 | ~60% | ⚡ | Baseline |
| **旋律運動分析** | **~75%** | ⚡ | **推薦** |
| + Vertical slice | ~80% | 中 | 最佳 |
| music21 chordify | ~85% | 🐌 | 備選 |

---

### `src/audio/humanize/rules/punctuation.py` ⭐ 新增 (Gemini 建議)

**職責**：實作 Punctuation 規則（樂句間的「氣口」）

**KTH 定義**：*Automatically locates small tone groups and marks them with lengthening of last note and a following micropause.*

**與 PhraseRubato 的差異**：
- PhraseRubato = tempo 變慢但**連續**
- Punctuation = 真正的**斷開 (Silence)**

**為什麼重要**：對 A2S 模型來說，**Silence 是判斷樂句邊界最強的 Feature**

**包含**：
```python
class PunctuationRule(Rule):
    """
    Punctuation: create silence (breathing) between phrases.

    Creates actual gaps, not just tempo changes.
    Critical for A2S models to learn phrase boundaries.
    """

    def __init__(self, config: RuleConfig,
                 micropause_ms: float = 30.0,
                 last_note_shorten_ratio: float = 0.15):
        super().__init__(config)
        self.micropause_ms = micropause_ms
        self.last_note_shorten_ratio = last_note_shorten_ratio

    def apply_duration(self, note, features: dict) -> float:
        """Shorten the last note of a phrase to create gap."""
        if not self.enabled:
            return 1.0

        if features.get('is_phrase_end', False):
            # Shorten note to create micropause
            return 1.0 - self.k * self.last_note_shorten_ratio

        return 1.0

    def apply_timing(self, note, features: dict) -> float:
        """Delay the first note of a new phrase."""
        if not self.enabled:
            return 0.0

        if features.get('is_phrase_start', False) and features.get('phrase_number', 0) > 0:
            # Add micropause before new phrase
            return self.k * self.micropause_ms / 1000

        return 0.0
```

**HumanizationConfig 設定**：
```python
punctuation: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.5, 1.5)
))
micropause_ms: float = 30.0
phrase_end_shorten_ratio: float = 0.15
```

---

### `src/audio/humanize/rules/leap.py` ⭐ 新增 (KTH)

**職責**：實作大跳相關規則

**KTH 規則**：
- **Leap-tone-duration**: 上跳縮短首音，下跳延長首音
- **Leap-articulation-dro**: 大跳後加 micropause

**包含**：
```python
class LeapRule(Rule):
    """
    Leap handling: adjust timing/duration around large intervals.

    - Upward leap: shorten first note (lighter)
    - Downward leap: lengthen first note (weightier)
    - After large leap: small micropause
    """

    def __init__(self, config: RuleConfig,
                 leap_threshold: int = 7,  # semitones (perfect 5th)
                 duration_effect: float = 0.1,
                 micropause_ms: float = 15.0):
        super().__init__(config)
        self.leap_threshold = leap_threshold
        self.duration_effect = duration_effect
        self.micropause_ms = micropause_ms

    def apply_duration(self, note, features: dict) -> float:
        if not self.enabled:
            return 1.0

        interval = features.get('interval_to_next', 0)
        if abs(interval) >= self.leap_threshold:
            if interval > 0:  # Upward leap
                return 1.0 - self.k * self.duration_effect  # Shorten
            else:  # Downward leap
                return 1.0 + self.k * self.duration_effect  # Lengthen

        return 1.0

    def apply_timing(self, note, features: dict) -> float:
        """Add micropause after landing from a large leap."""
        if not self.enabled:
            return 0.0

        interval_from_prev = features.get('interval_from_prev', 0)
        if abs(interval_from_prev) >= self.leap_threshold:
            return self.k * self.micropause_ms / 1000

        return 0.0
```

**HumanizationConfig 設定**：
```python
leap: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.5, 1.5)
))
leap_threshold_semitones: int = 7
leap_duration_effect: float = 0.1
leap_micropause_ms: float = 15.0
```

---

### `src/audio/humanize/rules/repetition.py` ⭐ 新增 (KTH)

**職責**：實作重複音規則

**KTH 定義**：*Repetition-articulation-dro: Micropause for repeated notes.*

**包含**：
```python
class RepetitionRule(Rule):
    """
    Repetition handling: add micropause between repeated notes.

    Prevents "machine gun" effect on repeated notes.
    """

    def __init__(self, config: RuleConfig,
                 micropause_ms: float = 20.0):
        super().__init__(config)
        self.micropause_ms = micropause_ms

    def apply_duration(self, note, features: dict) -> float:
        """Shorten repeated notes slightly."""
        if not self.enabled:
            return 1.0

        if features.get('is_repeated_note', False):
            return 1.0 - self.k * 0.1  # -10% duration

        return 1.0

    def apply_timing(self, note, features: dict) -> float:
        """Slight delay on repeated notes."""
        if not self.enabled:
            return 0.0

        if features.get('is_repeated_note', False):
            # Small random variation to avoid mechanical feel
            return self.k * self.micropause_ms / 1000

        return 0.0
```

---

### `src/audio/humanize/rules/safety.py` ⭐ 新增 (Gemini 建議)

**職責**：安全規則 + 全域正規化

**包含兩個元件**：

```python
class SocialDurationCareRule(Rule):
    """
    Social-duration-care: auto-lengthen very short notes.

    Prevents notes from being too short to hear.
    KTH principle: "care for the listener"
    """

    def __init__(self, config: RuleConfig,
                 min_duration_ms: float = 50.0):
        super().__init__(config)
        self.min_duration_ms = min_duration_ms

    def apply_duration(self, note, features: dict) -> float:
        if not self.enabled:
            return 1.0

        note_duration_ms = note.duration * 1000
        if note_duration_ms < self.min_duration_ms:
            # Extend to minimum audible duration
            target_ratio = self.min_duration_ms / note_duration_ms
            return 1.0 + self.k * (target_ratio - 1.0)

        return 1.0


class GlobalNormalizer:
    """
    Global velocity normalization / soft limiting.

    Prevents "smashing piano" when multiple rules stack up.
    Applied as post-processing after all rules.
    """

    def __init__(self,
                 target_rms_velocity: int = 70,
                 max_velocity: int = 115,
                 soft_clip_threshold: int = 100):
        self.target_rms_velocity = target_rms_velocity
        self.max_velocity = max_velocity
        self.soft_clip_threshold = soft_clip_threshold

    def normalize(self, velocities: np.ndarray) -> np.ndarray:
        """Apply global normalization and soft clipping."""
        # 1. RMS normalization (optional)
        # current_rms = np.sqrt(np.mean(velocities ** 2))
        # scale = self.target_rms_velocity / current_rms
        # velocities = velocities * scale

        # 2. Soft clipping for peaks
        # Use tanh-style soft clip above threshold
        above_threshold = velocities > self.soft_clip_threshold
        if np.any(above_threshold):
            excess = velocities[above_threshold] - self.soft_clip_threshold
            max_excess = self.max_velocity - self.soft_clip_threshold
            # Soft clip: compress excess into remaining headroom
            compressed = max_excess * np.tanh(excess / max_excess)
            velocities[above_threshold] = self.soft_clip_threshold + compressed

        # 3. Hard clip as safety
        velocities = np.clip(velocities, 1, 127)

        return velocities.astype(int)
```

**HumanizationConfig 設定**：
```python
social_duration_care: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.8, 1.2)
))
min_audible_duration_ms: float = 50.0

# Global normalizer settings
normalize_velocity: bool = True
target_rms_velocity: int = 70
max_velocity: int = 115
soft_clip_threshold: int = 100
```

---

### `src/audio/humanize/rules/final_ritard.py`

**職責**：實作 Final-ritard 規則（結尾漸慢）

**公式**：`tempo_ratio = 1 - k × sqrt(position_in_final_section)`

**包含**：
```python
class FinalRitardRule(Rule):
    """Final-ritard: gradual slowdown at the end (runner stopping model)."""

    def __init__(self, config: RuleConfig, start_position: float = 0.9):
        super().__init__(config)
        self.start_position = start_position

    def apply_timing(self, note, features: dict) -> float:
        piece_pos = features.get('piece_position', 0.0)  # 0-1
        if piece_pos < self.start_position:
            return 0.0
        # Position within final section (0 to 1)
        final_pos = (piece_pos - self.start_position) / (1 - self.start_position)
        # Cumulative delay based on sqrt model
        return self.k * 0.5 * np.sqrt(final_pos)  # Max 0.5s delay at end
```

---

### `src/audio/humanize/rules/rubato.py` ⭐ 新增

**職責**：實作樂句級 Rubato（速度變化）

**KTH Phrase-arch tempo 規則**：樂句開始慢 → 中間快 → 結尾漸慢

**公式**：
```
tempo_ratio = 1 + k × rubato_curve(phrase_position)
rubato_curve: 開始 -0.1, 中間 +0.1, 結尾 -0.15
```

**包含**：
```python
class PhraseRubatoRule(Rule):
    """
    Phrase-level rubato: tempo varies within phrases.

    Based on KTH Phrase-arch rule (tempo component):
    - Slower at phrase start (settling in)
    - Faster in middle (forward momentum)
    - Slower at phrase end (breathing, punctuation)
    """

    def __init__(self, config: RuleConfig, peak_position: float = 0.6):
        super().__init__(config)
        self.peak_position = peak_position

    def apply_timing(self, note, features: dict) -> float:
        """Return cumulative timing offset based on phrase position."""
        if not self.enabled:
            return 0.0

        phrase_pos = features.get('phrase_position')  # 0-1 within phrase
        if phrase_pos is None:
            return 0.0

        # Compute local tempo ratio
        # Asymmetric: slower start, faster middle, slowest end
        if phrase_pos < self.peak_position:
            # Accelerating phase: -0.1 → +0.1
            ratio = -0.1 + 0.2 * (phrase_pos / self.peak_position)
        else:
            # Decelerating phase: +0.1 → -0.15
            decel_pos = (phrase_pos - self.peak_position) / (1 - self.peak_position)
            ratio = 0.1 - 0.25 * decel_pos

        # Convert tempo ratio to timing offset
        # Negative ratio (slower) = positive timing offset (later)
        beat_duration = features.get('beat_duration', 0.5)  # seconds
        return -self.k * ratio * beat_duration * 0.5  # Scale down effect


class BeatRubatoRule(Rule):
    """
    Beat-level micro rubato: subtle push/pull on beat boundaries.

    Creates "breathing" quality by slightly delaying or rushing beats.
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)
        self.rng = None  # Set by engine

    def apply_timing(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0

        # Stronger effect on downbeats
        beat_strength = features.get('beat_strength', 0.5)  # 0-1

        # Random but correlated across nearby notes
        base_rubato = self.rng.normal(0, 0.02)  # ±20ms base

        # Downbeats tend to be slightly late (weight)
        if features.get('is_downbeat', False):
            base_rubato += 0.01  # +10ms tendency

        return self.k * base_rubato * beat_strength
```

**與 dynamics 的耦合**（KTH 原則）：
- Phrase-arch 同時影響 velocity 和 tempo
- 樂句中間較快**且**較大聲
- 這是真實演奏的特徵

---

### `src/audio/humanize/rules/dynamics_tempo.py` ⭐ 新增

**職責**：Dynamics marking → tempo 調整（velocity-tempo 耦合）

**KTH 原理**：人類演奏中，dynamics 和 tempo 高度相關：
- crescendo 時通常會 accelerando
- diminuendo 時通常會 ritardando
- sf/accent 會有 agogic accent（微微延遲）

**包含**：
```python
class CrescendoTempoRule(Rule):
    """
    Crescendo/diminuendo affects tempo.

    Based on KTH research: dynamics and tempo are coupled.
    - Crescendo → slight accelerando
    - Diminuendo → slight ritardando
    """

    def __init__(self, config: RuleConfig, max_tempo_change: float = 0.1):
        super().__init__(config)
        self.max_tempo_change = max_tempo_change  # 10% max

    def apply_timing(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0

        # loudness_incr from partitura's loudness_direction_basis
        loudness_change = features.get('loudness_incr', 0) - features.get('loudness_decr', 0)

        # Positive = crescendo = faster = negative timing offset
        # Effect scales with k
        tempo_ratio = self.k * self.max_tempo_change * loudness_change
        beat_duration = features.get('beat_duration', 0.5)

        return -tempo_ratio * beat_duration  # Negative = earlier


class AgogicAccentRule(Rule):
    """
    Agogic accent: accented notes are slightly delayed.

    Creates emphasis through timing, not just velocity.
    """

    def __init__(self, config: RuleConfig, delay_ms: float = 20):
        super().__init__(config)
        self.delay_ms = delay_ms

    def apply_timing(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0

        # Check for accent marking from articulation_basis
        has_accent = features.get('accent', 0) > 0.5
        has_sf = features.get('sf', 0) > 0.5 or features.get('sfz', 0) > 0.5

        if has_accent or has_sf:
            return self.k * self.delay_ms / 1000

        return 0.0
```

**HumanizationConfig 設定**：
```python
# Dynamics-tempo coupling
crescendo_tempo: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.3, 1.5)
))
crescendo_tempo_max_change: float = 0.1  # ±10% tempo

# Agogic accent
agogic_accent: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.5, 1.5)
))
agogic_delay_ms: float = 20.0
```

---

### `src/audio/humanize/rules/articulation_tempo.py` ⭐ 新增

**職責**：Articulation → tempo 微調

**包含**：
```python
class ArticulationTempoRule(Rule):
    """
    Articulation affects local tempo feel.

    - Legato passages: slightly slower, more connected
    - Staccato passages: can feel slightly faster
    - Tenuto: slight lengthening and delay of next note
    """

    def __init__(self, config: RuleConfig):
        super().__init__(config)

    def apply_timing(self, note, features: dict) -> float:
        if not self.enabled:
            return 0.0

        # Tenuto: hold slightly longer, delay next
        if features.get('tenuto', False):
            return self.k * 0.015  # +15ms

        # Legato context: slightly broader timing
        if features.get('in_slur', False):
            return self.k * 0.005  # +5ms tendency

        return 0.0
```

**HumanizationConfig 設定**：
```python
articulation_tempo: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.5, 1.5)
))

---

### Rubato 在 HumanizationConfig 中的設定

```python
# 在 HumanizationConfig 中新增：

# Phrase rubato: tempo variation within phrases
phrase_rubato: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.5, 1.5)
))

# Beat rubato: micro tempo fluctuations
beat_rubato: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=0.8, k_range=(0.3, 1.2)  # Default slightly less than 1
))
```

---

### `src/audio/humanize/rules/fermata.py` ⭐ 新增

**職責**：實作 Fermata（延長記號）處理

**參考**：BasisMixer 的 `fermata_basis`

**包含**：
```python
class FermataRule(Rule):
    """
    Fermata: extend note duration and add pause after.

    Typical fermata effect:
    - Note duration × 1.5-2.5 (depends on context)
    - Small pause (breath) after fermata note
    - Often accompanied by ritardando leading into fermata
    """

    def __init__(self, config: RuleConfig,
                 duration_multiplier: float = 2.0,
                 pause_beats: float = 0.5):
        super().__init__(config)
        self.duration_multiplier = duration_multiplier
        self.pause_beats = pause_beats

    def apply_duration(self, note, features: dict) -> float:
        if not self.enabled:
            return 1.0
        if not features.get('has_fermata', False):
            return 1.0
        # Extend duration
        return 1.0 + self.k * (self.duration_multiplier - 1.0)

    def apply_timing(self, note, features: dict) -> float:
        """Add pause AFTER fermata note (affects subsequent notes)."""
        if features.get('after_fermata', False):
            beat_duration = features.get('beat_duration', 0.5)
            return self.k * self.pause_beats * beat_duration
        return 0.0
```

**HumanizationConfig 設定**：
```python
fermata: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.7, 1.5)
))
fermata_duration_multiplier: float = 2.0
fermata_pause_beats: float = 0.5
```

---

### `src/audio/humanize/rules/ornaments.py` ⭐ 新增

**職責**：處理裝飾音（Grace notes, Trills, Mordents）

**包含**：
```python
class GraceNoteRule(Rule):
    """
    Grace notes: play before the beat, steal time from previous note.

    Two styles:
    - Acciaccatura (斜線): very short, "crushed" into main note
    - Appoggiatura (無斜線): longer, more expressive
    """

    def __init__(self, config: RuleConfig,
                 acciaccatura_ms: float = 50,
                 appoggiatura_ratio: float = 0.25):
        super().__init__(config)
        self.acciaccatura_ms = acciaccatura_ms
        self.appoggiatura_ratio = appoggiatura_ratio

    def compute_grace_timing(self, grace_note, main_note, features: dict) -> dict:
        """
        Compute timing for grace note.

        Returns dict with:
        - grace_onset: when grace note starts
        - grace_duration: how long grace note lasts
        - main_onset_shift: how much main note is delayed (usually 0)
        """
        if features.get('is_acciaccatura', True):
            # Short, before the beat
            grace_duration = self.k * self.acciaccatura_ms / 1000
            grace_onset = main_note.onset - grace_duration
            return {
                'grace_onset': grace_onset,
                'grace_duration': grace_duration,
                'main_onset_shift': 0
            }
        else:
            # Appoggiatura: takes time from main note
            beat_duration = features.get('beat_duration', 0.5)
            grace_duration = self.k * self.appoggiatura_ratio * beat_duration
            return {
                'grace_onset': main_note.onset,
                'grace_duration': grace_duration,
                'main_onset_shift': grace_duration
            }


class TrillRule(Rule):
    """
    Trills: rapid alternation between main note and upper neighbor.

    Parameters:
    - trill_speed: notes per second (typically 6-12)
    - start_on_upper: whether to start on upper note (Baroque) or main (Romantic)
    """

    def __init__(self, config: RuleConfig,
                 trill_speed: float = 8.0,
                 start_on_upper: bool = False):
        super().__init__(config)
        self.trill_speed = trill_speed
        self.start_on_upper = start_on_upper

    def expand_trill(self, note, features: dict) -> List[dict]:
        """
        Expand a trilled note into alternating pitches.

        Returns list of {pitch, onset, duration, velocity} dicts.
        """
        if not features.get('has_trill', False):
            return [{'pitch': note.pitch, 'onset': note.onset,
                     'duration': note.duration, 'velocity': note.velocity}]

        notes = []
        current_time = note.onset
        end_time = note.onset + note.duration
        note_duration = 1.0 / (self.k * self.trill_speed)

        upper_pitch = note.pitch + features.get('trill_interval', 2)  # Usually whole/half step
        is_upper = self.start_on_upper

        while current_time < end_time - note_duration * 0.5:
            pitch = upper_pitch if is_upper else note.pitch
            dur = min(note_duration, end_time - current_time)
            notes.append({
                'pitch': pitch,
                'onset': current_time,
                'duration': dur,
                'velocity': note.velocity - (5 if is_upper else 0)  # Upper slightly softer
            })
            current_time += note_duration
            is_upper = not is_upper

        return notes


class MordentRule(Rule):
    """Mordent: quick alternation (main-upper-main or main-lower-main)."""

    def expand_mordent(self, note, features: dict) -> List[dict]:
        """Expand mordent into 3 notes."""
        if not features.get('has_mordent', False):
            return [{'pitch': note.pitch, 'onset': note.onset,
                     'duration': note.duration, 'velocity': note.velocity}]

        mordent_duration = self.k * 0.08  # ~80ms total for ornament
        single_note_dur = mordent_duration / 3

        is_upper = features.get('mordent_type', 'upper') == 'upper'
        aux_pitch = note.pitch + (2 if is_upper else -2)

        return [
            {'pitch': note.pitch, 'onset': note.onset,
             'duration': single_note_dur, 'velocity': note.velocity},
            {'pitch': aux_pitch, 'onset': note.onset + single_note_dur,
             'duration': single_note_dur, 'velocity': note.velocity - 5},
            {'pitch': note.pitch, 'onset': note.onset + 2 * single_note_dur,
             'duration': note.duration - 2 * single_note_dur, 'velocity': note.velocity},
        ]
```

**HumanizationConfig 設定**：
```python
grace_note: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.7, 1.3)
))
acciaccatura_ms: float = 50.0
appoggiatura_ratio: float = 0.25

trill: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.8, 1.2)
))
trill_speed: float = 8.0  # notes per second
trill_start_on_upper: bool = False

mordent: RuleConfig = field(default_factory=lambda: RuleConfig(
    k=1.0, k_range=(0.8, 1.2)
))
```

---

### `src/audio/humanize/rules/tempo.py` ⭐ 新增

**職責**：解析 Tempo marking，設定基礎速度

**包含**：
```python
# Standard tempo ranges (BPM)
TEMPO_MARKINGS = {
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

    def __init__(self, default_bpm: float = 108):
        self.default_bpm = default_bpm

    def get_base_tempo(self, marking: Optional[str], rng=None) -> float:
        """
        Get base BPM from tempo marking.

        If rng provided, randomly sample within the marking's range.
        """
        if marking is None:
            return self.default_bpm

        marking_lower = marking.lower().strip()

        # Check for explicit BPM (e.g., "♩= 120")
        if '=' in marking_lower:
            try:
                bpm = float(marking_lower.split('=')[1].strip())
                return bpm
            except:
                pass

        # Check known markings
        for name, (low, high) in TEMPO_MARKINGS.items():
            if name in marking_lower:
                if rng is not None:
                    return rng.uniform(low, high)
                return (low + high) / 2

        return self.default_bpm

    def get_tempo_from_score(self, part) -> float:
        """Extract tempo marking from partitura Part."""
        # Look for tempo indications in the score
        for direction in part.iter_all(partitura.score.Tempo):
            return direction.bpm
        for direction in part.iter_all(partitura.score.Direction):
            if hasattr(direction, 'text'):
                bpm = self.get_base_tempo(direction.text)
                if bpm != self.default_bpm:
                    return bpm
        return self.default_bpm
```

**使用方式**（在 Engine 中）：
```python
class HumanizationEngine:
    def __init__(self, config: HumanizationConfig):
        self.config = config
        self.tempo_interpreter = TempoInterpreter(
            default_bpm=config.default_bpm
        )

    def humanize(self, score_path: str, output_path: str, ...):
        part = load_score(score_path)

        # Get base tempo from score markings
        base_bpm = self.tempo_interpreter.get_tempo_from_score(part)

        # Can also randomize within marking range for augmentation
        if self.config.randomize_tempo:
            base_bpm = self.tempo_interpreter.get_base_tempo(
                marking, rng=self.rng
            )
```

**HumanizationConfig 設定**：
```python
default_bpm: float = 108  # BasisMixer default
randomize_tempo: bool = True  # Sample within marking range
tempo_variation_range: Tuple[float, float] = (0.9, 1.1)  # ±10% variation
```

---

### `src/audio/humanize/rules/timing.py`

**職責**：Micro-timing jitter + chord asynchrony

**包含**：
```python
class MicroTimingRule(Rule):
    """Micro-timing: small random timing variations."""

    def __init__(self, config: RuleConfig, std_ms: float = 15.0):
        super().__init__(config)
        self.std_ms = std_ms
        self.rng = None  # Set by engine

    def apply_timing(self, note, features: dict) -> float:
        jitter = self.rng.normal(0, self.std_ms / 1000)
        return self.k * jitter


class ChordAsyncRule(Rule):
    """Chord asynchrony: melody leads bass."""

    def __init__(self, config: RuleConfig, lead_ms: float = 25.0):
        super().__init__(config)
        self.lead_ms = lead_ms

    def apply_timing(self, note, features: dict) -> float:
        if features.get('is_melody', False):
            return -self.k * self.lead_ms / 1000  # Negative = earlier
        return 0.0
```

---

### `src/audio/humanize/rules/articulation.py`

**職責**：Staccato/legato duration 調整

**包含**：
```python
class StaccatoRule(Rule):
    """Staccato: shorten note duration."""

    def apply_duration(self, note, features: dict) -> float:
        if features.get('articulation') == 'staccato':
            return 1.0 - self.k * 0.5  # 50% shorter at k=1
        return 1.0


class LegatoRule(Rule):
    """Legato: overlap notes slightly."""

    def __init__(self, config: RuleConfig, overlap_ms: float = 30.0):
        super().__init__(config)
        self.overlap_ms = overlap_ms

    def apply_duration(self, note, features: dict) -> float:
        if features.get('in_slur', False):
            # Extend by overlap amount
            base_dur = note.duration
            overlap_ratio = (self.k * self.overlap_ms / 1000) / base_dur
            return 1.0 + overlap_ratio
        return 1.0
```

---

### `src/audio/humanize/rules/pedal.py`

**職責**：自動踏板生成

**包含**：
```python
class AutoPedalRule(Rule):
    """Auto pedal: syncopated pedaling based on harmony changes."""

    def __init__(self, config: RuleConfig,
                 lift_before_ms: float = 30.0,
                 press_after_ms: float = 20.0):
        super().__init__(config)
        self.lift_before_ms = lift_before_ms
        self.press_after_ms = press_after_ms

    def generate_pedal_events(self, notes, features_list) -> List[PedalEvent]:
        """Generate CC64 events for sustain pedal."""
        events = []
        # Detect harmony changes from features
        # Lift pedal before change, press after
        ...
        return events
```

---

### `src/audio/humanize/engine.py`

**職責**：整合所有規則，執行 humanization pipeline

**包含**：
```python
class HumanizationEngine:
    """Main engine that applies all KTH rules to generate humanized MIDI."""

    def __init__(self, config: HumanizationConfig):
        self.config = config
        self._init_rules()

    def _init_rules(self):
        """Initialize all rule instances from config."""
        cfg = self.config

        # Velocity rules
        self.velocity_rules = [
            HighLoudRule(cfg.high_loud),
            PhraseArchRule(cfg.phrase_arch, cfg.phrase_peak_position),
            DurationContrastRule(cfg.duration_contrast),
            MelodicChargeRule(cfg.melodic_charge, cfg.non_chord_tone_boost_dB, cfg.leading_tone_boost_dB),
        ]

        # Timing rules
        self.timing_rules = [
            PhraseRubatoRule(cfg.phrase_rubato, cfg.phrase_peak_position),
            BeatRubatoRule(cfg.beat_rubato),
            FinalRitardRule(cfg.final_ritard, cfg.final_ritard_start),
            MicroTimingRule(cfg.timing_jitter, cfg.timing_jitter_std_ms),
            ChordAsyncRule(cfg.chord_async, cfg.melody_lead_base_ms),
            FermataRule(cfg.fermata, cfg.fermata_duration_multiplier, cfg.fermata_pause_beats),
            # Velocity-tempo coupling
            CrescendoTempoRule(cfg.crescendo_tempo, cfg.crescendo_tempo_max_change),
            AgogicAccentRule(cfg.agogic_accent, cfg.agogic_delay_ms),
            ArticulationTempoRule(cfg.articulation_tempo),
            # Phrasing
            PunctuationRule(cfg.punctuation, cfg.micropause_ms, cfg.phrase_end_shorten_ratio),
            LeapRule(cfg.leap, cfg.leap_threshold_semitones, cfg.leap_duration_effect, cfg.leap_micropause_ms),
            RepetitionRule(cfg.repetition),
        ]

        # Safety rules
        self.social_care = SocialDurationCareRule(cfg.social_duration_care, cfg.min_audible_duration_ms)
        self.normalizer = GlobalNormalizer(
            cfg.target_rms_velocity, cfg.max_velocity, cfg.soft_clip_threshold
        )

        # Articulation rules
        self.articulation_rules = [
            StaccatoRule(cfg.staccato),
            LegatoRule(cfg.legato, cfg.legato_overlap_base_ms),
        ]

        # Ornament handlers
        self.ornament_rules = [
            GraceNoteRule(cfg.grace_note, cfg.acciaccatura_ms, cfg.appoggiatura_ratio),
            TrillRule(cfg.trill, cfg.trill_speed, cfg.trill_start_on_upper),
            MordentRule(cfg.mordent),
        ]

        # Special
        self.pedal_rule = AutoPedalRule(cfg.pedal, cfg.pedal_lift_before_ms, cfg.pedal_press_after_ms)
        self.tempo_interpreter = TempoInterpreter(cfg.default_bpm)

    def humanize(self, score_path: str, output_path: str, format: str = 'kern'):
        """Main entry point: Score → Humanized MIDI."""
        # 1. Load score with partitura
        # 2. Extract features (basis functions)
        # 3. Apply all rules (additive in dB space)
        # 4. Generate pedal events
        # 5. Write MIDI with mido

    def _extract_features(self, part) -> List[dict]:
        """Extract per-note features using partitura basis functions."""

    def _compute_final_velocity(self, note, features: dict) -> int:
        """Apply all velocity rules (additive in dB)."""
        base_dB = velocity_to_dB(self._get_base_velocity(note, features))
        for rule in self.rules:
            base_dB += rule.apply_velocity(note, features)
        return dB_to_velocity(base_dB)
```

**依賴**：`partitura`, `mido`, 所有 rule 模組

---

### `src/audio/humanize/presets.py`

**職責**：預設風格設定

**包含**：
```python
# Romantic style: more expressive
ROMANTIC = HumanizationConfig(
    high_loud=RuleConfig(k=1.2, k_range=(0.8, 1.5)),
    phrase_arch=RuleConfig(k=1.3, k_range=(0.8, 1.8)),
    timing_jitter=RuleConfig(k=1.2, k_range=(0.8, 1.5)),
    final_ritard=RuleConfig(k=1.5, k_range=(1.0, 2.0)),
    ...
)

# Classical style: more restrained
CLASSICAL = HumanizationConfig(
    high_loud=RuleConfig(k=0.8, k_range=(0.5, 1.0)),
    phrase_arch=RuleConfig(k=0.8, k_range=(0.5, 1.2)),
    timing_jitter=RuleConfig(k=0.7, k_range=(0.5, 1.0)),
    final_ritard=RuleConfig(k=0.8, k_range=(0.5, 1.2)),
    ...
)

# Balanced (default)
BALANCED = HumanizationConfig()  # All k=1.0
```

---

## 核心類別設計：k 值系統

### 設計哲學

每條規則都有一個 **k 值**控制強度，這是 data augmentation 的核心：

```
最終效果 = Σ (規則_i 的基礎效果 × k_i)
```

- 隨機化 k 值 → 產生不同「演奏風格」的訓練資料
- k 值範圍定義了合理的音樂表達空間
- 所有規則效果疊加（additive），不會互相覆蓋

### RuleConfig（單一規則設定）

```python
@dataclass
class RuleConfig:
    """Configuration for a single KTH-style rule."""
    k: float = 1.0              # Current k value
    k_range: Tuple[float, float] = (0.5, 1.5)  # Randomization range
    enabled: bool = True        # Toggle rule on/off

    def randomize(self, rng: np.random.Generator) -> 'RuleConfig':
        """Return new config with randomized k within range."""
        new_k = rng.uniform(self.k_range[0], self.k_range[1])
        return RuleConfig(k=new_k, k_range=self.k_range, enabled=self.enabled)
```

### HumanizationConfig（完整設定）

```python
@dataclass
class HumanizationConfig:
    """
    KTH-style humanization configuration with k-value system.

    All effects are ADDITIVE (疊加), referenced to:
    - 0 dB = MIDI velocity 64 (KTH standard)
    - BasisMixer default velocity = 55
    """

    # === Global Settings ===
    reference_velocity: int = 64      # 0 dB reference point
    default_velocity: int = 55        # BasisMixer default (mf)

    # === Dynamics Rules ===
    # Dynamic marking → velocity mapping (not affected by k)
    dynamics_map: Dict[str, int] = field(default_factory=lambda: {
        'ppp': 20, 'pp': 35, 'p': 50,
        'mp': 60, 'mf': 70,
        'f': 85, 'ff': 100, 'fff': 115,
        'sf': 95, 'sfz': 100, 'fp': 85,
    })

    # === KTH Rules (each with k value) ===

    # High-loud: pitch 越高 → 越大聲
    # Effect: +k × 0.5 dB per semitone above middle C
    high_loud: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.3, 1.5)
    ))

    # Phrase-arch: 樂句中間強，兩端弱
    # Effect: ±k × 6 dB at phrase peak/boundaries
    phrase_arch: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    phrase_peak_position: float = 0.6  # 0-1, where peak occurs

    # Duration-contrast: 長音更長更大聲
    # Effect: ±k × 3 dB based on relative duration
    duration_contrast: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))

    # Melodic-charge: 強調旋律張力音
    # Effect: +k × 4 dB for non-chord tones
    melodic_charge: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.0, 1.5)
    ))

    # Final-ritard: 結尾漸慢
    # Effect: tempo × (1 - k × sqrt(position)) in final 10%
    final_ritard: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 2.0)
    ))
    final_ritard_start: float = 0.9   # Start position (0-1)

    # === Timing Rules ===

    # Micro-timing jitter
    # Effect: ±k × 15ms gaussian noise
    timing_jitter: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    timing_jitter_std_ms: float = 15.0  # Base std in ms

    # Chord asynchrony (melody lead)
    # Effect: melody leads by k × 25ms
    chord_async: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 2.0)
    ))
    melody_lead_base_ms: float = 25.0

    # Metrical accent
    # Effect: +k × 5 velocity on downbeats
    metrical_accent: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    downbeat_boost_base: int = 5

    # === Articulation Rules ===

    # Staccato shortening
    # Effect: duration × (1 - k × 0.5) for staccato notes
    staccato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.7, 1.3)
    ))

    # Legato overlap
    # Effect: +k × 30ms overlap for legato phrases
    legato: RuleConfig = field(default_factory=lambda: RuleConfig(
        k=1.0, k_range=(0.5, 1.5)
    ))
    legato_overlap_base_ms: float = 30.0

    # === Pedal Settings ===
    pedal_enabled: bool = True
    pedal_lift_before_ms: float = 30.0   # Lift before chord change
    pedal_press_after_ms: float = 20.0   # Press after new chord

    # === Methods ===

    def randomize(self, seed: Optional[int] = None) -> 'HumanizationConfig':
        """
        Create a new config with all k values randomized within their ranges.
        This is the core of data augmentation diversity.
        """
        rng = np.random.default_rng(seed)

        return HumanizationConfig(
            # Keep global settings
            reference_velocity=self.reference_velocity,
            default_velocity=self.default_velocity,
            dynamics_map=self.dynamics_map.copy(),

            # Randomize all rule k values
            high_loud=self.high_loud.randomize(rng),
            phrase_arch=self.phrase_arch.randomize(rng),
            phrase_peak_position=rng.uniform(0.5, 0.7),
            duration_contrast=self.duration_contrast.randomize(rng),
            melodic_charge=self.melodic_charge.randomize(rng),
            final_ritard=self.final_ritard.randomize(rng),
            final_ritard_start=rng.uniform(0.85, 0.95),
            timing_jitter=self.timing_jitter.randomize(rng),
            timing_jitter_std_ms=self.timing_jitter_std_ms,
            chord_async=self.chord_async.randomize(rng),
            melody_lead_base_ms=self.melody_lead_base_ms,
            metrical_accent=self.metrical_accent.randomize(rng),
            downbeat_boost_base=self.downbeat_boost_base,
            staccato=self.staccato.randomize(rng),
            legato=self.legato.randomize(rng),
            legato_overlap_base_ms=self.legato_overlap_base_ms,
            pedal_enabled=self.pedal_enabled,
            pedal_lift_before_ms=self.pedal_lift_before_ms,
            pedal_press_after_ms=self.pedal_press_after_ms,
        )

    def to_dict(self) -> Dict:
        """Export config for logging/reproducibility."""
        return {
            'high_loud_k': self.high_loud.k,
            'phrase_arch_k': self.phrase_arch.k,
            'duration_contrast_k': self.duration_contrast.k,
            'melodic_charge_k': self.melodic_charge.k,
            'final_ritard_k': self.final_ritard.k,
            'timing_jitter_k': self.timing_jitter.k,
            'chord_async_k': self.chord_async.k,
            'metrical_accent_k': self.metrical_accent.k,
            'staccato_k': self.staccato.k,
            'legato_k': self.legato.k,
        }
```

### k 值隨機化範例

```python
# 產生 4 個不同風格的 humanized 版本
base_config = HumanizationConfig()

for i in range(4):
    config = base_config.randomize(seed=i)
    print(f"Version {i}: {config.to_dict()}")

# Output example:
# Version 0: {'high_loud_k': 0.82, 'phrase_arch_k': 1.23, ...}
# Version 1: {'high_loud_k': 1.31, 'phrase_arch_k': 0.67, ...}
# Version 2: {'high_loud_k': 0.45, 'phrase_arch_k': 1.45, ...}
# Version 3: {'high_loud_k': 1.12, 'phrase_arch_k': 0.89, ...}
```

### HumanizationEngine

```python
class HumanizationEngine:
    """
    Apply KTH-style humanization rules with k-value system.

    All velocity effects are computed in dB, then converted to MIDI velocity.
    Effects are additive: final_dB = base_dB + Σ(rule_effect_i × k_i)
    """

    def __init__(self, config: HumanizationConfig):
        self.config = config

    def humanize_from_score(
        self,
        score_path: str,
        output_midi_path: str,
        format: str = 'kern'
    ) -> MidiFile:
        """Main entry point: Score → Humanized MIDI."""
        ...

    def _apply_velocity_rules(self, note, features) -> int:
        """
        Apply all velocity-affecting rules (additive in dB space).

        Returns: final MIDI velocity (1-127)
        """
        cfg = self.config

        # Start from dynamic marking or default
        base_vel = self._get_dynamic_velocity(note)
        base_dB = self._velocity_to_dB(base_vel)

        # Add rule effects (all in dB)
        dB_delta = 0.0

        # High-loud: +0.5 dB per semitone above C4
        if cfg.high_loud.enabled:
            semitones_above_c4 = note.pitch - 60
            dB_delta += cfg.high_loud.k * 0.5 * semitones_above_c4

        # Phrase-arch: based on position in phrase
        if cfg.phrase_arch.enabled and features.get('phrase_position') is not None:
            arch_effect = self._compute_phrase_arch(features['phrase_position'])
            dB_delta += cfg.phrase_arch.k * arch_effect

        # Duration-contrast: long notes louder
        if cfg.duration_contrast.enabled:
            dur_effect = self._compute_duration_effect(note, features)
            dB_delta += cfg.duration_contrast.k * dur_effect

        # ... more rules ...

        # Convert back to velocity
        final_dB = base_dB + dB_delta
        return self._dB_to_velocity(final_dB)

    def _velocity_to_dB(self, velocity: int) -> float:
        """Convert MIDI velocity to dB (0 dB = velocity 64)."""
        # KTH uses polynomial approximation
        return 20 * np.log10(velocity / 64 + 1e-6)

    def _dB_to_velocity(self, dB: float) -> int:
        """Convert dB back to MIDI velocity, clamped to 1-127."""
        velocity = int(64 * (10 ** (dB / 20)))
        return max(1, min(127, velocity))
```

---

## Partitura Basis Functions 對應表

### 從樂譜讀取的 Features

每個規則需要的 features 都來自 partitura 的 `make_note_feats()`：

| 規則 | 需要的 Feature | Partitura Basis |
|------|----------------|-----------------|
| HighLoud | `pitch` | 直接從 note 取得 |
| PhraseArch | `phrase_position`, `slur_incr`, `slur_decr` | `slur_basis` |
| DurationContrast | `duration`, `relative_duration` | `duration_basis` |
| **MelodicCharge** | `is_non_chord_tone`, `is_leading_tone` | 需要和聲分析 (music21) |
| **CrescendoTempo** | `loudness_incr`, `loudness_decr` | `loudness_direction_basis` |
| **AgogicAccent** | `accent`, `sf`, `sfz` | `articulation_basis`, `loudness_direction_basis` |
| **ArticulationTempo** | `tenuto`, `in_slur` | `articulation_basis`, `slur_basis` |
| **Punctuation** | `is_phrase_end`, `is_phrase_start` | `slur_basis` + 計算 |
| **Leap** | `interval_to_next`, `interval_from_prev` | 計算相鄰音高差 |
| **Repetition** | `is_repeated_note` | 計算相鄰音高相同 |
| Staccato | `staccato` | `articulation_basis` |
| Fermata | `fermata` | `fermata_basis` |
| MicroTiming | `beat_strength`, `is_downbeat` | `metrical_basis` |
| GraceNote | `is_grace`, `is_acciaccatura` | `grace_basis` |
| Trill | `has_trill`, `trill_interval` | 需要額外解析 |

### Engine 中的 Feature 提取

```python
class HumanizationEngine:
    # 需要的所有 basis functions
    REQUIRED_BASIS_FUNCTIONS = [
        'polynomial_pitch_basis',      # pitch, pitch², pitch³
        'loudness_direction_basis',    # p, f, mf, cresc, dim, sf
        'articulation_basis',          # accent, staccato, tenuto
        'duration_basis',              # note duration
        'slur_basis',                  # slur_incr, slur_decr (phrase)
        'fermata_basis',               # fermata
        'grace_basis',                 # grace notes
        'metrical_basis',              # beat positions
    ]

    def _extract_features(self, part) -> List[dict]:
        """Extract per-note features using partitura."""
        import partitura.musicanalysis as ma

        # Get basis function matrix
        basis_matrix, basis_names = ma.make_note_feats(
            part, self.REQUIRED_BASIS_FUNCTIONS
        )

        # Convert to per-note feature dicts
        features_list = []
        for i, note in enumerate(part.notes_tied):
            features = {
                name: basis_matrix[i, j]
                for j, name in enumerate(basis_names)
            }
            # Add computed features
            features['phrase_position'] = self._compute_phrase_position(i, features)
            features['piece_position'] = i / len(part.notes_tied)
            features['beat_duration'] = 60 / self.base_bpm
            features_list.append(features)

        return features_list
```

### Dynamics Marking 直接讀取

```python
def _get_base_velocity(self, note, features: dict) -> int:
    """Get base velocity from dynamics marking."""
    # Check loudness_direction_basis features
    for marking, velocity in self.config.dynamics_map.items():
        feature_name = f'loudness_direction_basis.{marking}'
        if features.get(feature_name, 0) > 0.5:
            return velocity

    return self.config.default_velocity
```

---

## 可重現性 (Reproducibility)

### Metadata Logging

每個 humanized MIDI 都要記錄完整設定，確保可重現：

```python
@dataclass
class HumanizationMetadata:
    """Metadata for reproducibility."""
    source_file: str           # Original kern/midi path
    version: int               # Augmentation version (0-3)
    seed: int                  # Random seed
    timestamp: str             # ISO format
    k_values: Dict[str, float] # All k values
    base_config: Dict          # HumanizationConfig settings
    soundfont: Optional[str]   # If rendered
    partitura_version: str
    humanize_version: str      # Module version


class HumanizationEngine:
    def humanize(self, ...) -> Tuple[MidiFile, HumanizationMetadata]:
        ...
        metadata = HumanizationMetadata(
            source_file=str(score_path),
            version=version_idx,
            seed=self.config.seed,
            timestamp=datetime.now().isoformat(),
            k_values=self.config.to_dict(),
            base_config=self.config.to_full_dict(),
            partitura_version=pt.__version__,
            humanize_version=__version__,
        )
        return midi, metadata
```

**存放方式**：同名 `.json` sidecar file

```
output/
├── chopin_op10_no1_v0.mid
├── chopin_op10_no1_v0.json    # ← Metadata
├── chopin_op10_no1_v1.mid
├── chopin_op10_no1_v1.json
...
```

**重現特定版本**：
```python
# From metadata JSON
metadata = json.load(open('chopin_op10_no1_v2.json'))
config = HumanizationConfig.from_dict(metadata['base_config'])
config.apply_k_values(metadata['k_values'])
# Guaranteed identical output
```

---

## 錯誤處理 & Graceful Degradation

**原則**：單首失敗不應拖累整個 pipeline

### Rule-level Fallback

每個 rule 都應該能容錯：

```python
class Rule(ABC):
    def apply_velocity(self, note, features: dict) -> float:
        try:
            return self._apply_velocity_impl(note, features)
        except Exception as e:
            logging.warning(f"{self.__class__.__name__} failed: {e}")
            return 0.0  # No effect on failure
```

### Feature-level Fallback

缺少某些 features 時使用預設值：

```python
def _extract_features(self, part):
    # Try full basis functions
    try:
        basis_matrix, names = make_note_feats(part, self.REQUIRED_BASIS)
    except Exception as e:
        logging.warning(f"Full basis failed, using critical only: {e}")
        basis_matrix, names = make_note_feats(part, self.CRITICAL_BASES)

    # Fill missing with defaults
    for features in features_list:
        features.setdefault('fermata', 0.0)
        features.setdefault('slur_incr', 0.0)
        features.setdefault('beat_strength', 0.5)
        ...
```

### File-level Fallback

單首處理失敗時記錄並繼續：

```python
def process_batch(kern_paths):
    results = []
    failed = []

    for kern in kern_paths:
        try:
            result = humanize(kern)
            results.append(result)
        except Exception as e:
            logging.error(f"Failed to process {kern}: {e}")
            failed.append((kern, str(e)))
            continue  # Don't stop the whole batch

    # Save failed list for debugging
    save_failed_list(failed, 'humanize_failures.txt')
    return results
```

---

## Valid/Test 處理

**決策**：Train 先做，Valid/Test 暫緩

| Split | 處理方式 | 理由 |
|-------|----------|------|
| **Train** | ✅ Humanize (4 versions, randomize k) | 需要多樣性 |
| **Valid** | ⏸️ 暫不處理 | 需要設計 evaluation 策略 |
| **Test** | ⏸️ 暫不處理 | 需要設計 evaluation 策略 |

**後續考量**：
- Valid/Test 是否也要 humanize？
- 如果是，用 fixed k=1.0 還是也 randomize？
- 評估指標如何設計？

這些問題等 Train 跑完、模型訓練後再決定。

---

## Data Augmentation 整合

### k 值系統 × 多 SoundFont = 訓練多樣性

```python
def generate_augmented_versions(
    kern_path: str,
    output_dir: str,
    n_versions: int = 4,
    soundfonts: List[str] = SOUNDFONTS,
) -> List[str]:
    """
    Generate multiple humanized versions of a score.

    Total outputs = n_versions × len(soundfonts)
    Example: 4 k-value variants × 4 soundfonts = 16 audio files per score
    """
    base_config = HumanizationConfig()
    engine = HumanizationEngine(base_config)
    outputs = []

    for version_idx in range(n_versions):
        # Randomize k values for this version
        config = base_config.randomize(seed=version_idx)
        engine.config = config

        # Generate humanized MIDI
        midi_path = output_dir / f"{kern_path.stem}_v{version_idx}.mid"
        engine.humanize_from_score(kern_path, midi_path)

        # Log k values for reproducibility
        log_config(midi_path, config.to_dict())

        # Render with each soundfont
        for sf_name in soundfonts:
            wav_path = output_dir / f"{kern_path.stem}_v{version_idx}~{sf_name}.wav"
            render_midi_to_audio(midi_path, wav_path, soundfont=sf_name)
            outputs.append(wav_path)

    return outputs
```

### 與 clef-piano-base 的差異

| 面向 | clef-piano-base | clef-piano-full |
|------|-----------------|-----------------|
| Velocity | Uniform (90) | k 值系統 humanization |
| Timing | 機械化 | Micro-timing + ritardando |
| Articulation | 無 | Staccato/legato rules |
| Pedal | 無 | Auto pedaling |
| 多樣性來源 | 僅 SoundFont | k 值 × SoundFont |
| 用途 | Zeng baseline 比較 | 主要訓練資料 |

---

## 延伸規則（clef-solo / clef-tutti 用）

以下規則對 **Piano Solo 不需要**，但對其他樂器有用：

### Intonation 規則（非固定音高樂器）

| 規則 | 說明 | 適用樂器 |
|------|------|----------|
| **High-sharp** | 高音微升 | 弦樂、管樂 |
| **Melodic-intonation** | 旋律律：導音升高 | 弦樂 |
| **Harmonic-intonation** | 和聲律：純律調整 | 弦樂合奏 |

### Ensemble 規則（多樂器）

| 規則 | 說明 | 適用情境 |
|------|------|----------|
| **Bar-sync** | 小節線對齊 | 管弦樂 |
| **Melodic-sync** | 旋律聲部同步 | 室內樂 |
| **Ensemble-swing** | 合奏 swing 比例 | Jazz ensemble |

### 其他可選規則

| 規則 | 說明 | 優先級 |
|------|------|--------|
| **Faster-uphill** | 上行加速 | ⚪ 可選 |
| **Inégales** | Baroque swing | ⚪ 古典不需要 |
| **Harmonic-charge** | 遠離調性的和弦強調 | ⚪ 需要複雜和聲分析 |

這些規則的 spec 可以在實作 clef-solo/tutti 時再補上。

---



## 實作優先順序

### Phase 1: Core Infrastructure
1. `config.py` — RuleConfig + HumanizationConfig + randomize()
2. `convert.py` — dB ↔ velocity 轉換
3. `rules/base.py` — Rule 抽象基底類別
4. `rules/tempo.py` — TempoInterpreter (解析 Allegro/Andante → BPM)

### Phase 2: Velocity Rules
5. `rules/high_loud.py` — Pitch → velocity
6. `rules/phrase_arch.py` — Phrase position → velocity
7. `rules/duration_contrast.py` — Duration → velocity
8. `rules/melodic_charge.py` — 非和弦音/導音 → velocity (需和聲分析)

### Phase 3: Core Timing Rules
9. `rules/rubato.py` — PhraseRubatoRule + BeatRubatoRule
10. `rules/final_ritard.py` — 結尾漸慢
11. `rules/timing.py` — MicroTiming + ChordAsync
12. `rules/fermata.py` — Fermata 延長 + pause
13. `rules/dynamics_tempo.py` — CrescendoTempo + AgogicAccent
14. `rules/articulation_tempo.py` — Tenuto/legato timing
15. `rules/punctuation.py` — 樂句間 micropause (氣口)
16. `rules/leap.py` — 大跳 timing/duration
17. `rules/repetition.py` — 重複音 micropause

### Phase 4: Articulation + Ornaments
18. `rules/articulation.py` — Staccato/legato duration
19. `rules/ornaments.py` — GraceNote, Trill, Mordent

### Phase 5: Safety + Special
20. `rules/safety.py` — SocialDurationCare + GlobalNormalizer
21. `rules/pedal.py` — Auto pedaling (CC64)

### Phase 6: Integration
22. `engine.py` — HumanizationEngine 整合所有規則
23. `presets.py` — romantic/classical/balanced 預設

### Phase 7: Pipeline + CLI
24. 整合到 `prepare_piano_full.py`
25. `cli.py` — CLI 工具（可獨立使用）
26. 端到端測試 + 聽覺驗證

---

## 依賴管理

### 新增依賴

```toml
[tool.poetry.dependencies]
partitura = "^1.4.0"
```

### 現有依賴（已在 pyproject.toml）

| 庫 | 用途 |
|------|------|
| **mido** | 底層 MIDI 操作（tick-level），精確控制 note_on/note_off/CC |
| **music21** | Score 解析、Kern 轉換 |
| **midi2audio** | FluidSynth 包裝，MIDI→WAV |

### mido 說明

mido 是 Python MIDI 庫，比 pretty_midi 更底層：
- 直接操作 MIDI events (note_on, note_off, control_change)
- tick-level timing 精度
- 支援修改現有 MIDI 檔案

```python
import mido
from mido import MidiFile, MidiTrack, Message

# 讀取 MIDI
midi = MidiFile('input.mid')

# 修改 note velocity
for track in midi.tracks:
    for msg in track:
        if msg.type == 'note_on':
            msg.velocity = new_velocity

# 加入踏板 CC64
track.append(Message('control_change', control=64, value=127, time=0))
```

### Partitura 整合方式

```python
import partitura as pt
from partitura.musicanalysis import make_note_feats

# 載入 Kern（partitura 支援）
score = pt.load_kern('score.krn')
part = score[0]

# 提取 BasisMixer basis functions
features, names = make_note_feats(part, [
    'polynomial_pitch_basis',
    'loudness_direction_basis',
    'articulation_basis',
    'slur_basis',
    'metrical_basis',
])

# features shape: (n_notes, n_features)
# 用於 rule-based humanization
```

### Partitura vs Music21 分工

| 功能 | 工具 | 理由 |
|------|------|------|
| Kern 讀寫 | converter21 (music21) | 已整合 |
| Dynamics/Articulation 解析 | partitura | 更豐富的 basis functions |
| Phrase detection | partitura (slur_basis) | slur_incr/slur_decr |
| MIDI 輸出 | mido | tick-level 精確控制 |
| Audio 渲染 | FluidSynth | 已整合 |

---

## KTH Director Musices 官方參數

**來源**: `docs/kth_director_musices_rules.pdf`

### K 值系統
- 所有規則有 global quantity parameter **k** (預設 = 1.0)
- 規則效果是 **additive**（疊加）
- 0 dB = MIDI velocity 64（標準化參考點）

### 規則列表（Table 1 from PDF）

| 規則 | 影響變數 | 說明 |
|------|----------|------|
| **High-loud** | sl | pitch 越高 → 越大聲 |
| **Melodic-charge** | sl dr va | 強調遠離和弦根音的音符 |
| **Duration-contrast** | dr sl | 長音更長更大聲，短音更短更小聲 |
| **Score-legato-art** | dro | legato 音符重疊下一個音符 |
| **Score-staccato-art** | dro | staccato 音符加 micropause |
| **Phrase-arch** | dr sl | 弧形 tempo：慢→快→漸慢；sl 與 tempo 耦合 |
| **Final-ritard** | dr | 結尾漸慢（模型來自跑步者停止） |
| **Punctuation** | dr dro | 自動標記樂句，最後音符延長 + micropause |

### 規則 Palette 預設值（Figure 4）

| 規則 | k | 額外參數 |
|------|---|----------|
| High-loud | 1.0 | — |
| Melodic-Charge | 1.0 | :Amp 1 :Dur 1 |
| Harmonic-Charge | 1.0 | :Amp 1 :Dur 0.5 |
| Duration-Contrast | 1.0 | :Amp 0 |
| Phrase-Arch | 1.0 | :Phlevel 5 :Amp 1 :Turn 0.5 |
| Final-Ritard | 1.0 | q=3 |
| Phrase-Articulation | 1.0 | :Phlevel 5 :Subphonelevel 6 |

### dB ↔ MIDI Velocity 轉換
- 非線性（3 次多項式）
- 0 dB = velocity 64
- -15 dB ≈ velocity 18-35（視合成器）
- +10 dB ≈ velocity 100-110

---

## BasisMixer 官方參數

**來源**: `docs/basismixer_src/`

### DEFAULT_VALUES (`utils/rendering.py`)

```python
DEFAULT_VALUES = {
    'velocity': 55,           # ≈ 43% (mf)
    'velocity_trend': 55,
    'velocity_dev': 0,
    'beat_period': 0.556,     # 108 BPM
    'timing': 0,              # 無偏差
    'articulation_log': 0,    # 100% duration
}
```

### RENDER_CONFIG 參數範圍

| 參數 | min | max | 說明 |
|------|-----|-----|------|
| velocity | 20/127 (≈16) | 108/127 (≈85) | MIDI velocity |
| timing | -0.05 | +0.05 | ±50ms |
| articulation_log | -1.25 | +1.5 | duration ratio 0.42-2.83 |
| beat_period_ratio | 1/3 | 3 | tempo 變化範圍 |

### Vienna 4x22 Basis Functions (`config.json`)

用於 `partitura.musicanalysis.make_note_feats()`:

```python
BASIS_FUNCTIONS = [
    # Pitch
    'polynomial_pitch_basis',    # pitch, pitch², pitch³

    # Dynamics
    'loudness_direction_basis',  # mf, pp, p, f, sf, ff, incr, decr

    # Tempo
    'tempo_direction_basis',     # andante, lento, decr

    # Articulation
    'articulation_basis',        # accent, staccato

    # Duration & Phrase
    'duration_basis',            # note duration
    'slur_basis',                # slur_incr, slur_decr (phrase boundary)

    # Special
    'fermata_basis',             # fermata
    'grace_basis',               # grace notes

    # Metrical
    'metrical_basis',            # beat positions (3/4, 6/8, 2/4)
]
```

---

## 關鍵檔案

| 檔案 | 用途 |
|------|------|
| `src/clef/piano/prepare_zeng_pretrain.py` | 參考現有 pipeline |
| `src/audio/zeng_synthesis.py` | MIDIProcess 參考 |
| `src/score/clean_kern.py` | `strip_non_kern_spines(keep_dynam=True)` |
| `src/preprocessing/humsyn_processor.py` | `keep_dynam` 已設定 |
| `configs/clef_piano_full.yaml` | 設定檔（需擴展 humanization） |

---

## 測試策略

1. **單元測試**：每個規則模組獨立測試
2. **聽覺測試**：A/B 比較 humanized vs uniform velocity
3. **參數驗證**：確保 randomization 範圍合理
4. **端到端**：10 首曲目完整 pipeline 測試

---

## 已下載的參考資料

| 檔案 | 來源 | 說明 |
|------|------|------|
| `docs/kth_director_musices_rules.pdf` | KTH | 完整規則系統說明 |
| `docs/kth_overview_rules_2006.pdf` | Advances in Cognitive Psychology | 2006 overview paper |
| `docs/basismixer_src/` | CPJKU GitHub | BasisMixer 完整原始碼 |

### 關鍵參考檔案
- `docs/basismixer_src/utils/rendering.py` — DEFAULT_VALUES, RENDER_CONFIG
- `docs/basismixer_src/performance_codec.py` — 編碼/解碼邏輯
- `docs/basismixer_src/assets/sample_models/vienna_4x22_*/config.json` — basis functions 列表

---

## 驗證步驟

完成後執行：
```bash
# 1. 安裝新依賴
poetry add partitura

# 2. 單元測試
pytest tests/test_humanize*.py -v

# 3. 端到端測試
python -m src.clef.piano.prepare_piano_full --phase 2 --limit 10

# 4. 聽覺驗證
# 比較 output/audio/*.wav 和 baseline
```

---

## 總結

**目標**：為 clef-piano-full 實作 rule-based MIDI humanization，把 DAW 搬到 Python。

**核心哲學**：Enlightenment, not Noise — 所有轉換都是音樂知識的編碼。

**技術堆疊**：
- **Score 解析**: partitura (basis functions) + music21 (kern I/O)
- **MIDI 操作**: mido (tick-level control)
- **Audio 渲染**: FluidSynth (已整合)

**參數來源**：
- KTH Director Musices (k 值系統)
- BasisMixer (DEFAULT_VALUES, RENDER_CONFIG)

**不會影響**：clef-piano-base（完全分離的 pipeline）

---

## CLI 工具（Bonus）

### `src/audio/humanize/cli.py`

**用途**：獨立 CLI 工具，可用於個人音樂製作（Logic Pro X workflow）

```python
"""
CLI tool for MIDI humanization.

Usage:
    python -m src.audio.humanize.cli input.mid output.mid --style romantic
    python -m src.audio.humanize.cli score.krn output.mid --format kern
"""

import click
from pathlib import Path
from .engine import HumanizationEngine
from .config import HumanizationConfig
from .presets import ROMANTIC, CLASSICAL, BAROQUE, BALANCED

PRESETS = {
    'romantic': ROMANTIC,
    'classical': CLASSICAL,
    'baroque': BAROQUE,
    'balanced': BALANCED,
}

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_midi', type=click.Path())
@click.option('--style', type=click.Choice(list(PRESETS.keys())), default='balanced',
              help='Performance style preset')
@click.option('--format', type=click.Choice(['midi', 'kern', 'musicxml']), default='midi',
              help='Input file format')
@click.option('--randomize/--no-randomize', default=True,
              help='Randomize k values for variation')
@click.option('--seed', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--render', type=click.Path(),
              help='Also render to audio (requires soundfont path)')
@click.option('--verbose', '-v', is_flag=True,
              help='Print k values and processing info')
def main(input_file, output_midi, style, format, randomize, seed, render, verbose):
    """
    Humanize a MIDI file or score with KTH performance rules.

    Examples:
        humanize song.mid humanized.mid --style romantic
        humanize score.krn output.mid --format kern --seed 42
    """
    config = PRESETS[style]

    if randomize:
        config = config.randomize(seed=seed)

    if verbose:
        click.echo(f"Style: {style}")
        click.echo(f"k values: {config.to_dict()}")

    engine = HumanizationEngine(config)

    if format == 'midi':
        engine.humanize_midi(input_file, output_midi)
    else:
        engine.humanize_from_score(input_file, output_midi, format=format)

    click.echo(f"✓ Humanized: {input_file} → {output_midi}")

    if render:
        from ..zeng_synthesis import MIDIProcess
        wav_path = Path(output_midi).with_suffix('.wav')
        # Render using FluidSynth
        click.echo(f"✓ Rendered: {wav_path}")


if __name__ == '__main__':
    main()
```

### 使用範例

```bash
# 基本用法：MIDI → Humanized MIDI
python -m src.audio.humanize.cli input.mid output.mid

# 指定風格
python -m src.audio.humanize.cli input.mid output.mid --style romantic

# 從 Kern 格式轉換
python -m src.audio.humanize.cli score.krn output.mid --format kern

# 固定 seed 確保可重現
python -m src.audio.humanize.cli input.mid output.mid --seed 42 --verbose

# 同時 render 成 audio
python -m src.audio.humanize.cli input.mid output.mid --render /path/to/soundfont.sf2
```

### Logic Pro X 工作流程

```
1. 在 Logic 寫好 MIDI
2. Export 成 .mid
3. Terminal: python -m src.audio.humanize.cli song.mid humanized.mid --style romantic
4. 把 humanized.mid 拖回 Logic
5. 用 Logic 的音源播放

或者直接 render:
python -m src.audio.humanize.cli song.mid humanized.mid --render ~/soundfonts/piano.sf2
# 產出 humanized.wav，直接 import 到 Logic
```
