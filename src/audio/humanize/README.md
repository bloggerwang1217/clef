# KTH-Style MIDI Humanization

Complete implementation of KTH Director Musices rule system for MIDI humanization, designed for clef-piano-full dataset generation.

## Core Philosophy

**"Enlightenment, not Noise"** — This is not about adding random noise, but encoding musical knowledge.

All transformations are based on music theory and performance practice:
- **Velocity rules**: Pitch height, phrase position, duration contrast, melodic tension
- **Timing rules**: Rubato, ritardando, micro-timing, chord asynchrony
- **Articulation**: Staccato, legato, ornaments
- **Safety**: Prevent over-short notes and velocity clipping

## Quick Start

### Basic Usage

```python
from src.audio.humanize import HumanizationEngine, HumanizationConfig

# Use default settings
config = HumanizationConfig()
engine = HumanizationEngine(config, seed=42)

# Score → Humanized MIDI
midi_file, metadata = engine.humanize_from_score(
    score_path='chopin.krn',
    output_midi_path='chopin_humanized.mid',
    format='kern',
    version=0
)
```

### Data Augmentation (Random k values)

```python
config = HumanizationConfig()

# Generate 4 different versions
for i in range(4):
    config_i = config.randomize(seed=1000 + i)
    engine = HumanizationEngine(config_i, seed=1000 + i)
    engine.humanize_from_score(
        score_path='chopin.krn',
        output_midi_path=f'chopin_v{i}.mid',
        version=i
    )
```

## Architecture

### The k-Value System

Each rule has a **k value** controlling its strength:

```
Final Effect = Σ (Rule_i Base Effect × k_i)
```

- k = 1.0: Standard effect
- k > 1.0: Stronger effect
- k < 1.0: Weaker effect
- k = 0.0: Rule disabled

Randomizing k values within ranges produces training data diversity.

### 20 Rule Components

| Category | Count | Rules |
|----------|-------|-------|
| **Velocity** | 8 | HighLoud, PhraseArch, DurationContrast, MelodicCharge, CrescendoTempo, Tenuto, Accent, Marcato |
| **Timing** | 7 | PhraseRubato, BeatJitter, FinalRitard, Fermata, CrescendoTempo, Punctuation, Repetition |
| **Duration** | 8 | Staccato, Legato, Tenuto, Marcato, Fermata, DurationContrast, FinalRitard, Punctuation, Repetition, SocialDurationCare |
| **Special** | 6 | GraceNote, Trill, Mordent, Tremolo, AutoPedal, GlobalNormalizer |

---

## Detailed Rule Reference

### Velocity Rules

#### HighLoudRule
**Effect**: Higher pitches are played louder

- **Formula**: `dB = k × 0.03 × (pitch - 60)`, capped at ±1.0 dB
- **k_range**: (0.0, 2.0)
- **Example** (k=1.0):
  - C3 (MIDI 48): -0.36 dB
  - C5 (MIDI 72): +0.36 dB
  - C7 (MIDI 96): +1.0 dB (capped)

#### PhraseArchRule
**Effect**: Phrase arch — louder in middle, softer at boundaries

- **Formula**: Parabolic dome shape, `dB = k × 1.0 dB × arch_value`
- **k_range**: (0.0, 2.0)
- **Peak position**: 0.5-0.7 (randomized, asymmetric)
- **Example** (k=1.0): ±1.0 dB swing from start/end to peak

#### DurationContrastRule
**Effect**: Long notes louder, short notes softer

- **Formula**: `dB = k × 0.5 dB × log₂(relative_duration)`
- **k_range**: (0.0, 2.0)
- **Example** (k=1.0):
  - 2× duration → +0.5 dB
  - 0.5× duration → -0.5 dB

#### MelodicChargeRule
**Effect**: Emphasize non-chord tones (dissonances, tensions)

- **Formula**: `dB = k × melodic_charge` (0-2 dB based on NCT type)
- **k_range**: (0.0, 2.0)
- **NCT boost**: 2.0 dB
- **Appoggiatura timing**: +k × 20ms agogic accent

#### CrescendoTempoRule
**Effect**: Dynamics-tempo coupling (crescendo → faster, diminuendo → slower)

- **Velocity**: `dB = (loudness_incr - loudness_decr) × k × 3.0 dB`
- **Timing**: `-tempo_ratio × beat_duration` (position-only, not cumulative)
- **k_range**: (0.0, 2.0)
- **Max tempo change**: ±1.5%

#### TenutoRule
**Effect**: Hold full value, slightly louder and longer

- **Velocity**: `dB = k × 1.0`
- **Duration**: `multiplier = 1.0 + k × 0.05`
- **k_range**: (0.0, 2.0)

#### AccentRule
**Effect**: Accent (>) — emphasis through velocity

- **Velocity**: `dB = k × 1.5 dB × duration_scale`
- **Duration scale**: `min(1.0, duration / beat_duration)` — short notes get less accent
- **k_range**: (0.0, 1.5)

#### MarcatoRule
**Effect**: Strong accent (^) — strongest emphasis

- **Velocity**: `dB = k × 5.0`
- **Duration**: `multiplier = 1.0 - k × 0.05` (5% shortening)
- **k_range**: (0.0, 2.0)

---

### Timing Rules

#### PhraseRubatoRule
**Effect**: Tempo variation within phrases (agogic shaping)

- **Formula**: BPM-scaled timing offset, `-k × bpm_scale × tempo_ratio × beat_duration × 0.5`
- **BPM scaling**: `(100 / current_bpm)²`, capped at 1.5
- **Tempo ratio**: -0.1 (start) → +0.1 (middle) → -0.15 (end)
- **k_range**: (0.0, 2.0)
- **Example** @ 145 BPM: ±10% tempo → ±4.76% after BPM scaling

#### BeatJitterRule
**Effect**: Micro-timing fluctuations (motor noise)

- **Formula**: `offset = k × N(0, 5ms)` (Gaussian)
- **k_range**: (0.0, 2.0)
- **Special**: Chord notes (same onset) share the same jitter

#### FinalRitardRule
**Effect**: Gradual slowdown at the end (runner stopping model)

- **Formula**: `v(x) = √(1 - k·x)` where x ∈ [0, 1] in final section
- **Duration multiplier**: `1 / v(x)`
- **k_range**: (0.0, 2.0)
- **Ritard measures**: 2.0 (final 2 bars)
- **v_min**: 0.8 (k=1 → 1.25× slower at end)
- **Cumulative**: Yes (all subsequent notes affected)

#### FermataRule
**Effect**: Hold fermata note longer + pause after

- **Duration**: `multiplier = 1.0 + k × 1.0`
  - k=0.5 → 1.5×
  - k=1.0 → 2.0×
  - k=1.5 → 2.5×
- **Pause**: 0.5 beats after fermata note
- **k_range**: (0.5, 1.5)
- **Cumulative**: Yes (propagates onset shift to all subsequent notes)

#### PunctuationRule
**Effect**: Breathing between phrases

- **Timing**: `+k × 20ms × (beat_duration / 0.5)` (tempo-scaled micropause)
- **Duration**: Phrase-end notes shortened by `1.0 - k × 0.15`
- **k_range**: (0.5, 1.5)

#### RepetitionRule
**Effect**: Repeated notes shortened slightly (avoid machine-gun effect)

- **Duration**: `multiplier = 1.0 - k × 0.1`
- **k_range**: (0.0, 2.0)

---

### Duration Rules

#### StaccatoRule
**Effect**: Shorten staccato notes

- **Duration**: `multiplier = 1.0 - k × 0.25`
- **k_range**: (0.0, 2.0)

#### LegatoRule
**Effect**: Overlap legato notes smoothly

- **Duration**: `+k × 30ms`, capped at 10% of note duration
- **k_range**: (0.0, 2.0)

#### SocialDurationCareRule
**Effect**: Ensure minimum audible duration

- **Min duration**: 50ms
- **k_range**: (0.8, 1.2)

---

### Special Rules (Expansion/Global)

#### GraceNoteRule
**Effect**: Expand grace notes (acciaccatura/appoggiatura)

- **Acciaccatura** (slashed): `grace_duration = k × 50ms`
- **Appoggiatura** (unslashed): `grace_duration = k × 0.25 × beat_duration`
- **Timing**: Grace note can start up to 16th note before the beat (randomized)
- **Main note delay**: Main note MUST start after grace note finishes (physical constraint)
- **k_range**: (0.5, 1.5)

#### TrillRule
**Effect**: Expand trill into rapid alternation

- **Speed**: `note_duration = (beat_duration / 4) / k`
  - k=1.0 → 16th note speed (normal)
  - k=2.0 → 32nd note speed (fast)
- **k_range**: (1.0, 2.0)
- **Start on**: Lower note (Romantic style)

#### MordentRule
**Effect**: Expand mordent into 3-note ornament (main-aux-main)

- **Duration**: `mordent_duration = 0.08 / k`
  - k=1.0 → 80ms total (3 notes)
  - k=2.0 → 40ms (faster)
- **k_range**: (0.5, 2.0)

#### TremoloRule
**Effect**: Expand tremolo into rapid repetitions/alternations

- **Speed** (Hz-based, tempo-independent):
  - **Single-note** (same key): 1 slash=6Hz, 2=8Hz, 3=10Hz
  - **Two-note** (alternating keys): 1 slash=8Hz, 2=11Hz, 3=14Hz
- **Humanization**:
  - Ramp-up: First 2 notes slightly slower (+8%, +4%)
  - Velocity contour: Gentle arch (±1 dB)
  - Timing jitter: ±2ms per note
- **k_range**: (0.0, 2.0)
- **Min hits**: 4 (regardless of duration)
- **Chord tremolo**: All chord notes repeat together on each hit

#### AutoPedalRule
**Effect**: Automatic sustain pedal (CC64) based on harmony changes

- **Lift timing**: `change_time - k × 30ms` (before harmony change)
- **Press timing**: `change_time + k × 20ms` (after change)
- **Depth** (CC64 value): Based on local velocity + tempo
  - `depth = 50 + k × vel_norm × tempo_scale × 60`
  - Velocity normalization: `(avg_velocity - 40) / 70`
  - Tempo scaling: `min(1.0, beat_dur / 0.75)` — faster = shallower
- **Harmony detection**:
  - Analysis window: `max(beat_duration, 0.6s)` (time-based, meter-independent)
  - Detects pitch set changes
  - Filters no-bass measures (likely melody-only)
- **Score pedal shift**: Score pedal markings shifted to humanized time domain via linear interpolation
- **k_range**: (0.5, 1.5)

#### GlobalNormalizer
**Effect**: Global velocity normalization + soft clipping

- **Target RMS**: 70
- **Soft clip threshold**: 100
- **Max velocity**: 115
- **Method**: Tanh soft clipping to avoid "smashing piano"

---

## Partitura Feature Integration

The engine uses Partitura's `make_note_feats()` to extract basis features from scores.

### Feature Functions Used

```python
feature_functions = [
    # Core features
    'polynomial_pitch_feature',       # Pitch analysis
    'loudness_direction_feature',     # Dynamics (crescendo/diminuendo)
    'tempo_direction_feature',        # Tempo markings
    'articulation_direction_feature', # Articulation (staccato, tenuto, accent, marcato)
    'duration_feature',               # Note duration
    'slur_feature',                   # Phrase boundaries (slur_incr/slur_decr)
    'fermata_feature',                # Fermata marks
    'grace_feature',                  # Grace notes

    # Enhanced features
    'metrical_strength_feature',      # Metrical strength (better than metrical_feature)
    'vertical_neighbor_feature',      # Harmonic context (for NCT detection)
    'onset_feature',                  # Score position (for ritardando)
]
```

### Feature Counts

**Total**: 58 features per note

| Feature Function | Descriptors | Main Outputs |
|------------------|-------------|--------------|
| `polynomial_pitch_feature` | 1 | pitch |
| `loudness_direction_feature` | 16 | ppp-fff, sf/sfz, loudness_incr/decr |
| `tempo_direction_feature` | 17 | adagio-prestissimo, tempo_incr/decr |
| `articulation_direction_feature` | 5 | staccato, tenuto, accent, marcato, unknown |
| `duration_feature` | 1 | duration |
| `slur_feature` | 2 | slur_incr, slur_decr |
| `fermata_feature` | 1 | fermata |
| `grace_feature` | 3 | grace_note, n_grace, grace_pos |
| `metrical_strength_feature` | 4 | beat_phase, downbeat, secondary, weak |
| `vertical_neighbor_feature` | 6 | n_total, n_above/below, pitch_range |
| `onset_feature` | 2 | onset, score_position |

### Fallback Handling

If `make_note_feats()` fails, the engine proceeds with empty basis features (all values = 0). Core functionality (onset, duration, articulation, ornaments) is computed independently and not affected.

### Known Partitura Issues

**Buggy features** (KeyError: 'None'):
- `articulation_feature` → Use `articulation_direction_feature` instead
- `metrical_feature` → Use `metrical_strength_feature` instead
- `ornament_feature` → Parse ornaments directly from Part objects

---

## File Structure

```
src/audio/humanize/
├── __init__.py              # Public API
├── config.py                # RuleConfig + HumanizationConfig
├── convert.py               # dB ↔ velocity conversion
├── engine.py                # HumanizationEngine main engine
├── metadata.py              # Reproducibility tracking
└── rules/
    ├── __init__.py
    ├── base.py              # Rule base class
    ├── tempo.py             # Tempo interpretation
    ├── high_loud.py         # HighLoudRule
    ├── phrase_arch.py       # PhraseArchRule
    ├── duration_contrast.py # DurationContrastRule
    ├── melodic_charge.py    # MelodicChargeRule
    ├── rubato.py            # PhraseRubatoRule
    ├── jitter.py            # BeatJitterRule
    ├── final_ritard.py      # FinalRitardRule
    ├── fermata.py           # FermataRule
    ├── dynamics_tempo.py    # CrescendoTempoRule
    ├── punctuation.py       # PunctuationRule
    ├── repetition.py        # RepetitionRule
    ├── articulation.py      # Staccato, Legato, Tenuto, Accent, Marcato
    ├── ornaments.py         # GraceNote, Trill, Mordent
    ├── tremolo.py           # TremoloRule
    ├── safety.py            # SocialDurationCare, GlobalNormalizer
    └── pedal.py             # AutoPedalRule
```

## Dependencies

- `partitura`: Score parsing and basis functions
- `mido`: MIDI file I/O
- `numpy`: Numerical computation
- `click`: CLI interface (if using CLI)

## Reproducibility

Every generated MIDI has a `.json` metadata file recording:
- All k values
- Random seed
- Source file
- Timestamp
- Partitura version

Can fully reproduce the same output from metadata.

## vs. clef-piano-base

| Aspect | clef-piano-base | clef-piano-full (this) |
|--------|-----------------|------------------------|
| Velocity | Uniform (90) | KTH rule system |
| Timing | Mechanical | Rubato + ritardando |
| Articulation | None | Staccato/legato/ornaments |
| Pedal | None | Auto pedaling |
| Diversity | SoundFont only | k-values × SoundFont |
| Purpose | Baseline comparison | Primary training data |

## References

- **KTH Director Musices**: `docs/kth_director_musices_rules.pdf`
- **BasisMixer**: Partitura-based performance rendering
- **Partitura Documentation**: https://partitura.readthedocs.io/

## License

MIT License

## Authors

clef research team
