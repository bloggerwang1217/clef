# Clef: Audio-to-Score Transcription via Vision-Language Models

> **Hearing as Seeing**: Polyphonic Music Transcription as a Vision-Language Task

## Project Overview

Clef aims to be the "Whisper of music transcription" — a state-of-the-art system that directly converts audio recordings into human-readable sheet music.

```bash
clef -i chopin.mp3 -o chopin.musicxml
```

The core innovation is treating audio spectrograms as "images" and leveraging Vision Transformer (ViT) architectures pre-trained on ImageNet.

### Research Hypothesis

Based on neuroscience evidence (Sur's ferret rewiring experiments, STRFs in auditory cortex), we hypothesize that:
- The brain uses a **universal geometric algorithm** for both visual and auditory processing
- A frozen ViT encoder can capture **long-range harmonic relationships** in spectrograms
- **Timbre Domain Randomization (TDR)** forces the model to learn pitch geometry, not timbre texture

### Three Innovation Pillars

1. **Architectural Innovation**: ViT encoder (frozen ImageNet weights) + Autoregressive Transformer decoder
2. **Data Innovation**: Timbre Domain Randomization (TDR) - extreme timbre variation during training
3. **Task Innovation**: Implicit Quantization - model learns music grammar, not physical time

## Project Structure

```
clef/
├── src/                      # Reusable utility modules
│   ├── audio/                # Audio processing
│   │   ├── converter.py      # Format conversion (MP3→WAV)
│   │   └── separator.py      # Demucs vocal separation
│   ├── score/                # Score handling
│   │   ├── parser.py         # MusicXML parsing
│   │   └── generator.py      # MusicXML generation
│   └── utils/                # Common utilities
│       ├── device.py         # GPU/MPS/CPU auto-detection
│       └── temp.py           # Temp file management
├── docs/                     # Research planning documents
│   ├── clef-plan.md          # Main research proposal (READ THIS FIRST)
│   ├── decision-record.md    # Why **Kern over ABC notation
│   ├── experiment-design.md  # Study 1 & Study 2 design
│   ├── data-pipeline-implementation.md  # TDR data generation pipeline
│   ├── model-comparison.md   # Whisper vs VLM vs Clef
│   ├── A1-V1-comparison.md   # Auditory vs Visual cortex evidence
│   └── legacy/               # Archived drafts
├── paper/                    # Reference papers (PDFs)
├── tests/                    # Unit tests
│   ├── test_audio.py
│   └── test_score.py
└── README.md
```

## Architecture

```
Input: Audio Waveform
    ↓
Log-Mel Spectrogram (2D "image")
    ↓
┌─────────────────────────────────────┐
│  Frozen ViT Encoder (ImageNet)      │  ← Cross-modal transfer
│  - 16x16 patches                    │
│  - Global self-attention            │
└─────────────────────────────────────┘
    ↓
Visual Feature Sequence
    ↓
┌─────────────────────────────────────┐
│  Autoregressive Transformer Decoder │  ← Implicit Quantization
│  - Predicts **Kern tokens           │
│  - Uses <coc> for multi-track       │
└─────────────────────────────────────┘
    ↓
**Kern Notation → MusicXML → Sheet Music
```

## Data Strategy

### Training Data (Synthetic-to-Real)

| Dataset | Purpose | Size | Role |
|---------|---------|------|------|
| **PDMX** | Pre-training | 254K+ | Massive score variety |
| **KernScores** | Pre-training | 108K+ | Musicologically accurate |
| **ASAP** | Real-world test (Piano) | ~50h | Primary benchmark |
| **URMP** | Real-world test (Multi-instrument) | 44 pieces | Generalization test |

### Augmentation Pyramid (TDR)

```
Level 3: Timbre Domain Randomization (CRITICAL)
├── Piano → Violin → 8-bit Chiptune → Sawtooth Synth
└── Forces pitch geometry learning, not timbre memorization

Level 2: Source-Level
├── Multi-SoundFont (Steinway, Yamaha, Upright)
├── Detuning (±10 cents)
└── Timing jitter (humanize)

Level 1: Signal-Level
├── Impulse Response convolution (reverb)
├── Additive noise (white, pink, ambient)
└── Frequency cutoff (lo-fi simulation)
```

## Evaluation

### Metrics

- **MV2H**: Music evaluation (pitch F_p, harmony F_harm, rhythm F_val)
- **TEDn**: Tree Edit Distance on MusicXML (structural correctness)
- NOT using WER (lacks musical semantics understanding)

### Target Benchmarks

| Study | Dataset | Target | Baseline to Beat |
|-------|---------|--------|------------------|
| Study 1 | ASAP | MV2H > 78% | Zeng et al. (2024): 74.2% |
| Study 2 | URMP | Zero-shot MV2H > 60% | MT3: ~40% |

## Output Format

Using **\*\*Kern notation** (not ABC or MusicXML):
- Native multi-track support (Tab-separated Spines)
- Explicit time alignment (same row = simultaneous)
- Serializable with `<coc>` (Change of Column) token
- Musicological standard (Humdrum Toolkit)

Example:
```
4c    <coc>    2C    <coc>    4e    <coc>    .
(C4 quarter, C3 half, E4 quarter, rest/sustain)
```

## Development Guidelines

### Code Style
- Comments in English
- Documentation in Traditional Chinese (Taiwan style)
- Prefer maintainable, automated solutions

### Tech Stack
- **Audio Rendering**: FluidSynth + diverse SoundFonts
- **Synth Timbres**: pyo / csound (sawtooth, FM, 8-bit)
- **Signal Processing**: Spotify Pedalboard / TorchAudio
- **Spectrogram**: nnAudio (GPU-accelerated)
- **Model**: PyTorch, Hugging Face Transformers

### System Dependencies
```bash
# Required for audio synthesis (MIDI → Audio)
sudo apt-get install fluidsynth  # Ubuntu/Debian (v2.2.5)
```

### Key Dependencies (managed by Poetry)
```
# Core ML
torch ^2.5          # GPU support (CUDA/MPS)
transformers ^4.51  # ViT, Transformer decoder
nnAudio ^0.3        # GPU-accelerated spectrogram

# Audio Processing
librosa ^0.10       # Audio analysis
soundfile ^0.12     # Audio I/O
pedalboard ^0.9     # Spotify audio effects

# Music Score
music21 ^9.1        # MusicXML handling

# Optional Groups
[separation] demucs ^4.0      # Source separation
[synthesis]  pyfluidsynth     # MIDI → Audio rendering (requires fluidsynth CLI)
```

## Current Status

### Completed
- [x] Research proposal and literature review
- [x] Architecture decision (ViT + Transformer)
- [x] Output format decision (**Kern)
- [x] Data augmentation strategy (TDR)
- [x] Evaluation metrics selection (MV2H + TEDn)
- [x] Reusable utility modules (src/audio, src/score, src/utils)

### In Progress / TODO
- [ ] Implement ViT encoder with frozen ImageNet weights
- [ ] Implement **Kern tokenizer with <coc> support
- [ ] Build autoregressive Transformer decoder
- [ ] Create TDR data generation pipeline
- [ ] Preprocess PDMX/KernScores datasets
- [ ] Implement MV2H and TEDn evaluation
- [ ] Run Study 1 experiments (ASAP)
- [ ] Run Study 2 experiments (URMP)

## Key Papers

| Paper | Relevance |
|-------|-----------|
| Zeng et al. (2024) | CNN-based SOTA, MV2H baseline |
| Zhang & Sun (2024) | Whisper-based approach (ABC notation for melody + chords) |
| Gong et al. (2021) AST | ViT for audio classification |
| Alfaro-Contreras (2024) | **Kern + <coc> serialization |
| Mayer et al. (2024) | TEDn metric justification |

**Footnote on Zhang & Sun (2024):** This entry hypothesizes that Whisper's optimization for speech recognition may inherit acoustic biases from speech data that could limit its generalization to music with different timbral and pitch distributions. This is an assumption about architectural inductive biases rather than an explicit claim in the paper.

## Usage Examples

```python
# Audio processing
from src.audio import convert_to_wav, VocalSeparator
from src.utils import get_device

wav_path = convert_to_wav("input.mp3", "output.wav")
separator = VocalSeparator(device=get_device())
vocals = separator.extract_vocals(audio, sample_rate=44100)

# Score handling
from src.score import parse_musicxml, extract_notes_from_score

score = parse_musicxml("input.musicxml")
notes = extract_notes_from_score(score)
```

```bash
# Future: Run Clef model (TBD)
# python -m clef.transcribe input.wav --output output.kern
```

