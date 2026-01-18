# MT3 Baseline

MT3 (Music Transcription with Transformers) baseline for comparison with Clef model.

## Files

- `mt3_inference.py` - Batch transcription using MT3 Docker API
- `mt3_evaluate.py` - MV2H evaluation with MuseScore 4.6.5 conversion

## Prerequisites

- Docker with NVIDIA GPU support
- MT3 Docker image: https://github.com/bloggerwang1217/mt3-docker
- MuseScore 4.6.5 (for evaluation)

## Quick Start

Use the shell script for common operations:

```bash
# Start MT3 Docker container
./scripts/mt3_inference.sh start

# Run inference on ASAP test set
./scripts/mt3_inference.sh asap

# Stop container
./scripts/mt3_inference.sh stop
```

## Manual Usage

### Inference

```bash
python -m src.baselines.mt3.mt3_inference \
  --mode asap_batch \
  --input-dir /data/datasets/asap_test_set \
  --metadata-csv /data/datasets/asap_test_set/metadata.csv \
  --output-dir /data/experiments/mt3/full_midi \
  --api-url http://localhost:5000/batch-transcribe \
  --model piano
```

### Evaluation

```bash
python -m src.baselines.mt3.mt3_evaluate \
  --mode full \
  --pred_dir data/experiments/mt3/full_midi \
  --gt_dir data/datasets/asap-dataset \
  --mv2h_bin MV2H/bin \
  --output results/mt3_mv2h.csv \
  --workers 8
```

## Notes

- `--model piano` uses ismir2021 piano checkpoint
- `--model mt3` uses multi-instrument checkpoint
- Evaluation uses MuseScore 4.6.5 for MIDI to MusicXML conversion
- MV2H_custom = (Multi-pitch + Voice + Value + Harmony) / 4 (excludes Meter)

---

## Technical Design

### Objective

Implement MT3 baseline MV2H evaluation in the Clef repo, supporting:
1. **Full Song Evaluation** — Evaluate entire pieces
2. **5-bar Chunk Evaluation** — Same granularity as Zeng et al. for apple-to-apple comparison

### File Structure

```
clef/
├── scripts/
│   ├── mt3_inference.sh          # Docker inference script
│   └── setup_musescore.sh        # MuseScore 4.6.5 setup
├── src/
│   ├── baselines/mt3/
│   │   ├── mt3_inference.py      # Batch transcription
│   │   ├── mt3_evaluate.py       # MV2H evaluation
│   │   └── README.md             # This file
│   └── evaluation/
│       ├── mv2h.py               # Shared MV2H module
│       ├── asap.py               # Shared ASAP dataset module
│       ├── evaluate_midi_mv2h.sh # MV2H shell wrapper (Zeng's)
│       └── humdrum.py            # Kern format processing (Zeng's)
├── tools/
│   ├── mscore.AppImage           # MuseScore 4.6.5 (gitignored)
│   └── mscore                    # Headless wrapper script
├── data/
│   ├── datasets/
│   │   └── asap_test_set/        # ASAP test set (audio + metadata)
│   └── experiments/mt3/
│       └── full_midi/            # MT3 inference output
└── MV2H/                         # MV2H Java tool (clone separately)
```

### Evaluation Pipeline

```
MT3 Audio-to-MIDI          MuseScore 4.6.5              MV2H
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Audio (MP3)   │       │  MIDI → MusicXML │       │   MV2H Java     │
│        ↓        │       │  - Voice sep.    │       │   with DTW      │
│   MT3 Docker    │  →    │  - Quantization  │  →    │   alignment     │
│        ↓        │       │  - Smart import  │       │        ↓        │
│   Raw MIDI      │       │        ↓         │       │   6 Metrics     │
└─────────────────┘       │  Quantized MIDI  │       └─────────────────┘
                          └─────────────────┘
```

### MV2H Metrics

| Metric | Description |
|--------|-------------|
| Multi-pitch | Pitch accuracy |
| Voice | Voice separation accuracy |
| Meter | Metrical structure accuracy |
| Value | Note value accuracy |
| Harmony | Harmonic structure accuracy |
| MV2H | Official average of all 5 metrics |
| **MV2H_custom** | Average of 4 metrics excluding Meter (Zeng et al.) |

### Why MuseScore 4.6.5?

We use MuseScore 4.6.5 (released Dec 18, 2025) as the "Industry Standard Baseline" for MIDI to MusicXML conversion instead of naive quantization libraries (e.g., music21).

**Advantages:**
- Sophisticated heuristic-based import engine
- Voice separation
- Tuplet detection
- Smart quantization
- Crash fixes critical for parallel processing (64+ workers)

**Reference:** https://github.com/musescore/MuseScore/releases/tag/v4.6.5

### Expected Results

- **Zeng et al. MV2H:** 74.2% (CNN + GRU, trained on real audio)
- **MT3 + MuseScore MV2H:** Expected < 60% (Straw Man baseline)

The gap demonstrates the limitation of audio-to-MIDI approaches that lack proper music notation understanding — which is exactly what the Clef model aims to solve.

### References

- MT3: https://github.com/magenta/mt3
- MV2H: McLeod & Steedman (2018) "Evaluating Automatic Polyphonic Music Transcription"
- Zeng et al. (IJCAI 2024): CNN + GRU approach with MV2H = 74.2%
- MuseScore: https://musescore.org/
