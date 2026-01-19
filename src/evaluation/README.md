# Evaluation Module

This directory contains evaluation tools for audio-to-score transcription models.

## Shared Modules

| File | Purpose |
|------|---------|
| `mv2h.py` | Standalone MV2H evaluation module (usable by MT3, Clef, Transkun+Beyer) |
| `asap.py` | ASAP dataset handling (ground truth finding, chunk loading) |

## Zeng et al. (2024) Scripts

Scripts adapted from [piano-a2s](https://github.com/wei-zeng98/piano-a2s) for fair comparison:

| File | Original Source | Purpose |
|------|----------------|---------|
| `evaluate.py` | [evaluate.py](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate.py) | Zeng model evaluation (MV2H/WER/F1/ER) |
| `evaluate_midi_mv2h.sh` | [evaluate_midi_mv2h.sh](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate_midi_mv2h.sh) | MV2H shell wrapper |
| `humdrum.py` | [humdrum.py](https://github.com/wei-zeng98/piano-a2s/blob/main/data_processing/humdrum.py) | **Kern ↔ symbolic conversion (used by Clef) |

---

## Setup

### 1. MV2H Java Tool

MV2H (Multi-pitch, Voice, Meter, Value, Harmony) evaluator by McLeod & Steedman (2018).

```bash
cd /path/to/clef
git clone https://github.com/apmcleod/MV2H.git
cd MV2H && make && cd ..
```

After compilation, the binaries will be at `MV2H/bin/`.

### 2. Python Dependencies

```bash
poetry install
```

See `pyproject.toml` for full dependency list.

---

## Usage

### Shared MV2H Module

```python
from evaluation.mv2h import MV2HEvaluator, MV2HResult, aggregate_mv2h_results

# Initialize evaluator
evaluator = MV2HEvaluator(mv2h_bin="MV2H/bin", timeout=120)

# Evaluate single pair
result = evaluator.evaluate(gt_midi_path, pred_midi_path)

print(result.mv2h)        # Official 5-metric average
print(result.mv2h_custom) # 4-metric average (excludes Meter)
print(result.to_dict())   # All metrics as dict
```

### Shared ASAP Module

```python
from evaluation.asap import ASAPDataset, ChunkInfo

# Initialize
asap = ASAPDataset("/path/to/asap-dataset")

# Find ground truth for a prediction file
gt_path = asap.find_ground_truth_midi("Bach_Prelude_bwv_875")

# Load Zeng's 5-bar chunks
chunks = asap.load_chunks("zeng_test_chunk_set.csv")
grouped = asap.group_chunks_by_piece(chunks)
```

---

## MV2H Metrics

| Metric | Description |
|--------|-------------|
| Multi-pitch | Pitch accuracy |
| Voice | Voice separation accuracy |
| Meter | Metrical structure accuracy |
| Value | Note value accuracy |
| Harmony | Harmonic structure accuracy |
| **MV2H** | Official average of all 5 metrics |
| **MV2H_custom** | Average of 4 (excludes Meter, per Zeng et al.) |

**Formula:**
```
MV2H = (Multi-pitch + Voice + Meter + Value + Harmony) / 5
MV2H_custom = (Multi-pitch + Voice + Value + Harmony) / 4
```

---

## License

Zeng's scripts are under Apache-2.0 License:

```
Copyright 2024 Wei Zeng, Xian He, Ye Wang
Licensed under the Apache License, Version 2.0
```

## Citation

```bibtex
@misc{zeng2024endtoendrealworldpolyphonicpiano,
  title={End-to-End Real-World Polyphonic Piano Audio-to-Score Transcription with Hierarchical Decoding},
  author={Wei Zeng and Xian He and Ye Wang},
  year={2024},
  eprint={2405.13527},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2405.13527}
}
```

## Acknowledgments

We thank Wei Zeng, Xian He, and Ye Wang for open-sourcing their evaluation pipeline.
