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
