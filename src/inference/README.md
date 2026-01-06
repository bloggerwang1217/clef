# MT3 Batch Inference (Docker)

How to run:
1) Single-file test
2) ASAP test dataset batch

## Prerequisites
- Docker with NVIDIA GPU (`--gpus all`)
- Reference image build: https://github.com/bloggerwang1217/mt3-docker (original by https://github.com/jsphweid/mt3-docker)
- Volumes (examples; adjust to your paths):
  - `/ABS/PATH/data/asap_test_set` → `/data/input`
  - `/ABS/PATH/output` → `/data/output`

Build and run the service container:
```bash
cd /path/to/mt3-docker
sudo docker build -t mt3 .
sudo docker run -p 5000:5000 --gpus all \
  -v /ABS/PATH/data/asap_test_set:/data/input \
  -v /ABS/PATH/output:/data/output \
  mt3
```

## Single file test
Assume the audio file is at `/data/input/test/input_audio.mp3`.
```bash
cd /path/to/clef
python -m inference.batch_transcribe \
  --mode single_file \
  --input-dir /data/input \
  --input-file /data/input/test/input_audio.mp3 \
  --output-dir /data/output/test_output \
  --api-url http://localhost:5000/batch-transcribe \
  --model piano
```
Result: MIDI written to `/data/output/test_output/input_audio.mid`.

## ASAP test dataset (batch)
Assume ASAP dataset and metadata are under `/data/input/asap_test_set/` with `metadata.csv`.
```bash
cd /path/to/clef
python -m inference.batch_transcribe \
  --mode asap_batch \
  --input-dir /data/input/asap_test_set \
  --metadata-csv /data/input/asap_test_set/metadata.csv \
  --output-dir /data/output/asap_midi_output \
  --api-url http://localhost:5000/batch-transcribe \
  --model piano
```
Result: MIDI written to `/data/output/asap_midi_output/` preserving composer/subfolder structure.

## Notes
- `--model piano` uses the ismir2021 (piano) checkpoint; use `--model mt3` for multitrack.
- The container reads audio paths and writes MIDI paths as given; ensure they match mounted volumes.
- Idempotent: reruns will overwrite the same MIDI paths; partial files are replaced atomically.
