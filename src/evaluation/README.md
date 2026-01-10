# Zeng et al. (2024) Evaluation Scripts

This directory contains evaluation scripts adapted from the [piano-a2s](https://github.com/wei-zeng98/piano-a2s) project to ensure fair comparison with Zeng et al. (2024).

## Source Files

| File | Original Source | Purpose |
|------|----------------|---------|
| `evaluate.py` | [evaluate.py](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate.py) | Main evaluation script for MV2H/WER/F1/ER metrics |
| `evaluate_midi_mv2h.sh` | [evaluate_midi_mv2h.sh](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate_midi_mv2h.sh) | Shell script for MV2H computation |
| `humdrum.py` | [humdrum.py](https://github.com/wei-zeng98/piano-a2s/blob/main/data_processing/humdrum.py) | **Kern ↔ symbolic conversion utilities |
| `LICENSE` | [LICENSE](https://github.com/wei-zeng98/piano-a2s/blob/main/LICENSE) | Apache-2.0 License |

## License

```
Copyright 2024 Wei Zeng, Xian He, Ye Wang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Prerequisites

### 1. Humdrum Toolkit

Already installed at `~/humdrum-tools/`. Verify:

```bash
which tiefix   # Should show: /Users/bloggerwang/humdrum-tools/humextra/bin/tiefix
which hum2xml  # Should show: /Users/bloggerwang/humdrum-tools/humextra/bin/hum2xml
```

### 2. MV2H Evaluator

```bash
cd evaluation
git clone https://github.com/cheriell/music-voice-separation.git mv2h_tool
```

### 3. Python Dependencies

```bash
pip install music21 numpy
```

## Evaluation Pipeline

Zeng's evaluation workflow:

```
**Kern output
    ↓
get_xml_from_target() conversion:
    ├── tiefix (fix tie/slur notation)
    ├── hum2xml (convert to XML)
    └── music21 (add clefs, key/time signatures)
    ↓
MusicXML
    ↓
Convert to MIDI → MV2H evaluation
```

## Usage Example

```python
import sys
sys.path.append('evaluation/zeng_baseline')
from humdrum import get_xml_from_target

# Convert **Kern to MusicXML
xml_path = get_xml_from_target(
    labels_upper=upper_voice_tokens,
    labels_lower=lower_voice_tokens,
    time_sig='4/4',
    key='C',
    output_path='output.xml'
)
```

## Modifications

Any modifications made to these scripts for our experiments:

- ✅ **No modifications**: Using original scripts as-is for evaluation
- 📝 If modified, changes will be documented here

## Citation

If you use these scripts, please cite the original paper:

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

We thank Wei Zeng, Xian He, and Ye Wang for open-sourcing their evaluation pipeline, which enables fair and reproducible comparison with their work.
