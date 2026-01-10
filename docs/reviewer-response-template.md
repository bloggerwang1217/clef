# Reviewer Response Template: Evaluation Protocol Defense

本文件提供針對評估流程設計可能遭受質疑的防守範本。

---

## Question 1: Why Not Unify at **Kern Format?

### Reviewer's Concern

> "Your evaluation protocol is inconsistent. Pipeline systems (MT3, Transkun) output MIDI and directly convert to MusicXML, while End-to-End systems (Zeng, Clef) output **Kern and then convert to XML. Why not enforce all systems to output **Kern first to ensure a unified conversion path?"

### Our Response

We appreciate the reviewer's attention to evaluation fairness. We chose **not** to enforce a unified **Kern intermediate format for the following reasons:

**1. **Kern is Not a Universal Standard**

- **Kern is a domain-specific format primarily used in musicology research and Humdrum Toolkit ecosystem
- It is the **native output format** of Zeng's and our models, but not of Pipeline systems
- Forcing MIDI-based systems to convert to **Kern would require implementing a non-standard converter that does not exist in established toolkits

**2. Avoiding Additional Conversion Errors**

- There is **no standard MIDI → **Kern converter** in widely-used libraries (music21, pretty_midi, mido)
- Implementing such a converter would introduce:
  - Additional quantization decisions (which notes belong to which voice?)
  - Arbitrary formatting choices (how to represent ornaments, articulations?)
  - **New sources of error** that would unfairly penalize Pipeline systems

**3. MusicXML as the Universal Symbolic Representation**

- **MusicXML is the de facto standard** for symbolic music notation interchange
- All systems (Pipeline and E2E) can produce MusicXML either:
  - **Natively** (MT3 → music21 → XML, Transkun → Beyer → XML)
  - **Via standard tools** (Zeng/Clef: **Kern → Humdrum Toolkit → XML)
- This respects each system's design philosophy while ensuring fair comparison

**4. The Beyer Baseline Cannot Be Evaluated Under **Kern**

- The Beyer model (ISMIR 2024, arXiv:2410.00210) is a Transformer-based **MIDI-to-Score** system
- It directly outputs **MusicXML tokens**, not **Kern
- Forcing Beyer → **Kern conversion would require:
  ```
  Beyer Model → XML → [???] → **Kern → [tiefix+hum2xml] → XML
  ```
- This introduces a **round-trip conversion** (XML → **Kern → XML) that would:
  - Introduce information loss
  - Create unfair disadvantage for the SOTA Pipeline baseline
  - Invalidate the "Strong Baseline" comparison

**5. Our Evaluation Focus**

Our research question is:
> "Can End-to-End models produce better **symbolic scores** than Pipeline systems?"

We evaluate **symbolic correctness at the score level**, not **intermediate format consistency**. As long as all systems converge to the same symbolic representation (MusicXML) before metric computation, the comparison is fair.

### Supporting Evidence

- **Precedent in Literature**: Other A2S papers (Liu et al. 2021, Román et al. 2019) evaluate at the symbolic level without enforcing intermediate format uniformity
- **Tool Availability**: music21 provides `write('musicxml')` for all score formats, ensuring a unified XML → MIDI conversion for MV2H

---

## Question 2: Different Conversion Paths are Unfair

### Reviewer's Concern

> "Pipeline systems use music21/Beyer for quantization, while E2E systems use tiefix+hum2xml. These different tools may introduce different biases, making the comparison unfair."

### Our Response

**This difference reflects the fundamental architectural distinction we aim to evaluate.**

**1. The Difference is Intentional**

- **Pipeline systems**: Quantization is performed by **external post-processing** (music21 or Beyer)
  - These are **domain-general** tools that apply heuristics or learned patterns
  - They operate without knowledge of the original audio

- **E2E systems**: Quantization is **implicitly learned** during training
  - The model learns to map audio features directly to discrete symbolic events
  - Quantization decisions are informed by acoustic context (timbre, dynamics, articulation)

**2. We Ensure Fairness at the Final Evaluation Stage**

All systems undergo **identical XML → MIDI conversion** using music21:

```python
# Applied to ALL systems uniformly
score = music21.converter.parse(xml_path)
score.write('midi', fp=midi_path)
```

This ensures that MV2H metrics are computed on **identically processed** symbolic MIDI, eliminating tool-specific biases at the evaluation stage.

**3. Alternative Design Would Introduce Greater Unfairness**

If we forced all systems through the **same quantization tool** (e.g., music21):

```
MT3 → MIDI → [music21] → XML  ✓ Native path
Zeng → **Kern → [music21] → XML  ✗ Forced to use wrong tool
```

This would:
- Nullify Zeng's learned quantization advantage
- Create an unfair **"level playing field"** that actually disadvantages E2E systems
- Violate the principle of evaluating systems **as designed**

**4. The Conversion Tools Are Publicly Available**

Both conversion paths use **open-source, reproducible tools**:

| System Type | Conversion Tools | Reproducibility |
|-------------|-----------------|-----------------|
| Pipeline | music21 (MIT), Beyer model (open-source) | ✓ Full |
| E2E | Humdrum Toolkit (BSD), music21 (MIT) | ✓ Full |

Any researcher can reproduce our results using the same toolchain.

---

## Question 3: Why Include Transkun + Beyer?

### Reviewer's Concern

> "Transkun + Beyer is a novel combination not previously published. This seems unfair—you're creating a new baseline just to make your method look better. Why not compare only against MT3 or Zeng?"

### Our Response

**Including Transkun + Beyer is essential to demonstrate that our improvements are not due to comparing against weak baselines.**

**1. Defensive Attack: Testing Against the Strongest Possible Pipeline**

Academic rigor requires testing against the **strongest conceivable opponent**:

- **MT3 + music21**: Industry standard (many users rely on this)
  - But a reviewer might say: "MT3 is old (2022), of course you beat it"

- **Transkun + Beyer**: SOTA components (as of 2024)
  - Audio-to-MIDI: Transkun (ISMIR 2023) beats MT3 on piano tasks
  - MIDI-to-Score: Beyer (ISMIR 2024) beats rule-based methods
  - **If we beat this, we prove E2E > Pipeline SOTA**

**2. Establishing the Pipeline Ceiling**

This combination represents the **theoretical maximum** of Pipeline approaches:

| Component | Selection Rationale |
|-----------|-------------------|
| Transkun | Best published Audio-to-MIDI for piano (F1-score 92%+) |
| Beyer | Best published MIDI-to-Score (beats HMM baselines) |

If our E2E method outperforms this combination, it demonstrates:
- **Error propagation** is inevitable in Pipeline systems
- Even perfect MIDI transcription cannot fully capture score-level semantics
- Direct audio-to-score modeling is necessary

**3. This is Standard Practice in ML**

Many papers combine SOTA components to create strong baselines:

- Computer Vision: Combining best backbone + best head
- NLP: Combining best encoder + best decoder
- Music: Combining best separation + best transcription

**4. Full Transparency**

We clearly state:
- This is a **constructed baseline**, not a published system
- We provide the exact models, versions, and hyperparameters used
- Other researchers can reproduce this baseline exactly

If we beat Transkun+Beyer, reviewers cannot claim we cherry-picked weak baselines.

---

## Question 4: Code and Reproducibility

### Reviewer's Concern

> "Will you release the evaluation code? How can we verify your MV2H scores?"

### Our Response

**Full reproducibility is guaranteed.**

**1. Evaluation Code Release**

We will release:
```
clef/
├── evaluation/
│   ├── zeng_baseline/          # Zeng's scripts (Apache-2.0)
│   │   ├── evaluate.py
│   │   ├── evaluate_midi_mv2h.sh
│   │   └── humdrum.py
│   ├── run_mt3_baseline.py     # Our MT3 pipeline
│   ├── run_transkun_beyer.py   # Our SOTA pipeline
│   └── run_clef_evaluation.py  # Our E2E evaluation
└── docs/
    └── evaluation_protocol.md  # Step-by-step instructions
```

**2. Exact Tool Versions**

| Tool | Version | Installation |
|------|---------|-------------|
| Humdrum Toolkit | 2024.01 | `git clone https://github.com/humdrum-tools/humdrum-tools` |
| music21 | 9.1.0 | `pip install music21==9.1.0` |
| MV2H Evaluator | 2019 | `git clone https://github.com/cheriell/music-voice-separation` |
| MT3 | 0.1.0 | `pip install mt3` |
| Transkun | - | Provided checkpoint + inference script |
| Beyer | - | Provided checkpoint + inference script |

**3. Ground Truth Files**

We use **Zeng's exact test split**:
- File: `data_processing/metadata/test_asap.txt` from piano-a2s repo
- 25 pieces / 80 recordings
- Publicly available from ASAP dataset

**4. Deterministic Evaluation**

All non-deterministic steps are controlled:
```python
# Fixed random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic quantization
score.quantize(quarterLengthDivisors=[4, 3], inPlace=True)

# Identical XML → MIDI conversion
score.write('midi', fp=output_midi)
```

---

## Question 5: Cherry-Picking Metrics

### Reviewer's Concern

> "You report MV2H, but what about other metrics like MUSTER or onset F1-score? Are you cherry-picking metrics that favor your method?"

### Our Response

**We follow Zeng et al.'s evaluation protocol exactly to ensure fair comparison.**

**1. Metric Selection Rationale**

| Metric | Used? | Reason |
|--------|-------|--------|
| **MV2H** | ✓ Yes | Standard for **symbolic score evaluation** (pitch, voice, harmony, value) |
| **WER** | ✓ Yes | Sequence-level error (used by Zeng) |
| **F1 (key/time)** | ✓ Yes | Bar-level accuracy (used by Zeng) |
| **STEPn** | ✓ Yes | **Our contribution**: Notation structure evaluation |
| Onset F1 | ✗ No | **Performance-level** metric, not score-level |
| MUSTER | ✗ No | XML edit distance, highly correlated with MV2H |

**2. We Report ALL Metrics from Zeng's Paper**

We do not selectively report metrics. Our Table includes:
- $F_p$ (multi-pitch detection)
- $F_{voi}$ (voice separation)
- $F_{val}$ (note value)
- $F_{harm}$ (harmony)
- $F_{MV2H}$ (average)

**3. We Add New Metrics Transparently**

STEPn is clearly marked as **our contribution** to measure notation quality. We explain:
- Why it's needed (MV2H doesn't measure beaming, slurs, tuplets)
- How it's computed (tree edit distance on notation tree)
- That it's a **supplement**, not a replacement for MV2H

**4. Onset F1 is Not Suitable for Score Evaluation**

- Onset F1 measures **timing accuracy at millisecond level**
- Scores use **discrete time** (quarter notes, eighth notes)
- A score can be "perfect" (correct note values) even if performance timing varies
- This is why **MV2H exists**: to evaluate symbolic correctness, not performance accuracy

---

## Summary: Evaluation Protocol Defense

| Challenge | Our Defense |
|-----------|-------------|
| Why not unified **Kern? | No standard MIDI→**Kern converter; MusicXML is universal |
| Different conversion paths? | Reflects architectural differences; unified at XML→MIDI stage |
| Why Transkun+Beyer baseline? | Proves we beat Pipeline SOTA, not just weak baselines |
| Reproducibility? | Full code release + exact tool versions + Zeng's test split |
| Cherry-picked metrics? | Use all metrics from Zeng; add STEPn transparently |

**Core Argument**:

> We evaluate systems **as designed** (respecting their native representations) while ensuring fairness through **unified evaluation at the symbolic level** (MusicXML → MIDI → MV2H). This protocol is rigorous, reproducible, and standard practice in music information retrieval research.
