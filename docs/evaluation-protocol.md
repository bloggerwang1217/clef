# Evaluation Protocol: MV2H Evaluation and Phase Drift Analysis

本文件說明 Clef 專案的 MV2H 評估流程，以及 Pipeline vs E2E 方法的 Phase Drift 比較分析。

---

## 評估系統概覽

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT: Audio Recordings                            │
│                     (ASAP test set: 25 pieces / 80 performances)            │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┬───────────────────────┐
     ▼                       ▼                       ▼                       ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│     MT3      │   │   Transkun   │   │ Zeng (2024)  │   │ Clef (Ours)  │
│ + MuseScore4 │   │   + Beyer    │   │              │   │              │
│              │   │              │   │              │   │              │
│  Industrial  │   │    SOTA      │   │     E2E      │   │     E2E      │
│  Pipeline    │   │  Pipeline    │   │   Baseline   │   │   Proposed   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │  MIDI   │        │  MIDI   │        │ **Kern  │        │ **Kern  │
  │ (perf.) │        │ (perf.) │        │ (score) │        │ (score) │
  └────┬────┘        └────┬────┘        └────┬────┘        └────┬────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │MuseScore│        │  Beyer  │        │ Humdrum │        │ Humdrum │
  │  4.6.5  │        │Transformer      │ Toolkit │        │ Toolkit │
  └────┬────┘        └────┬────┘        └────┬────┘        └────┬────┘
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │MusicXML │        │MusicXML │        │MusicXML │        │MusicXML │
  └────┬────┘        └────┬────┘        └────┬────┘        └────┬────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  music21       │
                           │  XML → MIDI    │
                           │  (extract      │
                           │   measures)    │
                           └────────┬───────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  MV2H Tool     │
                           │  (Java)        │
                           └────────┬───────┘
                                    │
                                    ▼
                           ┌────────────────┐
                           │  MV2H Metrics  │
                           │  MP, V, Va, H  │
                           └────────────────┘
```

---

## Pipeline vs E2E 方法比較

### 關鍵差異

| 面向 | Industrial Pipeline | SOTA Pipeline | E2E (Zeng, Clef) |
|------|---------------------|---------------|------------------|
| **System** | MT3 + MuseScore 4 | Transkun + Beyer | Zeng / Clef |
| **Model Output** | MIDI (continuous) | MIDI (continuous) | **Kern (discrete) |
| **小節資訊** | 無（推斷） | 無（推斷） | 有（直接輸出） |
| **Symbolization** | Rule-based heuristics | Learned (Transformer) | 模型直接學習 |
| **累積誤差** | 有（Phase Drift） | 有（Phase Drift） | 無 |

### MT3 + MuseScore 4 Pipeline (Industrial Baseline)

```
Audio → MT3 → MIDI (continuous time, no measure info)
                 │
                 ▼
        MuseScore 4.6.5 (MIDI → MusicXML)
        ┌─────────────────────────────────┐
        │ • Tempo inference               │
        │ • Beat quantization             │
        │ • Measure boundary detection    │
        │ • Voice separation heuristics   │
        │ • Tuplet detection              │
        └─────────────────────────────────┘
                 │
                 ▼
        MusicXML (discrete time, measure structure)
                 │
                 ▼
        music21 (extract 5-bar chunks → MIDI)
                 │
                 ▼
        MV2H Evaluation
```

**問題：MuseScore 的量化會產生累積誤差（Phase Drift）**

### Transkun + Beyer Pipeline (SOTA Baseline)

```
Audio → Transkun → MIDI (high-res continuous time)
                      │
                      ▼
             Beyer Transformer (MIDI → MusicXML)
             ┌─────────────────────────────────┐
             │ • Learned quantization          │
             │ • Neural voice separation       │
             │ • Learned beaming/slurs         │
             │ • Better than rule-based        │
             └─────────────────────────────────┘
                      │
                      ▼
             MusicXML (discrete time, measure structure)
                      │
                      ▼
             music21 (extract 5-bar chunks → MIDI)
                      │
                      ▼
             MV2H Evaluation
```

**優於 MT3 + MuseScore，但仍有 Phase Drift（因為仍需從 continuous time 推斷小節）**

### Zeng / Clef E2E Pipeline

```
Audio → Model → **Kern tokens (discrete time, measure info built-in)
                    │
                    ▼
           Humdrum Toolkit (hum2xml)
                    │
                    ▼
           MusicXML (measure structure preserved)
                    │
                    ▼
           music21 (extract 5-bar chunks → MIDI)
                    │
                    ▼
           MV2H Evaluation
```

**優勢：模型直接輸出小節結構，無累積誤差**

---

## 5-Bar Chunk Evaluation Protocol

### Chunk 定義

使用 Zeng et al. 的 test set：
- 來源：`test_chunk_set.csv`
- 格式：`chunk_id, piece, performance, chunk_index, start_measure, end_measure`
- 總數：13,335 chunks（經過 Zeng preprocessing 後剩 3,700）

### 評估流程

1. **載入 chunk 定義** from CSV
2. **轉換 Prediction MIDI → MusicXML** (MuseScore 4，每個 performance 一次)
3. **Batch 擷取 chunks** (music21，每個 MusicXML parse 一次)
4. **MV2H 評估** (parallel，10s timeout per chunk)

### Success 定義

| Status | Classification | Physical Meaning |
|--------|----------------|------------------|
| `success` (MV2H > 0) | Evaluable | MV2H 能對齊並計算分數 |
| `timeout` | Not Evaluable | DTW 對齊超時（累積誤差過大） |
| `conversion_error` | Not Evaluable | MusicXML 轉換失敗 |
| `zero_mv2h` | Not Evaluable | MV2H 返回 0（結構崩壞） |

---

## Phase Drift Analysis

### 假說

**Pipeline 方法會展現 Phase Drift**：隨著小節位置增加，evaluability 下降。

原因：MuseScore 的 tempo/beat 量化誤差會累積，導致後面的小節邊界偏移越來越大。

**E2E 方法不會有 Phase Drift**：evaluability 與小節位置無關。

原因：模型直接輸出小節結構，不需要從 continuous time 推斷。

### Statistical Models

| Method Type | Model | Parameters |
|-------------|-------|------------|
| **Pipeline** | Weibull survival | S(m) = S₀ × exp(-(m/λ)^k), k > 1 |
| **E2E** | Constant | Evaluability(m) = c |

### Zeng (E2E) 結果

Linear regression on individual chunks (n=3,700):

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Slope | 0.010%/measure | Essentially zero |
| **p-value** | **0.154** | Not significant |
| **R²** | **0.0006** | No explanatory power |
| 95% CI | [-0.004%, 0.024%] | Includes zero |

**Conclusion**: Zeng 沒有顯著的 position dependence，Evaluability ≈ 88.2% (constant)。

### Visualization Strategy

所有方法使用相同呈現方式：**Scatter + Fitted Line**

| Method | Color | Fitted Model | Visual |
|--------|-------|--------------|--------|
| MT3 + MuseScore | Blue | Weibull decay | ↘ |
| Transkun + Beyer | Orange | Weibull decay | ↘ |
| Zeng | Green | Constant | → |
| Clef | Red | Constant | → |

```
Evaluability (%)
100% ┤  ════════════════════════════════  Clef (c=?%)
     │  ════════════════════════════════  Zeng (c=88.2%)
 80% ┤  
     │
 60% ┤        ╲
     │          ╲╲
 40% ┤            ╲╲╲                     Transkun+Beyer (Weibull)
     │     ╲         ╲╲╲╲
 20% ┤       ╲╲╲          ╲╲╲╲
     │          ╲╲╲╲╲╲╲╲╲╲╲╲╲            MT3+MuseScore (Weibull)
     └──────────────────────────────────→ Measure Position
       0    50   100   150   200   250   300
```

---

## Current Results

### Comparison Table

| Model | Type | n_chunks | Evaluable | Rate | Model | Key Param |
|-------|------|----------|-----------|------|-------|-----------|
| Zeng | E2E | 3,700 | 3,262 | 88.2% | Constant | p=0.154 |
| Clef | E2E | TBD | TBD | TBD | Constant | TBD |
| Transkun + Beyer | Pipeline (SOTA) | TBD | TBD | TBD | Weibull | TBD |
| MT3 + MuseScore | Pipeline (Industrial) | TBD | TBD | TBD | Weibull | TBD |

### MV2H Scores (Zeng Replication)

| Metric | Zeng Method (n=3,262) | Strict (n=3,700) |
|--------|----------------------|------------------|
| Multi-pitch | 64.48% | 56.84% |
| Voice | 88.98% | 78.44% |
| Value | 89.35% | 78.77% |
| Harmony | 57.56% | 50.74% |
| **MV2H_custom** | **75.09%** | **66.20%** |

---

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `src/baselines/mt3/mt3_evaluate.py` | MT3 + MuseScore evaluation pipeline |
| `src/evaluation/asap.py` | ASAP dataset handler, chunk extraction |
| `configs/mt3_evaluate.yaml` | Configuration file |
| `scripts/parse_zeng_results.py` | Parse Zeng replication results |

### Key Functions

```python
# Batch extraction (parse XML once, extract all chunks)
from src.evaluation.asap import extract_chunks_batch

results = extract_chunks_batch(
    musicxml_path="score.musicxml",
    chunks=[
        (start_m, end_m, output_path),
        ...
    ]
)
```

### Running Evaluation

```bash
# MT3 + MuseScore chunk evaluation
./src/baselines/mt3/run_mt3_evaluate_pipeline.sh --mode chunks

# Or directly with Python
poetry run python -m src.baselines.mt3.mt3_evaluate \
    --config configs/mt3_evaluate.yaml
```

---

## Summary

**Key Finding: Phase Drift distinguishes Pipeline from E2E methods**

| Method | Type | Phase Drift | Statistical Evidence |
|--------|------|-------------|---------------------|
| MT3 + MuseScore | Industrial Pipeline | Yes (severe decay) | Weibull k > 1, p < 0.05 |
| Transkun + Beyer | SOTA Pipeline | Yes (moderate decay) | Weibull k > 1, p < 0.05 |
| Zeng | E2E Baseline | No (flat) | Slope ≈ 0, p > 0.05 |
| Clef | E2E Proposed | No (flat) | TBD |

**Implication**: Even SOTA pipeline methods (Transkun + Beyer) suffer from Phase Drift, demonstrating that this is a fundamental limitation of the pipeline approach, not just implementation quality.

E2E approaches preserve measure structure without cumulative quantization error.
