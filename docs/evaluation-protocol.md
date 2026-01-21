# Evaluation Protocol: MV2H Evaluation and Mode Locking Analysis

本文件說明 Clef 專案的 MV2H 評估流程，以及 Pipeline vs E2E 方法的 Mode Locking 比較分析。

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
| **Mode Locking** | 嚴重 | 中等 | 無 |

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

**問題：MuseScore 若在曲首無法正確推斷拍號/tempo，整首曲子都會鎖定在錯誤模式（Mode Locking）**

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

**優勢：模型直接輸出小節結構，每個 chunk 獨立評估，不會被早期錯誤鎖定**

---

## 5-Bar Chunk Evaluation Protocol

### Chunk 定義

使用 Zeng et al. 的 test set：
- 來源：`test_chunk_set.csv`
- 格式：`chunk_id, piece, performance, chunk_index, start_measure, end_measure`
- 總數：13,335 chunks（MT3）/ 3,700 chunks（Zeng preprocessing 後）

### 評估流程

1. **載入 chunk 定義** from CSV
2. **轉換 Prediction MIDI → MusicXML** (MuseScore 4，每個 performance 一次)
3. **Batch 擷取 chunks** (music21，每個 MusicXML parse 一次)
4. **MV2H 評估** (parallel，10s timeout per chunk)

### Success 定義

| Status | Classification | Physical Meaning |
|--------|----------------|------------------|
| `success` (MV2H > 0) | Evaluable | MV2H 能對齊並計算分數 |
| `mv2h_failed` | Not Evaluable | MV2H 無法對齊（結構差異過大）|
| `zero_score` | Not Evaluable | MV2H 返回 0（MIDI 解析錯誤）|

---

## Mode Locking Analysis

### 現象描述

**Mode Locking（模式鎖定）**：Pipeline 方法若在曲首無法正確推斷拍號或 tempo（如遇到弱起拍），則整首曲子都會被「鎖定」在錯誤模式，無法恢復。

這與「累積誤差」(Phase Drift) 不同：
- ~~累積誤差~~：誤差隨時間逐漸累積惡化
- **Mode Locking**：早期就決定成敗，之後維持不變

### 統計驗證

我們透過 **Recovery Rate** 來量化 Mode Locking 的嚴重程度。

#### Recovery Rate 定義

```
Recovery Rate = P(success | prev_fail)

定義：給定「前一個 chunk 評估失敗」，當前 chunk 評估成功的機率。

計算方式：
1. 對每個 performance，將 chunks 按位置排序，得到 success/fail 序列
2. 統計所有相鄰 chunk pairs 中：
   - F→S (前一個失敗，當前成功) 的數量
   - F→F (前一個失敗，當前失敗) 的數量
3. Recovery Rate = count(F→S) / (count(F→S) + count(F→F))

分析單位：相鄰 chunk pairs（不是 performances）
```

#### 物理意義

| Recovery Rate | 意義 |
|---------------|------|
| **~50%** | 無 Mode Locking，失敗後有一半機會恢復（接近隨機）|
| **~10%** | 嚴重 Mode Locking，失敗後幾乎無法恢復 |
| **~70%** | 無 Mode Locking，失敗後大多能恢復 |

### 實驗結果

| Method | Type | After-Fail Pairs (n) | Recovery Rate |
|--------|------|---------------------|---------------|
| MT3 + MuseScore | Pipeline | 8,591 | **9.6%** |
| Zeng | E2E | 430 | **67.9%** |

**關鍵發現**：
- **Pipeline (MT3)**：失敗後只有 9.6% 機率恢復 → 嚴重 Mode Locking
- **E2E (Zeng)**：失敗後有 67.9% 機率恢復 → 每個 chunk 獨立評估

### 解釋

**為什麼 Pipeline 會 Mode Lock？**

MuseScore 的量化推斷是「全局性」的：
1. 從 MIDI 開頭推斷 tempo 和拍號
2. 一旦推斷錯誤（如遇到弱起拍），整首曲子的小節邊界都會偏移
3. 所有後續 chunks 都會因為小節對不齊而評估失敗

**為什麼 E2E 不會 Mode Lock？**

E2E 模型直接輸出小節結構：
1. 每個 chunk 的小節邊界由模型獨立決定
2. 前一個 chunk 的錯誤不會影響下一個 chunk
3. 即使某個 chunk 失敗，下一個仍有高機率成功

---

## Current Results

### MV2H Scores

| Model | n_success | n_total | Coverage | Multi-pitch | Voice | Value | Harmony | MV2H |
|-------|-----------|---------|----------|-------------|-------|-------|---------|------|
| **Zeng (Zeng Method)** | 3,262 | 3,700 | 88.2% | 64.48% | 88.98% | 89.35% | 57.56% | 75.09% |
| **Zeng (Strict)** | 3,262 | 3,700 | 88.2% | 56.84% | 78.44% | 78.77% | 50.74% | 66.20% |
| **MT3 (Zeng Method)** | 4,685 | 13,335 | 35.1% | 21.92% | 56.94% | 71.22% | 76.09% | 47.76% |
| **MT3 (Strict)** | 4,685 | 13,335 | 35.1% | 7.70% | 20.00% | 25.02% | 26.73% | 16.78% |

### Mode Locking Comparison

| Method | Type | Recovery Rate | Interpretation |
|--------|------|---------------|----------------|
| MT3 + MuseScore | Pipeline | 9.6% | Severe Mode Locking |
| Zeng | E2E | 67.9% | No Mode Locking |

---

## Implementation Details

### Files

| File | Description |
|------|-------------|
| `src/baselines/mt3/mt3_evaluate.py` | MT3 + MuseScore evaluation pipeline |
| `src/evaluation/asap.py` | ASAP dataset handler, chunk extraction |
| `src/analysis/analyze_mv2h_results.py` | Mode Locking analysis and visualization |
| `configs/mt3_evaluate.yaml` | Configuration file |

### Running Evaluation

```bash
# MT3 + MuseScore chunk evaluation
./src/baselines/mt3/run_mt3_evaluate_pipeline.sh --mode chunks

# Or directly with Python
poetry run python -m src.baselines.mt3.mt3_evaluate \
    --config configs/mt3_evaluate.yaml

# Run analysis (generates plots)
poetry run python src/analysis/analyze_mv2h_results.py
```

### Output Plots

| Plot | Description |
|------|-------------|
| `results/pipeline_vs_e2e.png` | Recovery Rate comparison between Pipeline and E2E |

---

## Summary

**Key Finding: Mode Locking distinguishes Pipeline from E2E methods**

Pipeline 方法（如 MT3 + MuseScore）會發生 Mode Locking：若 MuseScore 在曲首無法正確推斷拍號/tempo，整首曲子都會被鎖定在錯誤模式（Recovery Rate = 9.6%）。

E2E 方法（如 Zeng）不會發生 Mode Locking：每個 chunk 獨立評估，即使某個 chunk 失敗，下一個仍有高機率成功（Recovery Rate = 67.9%）。

這是因為 E2E 方法直接輸出小節結構，而 Pipeline 方法需要從 continuous time 全局推斷小節邊界，容易在曲首就鎖定錯誤模式。
