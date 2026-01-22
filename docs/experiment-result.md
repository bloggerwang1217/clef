# Experiment Results

本文件記錄 Clef 研究中各系統在 ASAP Dataset 上的評估結果。

---

## Table of Contents

1. [資料集定義](#資料集定義)
2. [系統比較總覽](#系統比較總覽)
3. [MT3 Baseline 評估](#mt3-baseline-評估)
4. [Transkun + Beyer 評估](#transkun--beyer-評估)
5. [Zeng Model 評估](#zeng-model-評估)
6. [Clef 評估](#clef-評估) (待補充)

---

## 資料集定義

### ASAP Test Split (Zeng 定義)

| 項目 | 數量 | 說明 |
|-----|------|------|
| Pieces | 25 | 曲目數 |
| Performances | 80 | 演奏錄音數（從 ASAP 186 個中選出） |
| **Chunks** | **13,335** | 5-bar chunks (stride=1) |

**定義檔案**：`src/evaluation/asap/test_chunk_set.csv`

> **Note**: 13,335 chunks 是根據 Ground Truth 樂譜的小節數計算出的完整測試集定義。不同系統的差異在於「能成功評估多少」，而非「定義了多少」。

---

## 系統比較總覽

### 資料來源索引

| 系統 | 評估結果 CSV | Summary JSON | Error Log | 狀態 |
|-----|-------------|--------------|-----------|------|
| **MT3** | `data/experiments/mt3/results/chunks_song.csv` | `chunks_song.summary.json` | `full_musicxml/errors.txt` | ✅ 完成 |
| **Transkun + Beyer** | `data/experiments/transkun_beyer/results/chunks.csv` | `chunks.summary.json` | - | ✅ 完成 |
| **Zeng (hum2xml)** | `/home/bloggerwang/piano-a2s/results/chunk_results.csv` | - | - | ✅ 完成 |
| **Clef** | (待產生) | (待產生) | (待產生) | ⏳ 待跑 |

### 評估結果比較表

基於 13,335 chunks (Zeng test split 完整定義)：

| 系統 | Type | Success Rate | Success | Failed | MV2H_custom (成功集) | MV2H_custom (全集) |
|-----|------|-------------|---------|--------|---------------------|-------------------|
| MT3 + MuseScore | Pipeline | 35.1% | 4,685 | 8,650 | 56.5% | 19.9% |
| **Transkun + Beyer** | **Pipeline** | **66.0%** | **8,806** | **4,529** | **78.7%** | **52.0%** |
| Zeng (hum2xml) | E2E | 88.2%* | 3,262 | 438 | 75.1% | 66.2% |
| Clef | E2E | ? | ? | ? | ? | ? |

> **Note**: Zeng 的數據基於 3,700 chunks 的子集評估（非完整 13,335）

### 關鍵發現

1. **Survivorship Bias 的典型案例**
   - Transkun + Beyer 成功集 MV2H (78.7%) > Zeng (75.1%)
   - 但這是因為 Transkun + Beyer 失敗了 34% 的「困難」chunks
   - **全集比較才是公平的**：Zeng (66.2%) > Transkun + Beyer (52.0%)

2. **SOTA Pipeline 大幅優於 Industrial Pipeline**
   - Transkun + Beyer: 66.0% success rate, 52.0% MV2H_custom (全集)
   - MT3 + MuseScore: 35.1% success rate, 19.9% MV2H_custom (全集)
   - Beyer Transformer 的學習式量化優於 MuseScore 的規則式量化

3. **E2E 方法的優勢**
   - Zeng 的 success rate (88.2%) 遠高於 Pipeline 方法
   - E2E 方法不會發生 Mode Locking（見 evaluation-protocol.md）
   - 每個 chunk 獨立評估，不受前序錯誤影響

---

## MT3 Baseline 評估

### 資料來源

| 檔案類型 | 路徑 | 說明 |
|---------|------|------|
| **評估結果** | `data/experiments/mt3/results/chunks_song.csv` | 13,335 chunks 的詳細評估結果 |
| **統計摘要** | `data/experiments/mt3/results/chunks_song.summary.json` | 統計數據 JSON |
| **錯誤記錄** | `data/experiments/mt3/full_musicxml/errors.txt` | 8,650 個失敗 chunks 的詳細記錄 |
| **Chunk MIDI** | `data/experiments/mt3/full_musicxml/chunk_midi/` | 擷取的 chunk MIDI 檔案 |
| **MusicXML** | `data/experiments/mt3/full_musicxml/` | MuseScore 轉換的 MusicXML |

### 評估流程

```
MT3 Pipeline (Baseline):
  Audio (full song)
    → MT3 → Raw MIDI (no measure structure)
    → MuseScore 4.6.5 → MusicXML (quantized, with measures)
    → music21 extract measures [start:end]
    → Chunk MIDI
    → MV2H evaluation vs Ground Truth chunk
```

### 評估結果

#### Success Rate

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Chunks | 13,335 | 100% |
| **Successful** | 4,685 | **35.1%** |
| Failed | 8,650 | 64.9% |

#### Failure Breakdown

| Status | Count | Percentage | 說明 |
|--------|-------|------------|------|
| `success` | 4,685 | 35.1% | MV2H 評估成功 |
| `mv2h_failed` | 5,741 | 43.1% | MV2H DTW timeout（Phase Drift 導致對齊失敗）|
| `zero_score` | 2,909 | 21.8% | Prediction MIDI 為空（小節超出範圍）|

#### MV2H Scores

**Zeng's Method（排除失敗，n=4,685）**：

| Metric | Score |
|--------|-------|
| Multi-pitch | 21.92% |
| Voice | 56.94% |
| Meter | 12.64% |
| Value | 71.22% |
| Harmony | 76.09% |
| MV2H (official) | 47.76% |
| **MV2H_custom** | **56.54%** |

**Include Failures（全集，n=13,335）**：

| Metric | Score |
|--------|-------|
| Multi-pitch | 7.70% |
| Voice | 20.00% |
| Meter | 4.44% |
| Value | 25.02% |
| Harmony | 26.73% |
| MV2H (official) | 16.78% |
| **MV2H_custom** | **19.86%** |

### Phase Drift 分析

#### Success Rate by Chunk Position

| Chunk Position | Success Rate | Sample Count | Interpretation |
|----------------|--------------|--------------|----------------|
| 1-10 | **50.9%** | 666 | 開頭尚可對齊 |
| 11-20 | 40.4% | 713 | 開始漂移 |
| 21-30 | 46.7% | 719 | 波動 |
| 31-40 | 54.0% | 724 | 局部穩定 |
| 41-50 | 45.9% | 725 | 持續波動 |
| 51-100 | 42.2% | 3,032 | 明顯下降 |
| 101-200 | **31.3%** | 4,456 | 嚴重漂移 |
| 201-500 | **13.0%** | 2,253 | 完全崩潰 |

#### Failure Mode by Chunk Position

| Status | Mean Index | Median Index | Count | Interpretation |
|--------|------------|--------------|-------|----------------|
| `success` | 85.7 | **70** | 4,685 | 成功案例集中在前半段 |
| `zero_score` | 188.3 | **196** | 2,909 | 空 MIDI 集中在後半段 |
| `mv2h_failed` | 96.5 | **82** | 5,741 | DTW 超時集中在中段 |

#### Phase Drift 機制

```
時間軸示意圖：

Ground Truth:
|--Bar 1--|--Bar 2--|--Bar 3--|...|--Bar 50--|--Bar 51--|...|--Bar 200--|
   2s        2s        3s (rubato)     2s         2s            2s

MT3 + MuseScore (量化後):
|--Bar 1--|--Bar 2--|--Bar 3--|-Bar 4-|...|--Bar 52--|...|--Bar 220--|
   2s        2s        2s       1s           2s              2s
                       ↑
                  MuseScore 把 3s 的 rubato
                  硬塞進 2s，多出的音跑到 Bar 4

結果：
- GT Bar 50 ≠ MS4 Bar 50（內容不同）
- GT Bar 200 → MS4 可能已經是 Bar 220（相位漂移 10%）
- 當 chunk_index > 200 時，擷取出的 MIDI 可能完全是空的或錯誤的音
```

### Root Cause Analysis

#### 為什麼 Multi-pitch 只有 21.92%？

即使 MT3 的 onset detection 準確率高達 90%+，MV2H 的 Multi-pitch 卻只有 21.92%。

**原因**：MV2H 評估的是「在正確的小節裡有沒有正確的音」，而非物理時間的 onset 準確度。

```
範例：
- GT Bar 30: [C4, E4, G4]（正確的和弦）
- MS4 Bar 30: [D4, F4, A4]（因為相位漂移，這其實是 GT Bar 28 的音）
- MV2H: 3 個 False Negative + 3 個 False Positive = 0% precision/recall
```

#### 為什麼 `zero_score` 集中在後段？

| Chunk Index Range | `zero_score` Count | Interpretation |
|-------------------|-------------------|----------------|
| 1-100 | ~200 | 少數邊界情況 |
| 100-200 | ~800 | 漂移開始顯現 |
| **200-500** | **~1,100** | 漂移嚴重，MS4 的小節已超出實際曲長 |

**機制**：
1. MT3 輸出的 MIDI 時長 = 實際音訊時長（例如 5 分鐘）
2. MuseScore 量化時，因為 rubato，產生了「額外的小節」
3. 當請求 measure 200-204 時，MS4 的 MusicXML 可能只有 180 個小節
4. music21 擷取結果 = 空的 MIDI = `zero_score`

### 結論

本實驗證明了 **Audio → MIDI → Score** 的 Pipeline 方法存在根本性缺陷：

1. **無法維持全域結構一致性 (Global Structural Consistency)**
   - MT3 只輸出物理時間的音符，沒有小節概念
   - MuseScore 的啟發式量化無法正確處理 Rubato
   - 累積誤差導致後段小節完全錯位

2. **量化災難 (Quantization Artifacts)**
   - 表達性演奏（Rubato）被硬塞進固定拍號
   - 產生大量碎片化的音符和休止符
   - 導致 MV2H DTW 對齊超時

3. **Phase Drift（相位漂移）**
   - Success rate 從前段的 53% 下降到後段的 14%
   - `zero_score` 集中在 chunk_index > 200 的區域
   - 證明 Pipeline 方法無法處理長篇幅的音樂

### Reproduction

```bash
# 執行完整評估 pipeline
cd /home/bloggerwang/clef
poetry run ./src/baselines/mt3/run_mt3_evaluate_pipeline.sh --mode chunks -j 4
```

詳見 `configs/mt3_evaluate.yaml`

### Appendix: Raw Data

#### errors.txt 格式

```
# Total: 8650 chunks with errors
# Format: chunk_id	status	error_message
Bach#Prelude#bwv_875#Ahfat01M.12	mv2h_failed	MV2H returned None (check Java process logs)
Bach#Prelude#bwv_875#Ahfat01M.32	zero_score	MV2H returned 0 (likely MIDI parsing error)
...
```

#### Summary JSON

```json
{
  "n_total": 13335,
  "n_evaluated": 13335,
  "n_successful": 4685,
  "n_failed": 8650,
  "success_rate": 0.351,
  "status_breakdown": {
    "mv2h_failed": 5741,
    "success": 4685,
    "zero_score": 2909
  },
  "zeng_method": {
    "Multi-pitch": 0.2192,
    "Voice": 0.5694,
    "Meter": 0.1264,
    "Value": 0.7122,
    "Harmony": 0.7609,
    "MV2H": 0.4776,
    "MV2H_custom": 0.5654
  },
  "include_failures": {
    "Multi-pitch": 0.0770,
    "Voice": 0.2000,
    "Meter": 0.0444,
    "Value": 0.2502,
    "Harmony": 0.2673,
    "MV2H": 0.1678,
    "MV2H_custom": 0.1986
  }
}
```

---

## Transkun + Beyer 評估

### 系統說明

**Transkun + Beyer** 是目前 SOTA 的 Audio-to-Score Pipeline：
- **Transkun** (Li et al., 2023): SOTA AMT 模型，輸出 raw MIDI
- **Beyer** (Beyer et al., 2024): Transformer-based 量化模型，MIDI → MusicXML

這個系統比 MT3 + MuseScore 更先進，因為量化步驟是學習式的（Transformer）而非規則式的（MuseScore）。

### 資料來源

| 檔案類型 | 路徑 | 說明 |
|---------|------|------|
| **評估結果** | `data/experiments/transkun_beyer/results/chunks.csv` | 13,335 chunks 的詳細評估結果 |
| **統計摘要** | `data/experiments/transkun_beyer/results/chunks.summary.json` | 統計數據 JSON |
| **Chunk MIDI** | `data/experiments/transkun_beyer/chunk_midi/` | 擷取的 chunk MIDI 檔案 |
| **MusicXML** | `data/experiments/transkun_beyer/musicxml/` | Beyer 轉換的 MusicXML |

### 評估流程

```
Transkun + Beyer Pipeline (SOTA):
  Audio (full song)
    → Transkun → Raw MIDI (no measure structure)
    → Beyer Transformer → MusicXML (quantized, with measures)
    → music21 extract measures [start:end]
    → Chunk MIDI
    → MV2H evaluation vs Ground Truth chunk
```

### 評估結果

#### Success Rate

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Chunks | 13,335 | 100% |
| **Successful** | 8,806 | **66.0%** |
| Failed | 4,529 | 34.0% |

#### Failure Breakdown

| Status | Count | Percentage | 說明 |
|--------|-------|------------|------|
| `success` | 8,806 | 66.0% | MV2H 評估成功 |
| `mv2h_failed` | 4,217 | 31.6% | MV2H DTW timeout |
| `zero_score` | 312 | 2.3% | Prediction MIDI 為空 |

#### MV2H Scores

**Zeng's Method（排除失敗，n=8,806）**：

| Metric | Score |
|--------|-------|
| Multi-pitch | 65.90% |
| Voice | 89.16% |
| Meter | 40.36% |
| Value | 87.83% |
| Harmony | 71.79% |
| MV2H (official) | 71.01% |
| **MV2H_custom** | **78.67%** |

**Include Failures（全集，n=13,335）**：

| Metric | Score |
|--------|-------|
| Multi-pitch | 43.52% |
| Voice | 58.88% |
| Meter | 26.65% |
| Value | 58.00% |
| Harmony | 47.41% |
| MV2H (official) | 46.89% |
| **MV2H_custom** | **51.95%** |

### 與 MT3 比較

| Metric | MT3 + MuseScore | Transkun + Beyer | 提升 |
|--------|-----------------|------------------|------|
| Success Rate | 35.1% | **66.0%** | **+30.9%** |
| MV2H_custom (成功集) | 56.5% | **78.7%** | **+22.2%** |
| MV2H_custom (全集) | 19.9% | **52.0%** | **+32.1%** |

**關鍵發現**：
1. **Beyer Transformer 大幅優於 MuseScore**
   - Success rate 從 35.1% 提升到 66.0%
   - 學習式量化能更好地處理 rubato

2. **但仍無法與 E2E 方法競爭**
   - Zeng (E2E) 全集 MV2H: 66.2%
   - Transkun + Beyer (Pipeline) 全集 MV2H: 52.0%
   - 差距 14.2%，證明 Pipeline 的根本限制

3. **Survivorship Bias 的警示**
   - 成功集 MV2H: Transkun + Beyer (78.7%) > Zeng (75.1%)
   - 全集 MV2H: Zeng (66.2%) > Transkun + Beyer (52.0%)
   - **只報告成功集會得出錯誤結論**

### 結論

即使是 SOTA Pipeline (Transkun + Beyer)，在全集評估下仍然落後 E2E 方法 (Zeng) 14.2%。這證明了：

1. **Pipeline 方法的根本限制**：即使用 Transformer 學習量化，仍無法完全解決 Phase Drift
2. **全集評估的重要性**：成功集的分數會產生 Survivorship Bias，誤導比較結果
3. **E2E 方法的優勢**：直接輸出小節結構，避免累積誤差

---

## Zeng Model 評估

### 背景：為何需要回到 Zeng 的原始 Pipeline？

我們最初嘗試用自己的 `converter21` pipeline 來評估 Zeng 模型，希望能：
1. 提高 Success Rate（救回更多 chunks）
2. 使用標準化的轉換工具

但實驗結果揭示了一個重要問題：

#### 實驗：converter21 Pipeline 評估 Zeng 模型

| 項目 | 值 |
|------|---|
| Pipeline | `converter21` + `clean_kern.py` |
| Evaluated Chunks | 8,417 / 9,363 |
| Success Rate | **89.9%** |

| Metric | Score |
|--------|-------|
| **Multi-pitch** | **45.57%** |
| Voice | 88.55% |
| Meter | 24.14% |
| Value | 89.10% |
| Harmony | 58.15% |
| **MV2H** | **61.10%** |

#### 對照：接近 Zeng 原始 Pipeline 的評估

| 項目 | 值 |
|------|---|
| Pipeline | music21（接近 Zeng 原始做法） |
| Evaluated Chunks | ~3,700 / ~8,111 |
| Success Rate | **~46%** |

| Metric | Score |
|--------|-------|
| **Multi-pitch** | **63.95%** |
| Voice | 88.93% |
| Meter | 33.16% |
| Value | 89.32% |
| Harmony | 57.20% |
| **MV2H** | **66.51%** |

#### 並排比較

| Metric | Zeng-like Pipeline | converter21 Pipeline | 差異 |
|--------|-------------------|---------------------|------|
| Success Rate | ~46% | **89.9%** | **+44%** |
| Multi-pitch | **63.95%** | 45.57% | **-18.4%** |
| Voice | 88.93% | 88.55% | -0.4% |
| Meter | **33.16%** | 24.14% | -9.0% |
| Value | 89.32% | 89.10% | -0.2% |
| Harmony | 57.20% | 58.15% | +1.0% |
| MV2H | **66.51%** | 61.10% | -5.4% |

### 關鍵發現：Multi-pitch 大幅下降的原因

**Multi-pitch 從 63.95% 下降到 45.57%（-18%）** 揭示了重要訊息：

1. **Zeng 模型可能 Overfit 在髒資料上**
   - Zeng 的訓練資料可能包含特定格式偏差（如 `8rGG`、stem directions）
   - 當輸入經過標準化（converter21）後，這些 artifacts 消失
   - 模型反而無法正確辨識「乾淨」的資料

2. **Survivorship Bias（倖存者偏差）**
   - Zeng pipeline 的低 Success Rate（~46%）可能不是缺點
   - 它可能「剛好」只處理了模型熟悉的簡單/髒資料
   - 這些資料的分數自然較高

3. **converter21 救回的 Chunks 是更難的樣本**
   - 多救回了 ~4,700 個 chunks（89.9% - 46% ≈ 44%）
   - 這些困難樣本拉低了平均分數
   - 但這也揭示了 Zeng 模型的 generalizability 問題

### 結論：公平比較策略

> **核心原則：每個系統用自己的原生 Pipeline = 系統級公平比較**

用我們的 converter21 去評估 Zeng 模型是「不公平」的：
- Zeng 模型是在 hum2xml pipeline 上訓練的
- 強迫它接受標準化輸入會產生 train/test mismatch

因此，最終比較策略：

| System | Pipeline | 理由 |
|--------|----------|------|
| **Zeng** | hum2xml（原始） | 模型的原生環境 |
| **MT3** | MuseScore 4 | Pipeline baseline 的標準做法 |
| **Clef** | converter21 | 我們的標準化 pipeline |

### 待執行：Zeng hum2xml Pipeline 完整評估

#### 資料來源

| 檔案類型 | 路徑 | 說明 |
|---------|------|------|
| **評估結果** | (待產生) | |
| **統計摘要** | (待產生) | |
| **錯誤記錄** | (待產生) | |

#### 評估環境

| 項目 | 值 |
|------|---|
| Pipeline | `hum2xml`（Zeng 原本） |
| Test Set | ASAP test split (13,335 chunks) |
| Timeout | 10 秒（待確認是否延長到 300 秒）|
| MV2H Version | Non-aligned (McLeod 2019) |

#### 待執行事項

- [ ] 重跑 Zeng hum2xml pipeline
- [ ] 記錄完整的 success/timeout/error log
- [ ] 計算 Success Rate 和 MV2H 分數

### 參考：Zeng 論文報告值 vs 我們的復現

| Metric | Zeng 論文 (2024) | 我們的復現 (2026-01-16) | 差異 |
|--------|-----------------|------------------------|------|
| Multi-pitch | 63.30% | **63.95%** | +0.65% |
| Voice | 88.40% | 88.93% | +0.53% |
| Value | 90.70% | 89.32% | -1.38% |
| Harmony | 54.50% | 57.20% | +2.70% |
| Meter | (未報告) | 33.16% | - |
| **MV2H** | **74.20%** | **74.85%** | +0.65% |

> **驗證成功**：我們的復現結果與 Zeng 論文報告值高度吻合（差異 < 3%），確認評估流程正確。

> **Note**: 論文未報告評估覆蓋率（Success Rate），需要重跑確認實際成功數量。MV2H 為 4-metric 版本（排除 Meter）。

---

## Clef 評估

(待補充)

---

*Last updated: 2026-01-22*
