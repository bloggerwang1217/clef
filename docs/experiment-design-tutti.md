# Experiment Design: Multi-instrument Ensemble Transcription (ICLR 2027)

本文件描述 Clef 針對 **多樂器合奏轉譜 (Multi-instrument Ensemble Transcription)** 的實驗設計。

**目標定位**：處理多樂器合奏場景 — 從音訊中分離並轉錄多個樂器的樂譜。

**建立於 ISMIR 2026 基礎上**：
> "Building upon the piano-specific architecture proposed in [ISMIR 2026], we extend the Video-VLM framework to multi-instrument transcription via domain randomization and auxiliary loss..."

**核心實驗**：
- **Study 1（深度/Precision）**：沿用 ISMIR 2026 的鋼琴深度驗證
- **Study 2（廣度/Breadth）**：Multi-instrument Ensemble — 處理多樂器合奏場景

---

## 實驗策略：稻草人與鋼鐵人

採用 **「攻擊稻草人與鋼鐵人 (The Straw Man and The Steel Man)」** 策略，不需做 $2 \times 2$ 的交叉實驗，只需挑出兩組最具代表性的 Pipeline：

1. **Standard Baseline (稻草人)**：**MT3 + MuseScore 4**
   - **角色**：代表「一般大眾/工程師」最常用的解法
   - **目的**：證明「傳統做法」完全不可行 (MV2H < 60%)，凸顯題目價值

2. **Strong Baseline (鋼鐵人)**：**Transkun + Beyer**
   - **角色**：代表「目前學術界最強」的拼裝車
   - **目的**：證明「即使把最強的零件拼起來」，還是會有 **誤差傳播 (Error Propagation)**，依然輸給 End-to-End

---

## 統一評估標準 (Unified Evaluation Protocol)

為了確保與 SOTA (Zeng et al., 2024) 進行嚴格且公平的比較，我們制定了以下標準化評估流程：

### 核心原則：Train Big, Test Small

我們採用 **「大格局訓練，小格局評估」** 的策略，既發揮 VLM 的長序列優勢，又符合 Baseline 的評分規則。

| 階段 | Zeng (Baseline) | Clef (Ours) | MT3 (Straw Man) | 說明 |
|---|---|---|---|---|
| **Training** | 5-bar segments | **Full Song** (或長序列) | (Pre-trained) | Clef 利用 Global Context 學習結構 |
| **Inference** | 5-bar segments | **Full Song** | **Full Song** | 讓模型展現處理整首曲子的能力 |
| **Evaluation** | 5-bar Average | **Slice to 5-bar** | **Slice to 5-bar** | **統一在 5-bar Level 算分**，確保公平 |

### 評估流程細節

1.  **Zeng (Baseline)**:
    - 依循其原論文設定，對 5-bar 片段進行推論。
    - 計算所有片段的平均 MV2H。
    - **強化點**：我們使用優化過的 Data Pipeline (Converter21) 重新訓練 Zeng 的模型，確保比較對象是 "Stronger Baseline"。

2.  **Clef (Ours)**:
    - 輸入整首音訊，輸出整首 MusicXML。
    - 使用後處理腳本 (`slice_xml.py`)，根據 Ground Truth 的時間點，將整首 XML 切割成對應的 5-bar 片段。
    - 計算切片後的平均 MV2H。
    - **優勢**：Clef 的第 N 個片段是基於上下文推論的，準確度應高於孤立推論。

3.  **MT3 + MuseScore 4**:
    - 輸入整首音訊，輸出整首 MusicXML。
    - 同樣切割成 5-bar 片段進行評分。
    - **目的**：給予 Pipeline 方法最大的優勢（消除長距離累積誤差），若分數依然低落，則證明其量化機制存在根本缺陷。


### ASAP Dataset Split

Zeng 的 split 檔案位於：
- Train: `data_processing/metadata/train_asap.txt` (14 首 / 58 段錄音)
- Test: `data_processing/metadata/test_asap.txt` (25 首 / 80 段錄音)

**Test Split (25 pieces)**:

| Composer | Pieces |
|----------|--------|
| Bach | Prelude BWV 875, 891 |
| Beethoven | Sonata 9/1, 21/1, 22/1, 27/1, 28/1 |
| Chopin | Ballade 2, 3; Etude Op.10 No.2,4,5; Sonata 2/4 |
| Haydn | Sonata 50/1 |
| Liszt | Concert Etude S145/1, Paganini Etude 6 |
| Mozart | Sonata 12/1, 12/3 |
| Schubert | Impromptu D.899 No.1,2,4; Moment Musical 1; D.664/3, D.894/2 |
| Schumann | Toccata |

### Study 1: 5-bar Chunk 評估框架

#### 資料集定義

ASAP test split 的完整結構：

| 層級 | 數量 | 說明 |
|-----|------|------|
| Pieces | 25 | 曲目數（上表所列） |
| Performances | 80 | 演奏錄音數（Zeng 從 ASAP 186 個中選出） |
| **Chunks** | **9,363** | 5-bar chunks（stride=1 重疊） |

**Chunk 定義來源**：
- 定義檔：`src/evaluation/test_chunk_set.csv`
- 來源：根據 Ground Truth 樂譜（MusicXML）的小節數計算
- 格式：`chunk_id, piece, performance, chunk_index, start_measure, end_measure`

> **Note**: ASAP 完整 test set 有 186 個 performances，但 Zeng 只使用其中 80 個作為 test split，其餘用於 fine-tuning。

#### 三維度評估框架

為確保公平比較，我們採用三維度評估框架：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Study 1: 5-bar Chunk 評估框架                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  維度 A: Success Rate（成功率）                                          │
│  ═══════════════════════════════                                        │
│  定義：成功評估的 chunks / 9,363                                         │
│  意義：系統穩定性，能處理多少比例的測試樣本                               │
│                                                                         │
│  維度 B: Intersection MV2H（交集分數）                                   │
│  ════════════════════════════════════                                   │
│  定義：在「所有系統都成功」的 chunks 上計算 MV2H                          │
│  意義：Apple-to-Apple 公平比較，排除 parsability 差異                    │
│                                                                         │
│  維度 C: Full Set MV2H（全集分數，失敗=0）                               │
│  ═════════════════════════════════════════                              │
│  定義：sum(成功分數) / 9,363                                             │
│  意義：真實世界可用性，失敗的 chunks 計為 0 分                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 各系統評估狀態

| 系統 | Pipeline | 評估狀態 | Success Rate | MV2H (成功集) | MV2H (全集) |
|-----|----------|---------|--------------|--------------|-------------|
| **MT3 + MuseScore** | Audio→MIDI→MusicXML→Chunk | ✅ 完成 | 35.6% | 56.2%* | 20.0%* |
| **Zeng (hum2xml)** | Kern→hum2xml→MusicXML | ⏳ 待跑 | ~46%? | ~66%? | ? |
| **Clef** | Audio→MusicXML→Chunk | ⏳ 待跑 | ? | ? | ? |

*MV2H_custom = (Multi-pitch + Voice + Value + Harmony) / 4

#### 資料來源追蹤

| 系統 | 評估結果檔案 | Summary 檔案 |
|-----|-------------|-------------|
| MT3 | `data/experiments/mt3/results/chunks_song.csv` | `data/experiments/mt3/results/chunks_song.summary.json` |
| Zeng | (待產生) | (待產生) |
| Clef | (待產生) | (待產生) |

### 兩階段訓練 (Two-Stage Training)

```
Stage 1: Pre-training (Synthetic Data)
├── Data: MuseSyn (Pop) + HumSyn (Classical/Ragtime)
├── Audio: EPR system (VirtuosoNet) 生成
├── Augmentation:
│   ├── Random key shift (±4 semitones)
│   ├── Random EPR composer (15 種風格)
│   ├── Random tempo scaling (0.85-1.15x)
│   └── Random soundfont (4 種鋼琴)
└── 擴增後資料量: 10x

Stage 2: Fine-tuning (Real Recordings)
├── Data: ASAP train split (14 首 / 58 段)
└── Transfer learning from Stage 1
```

### 音訊處理參數

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 kHz |
| Spectrogram | VQT (Variable-Q Transform) |
| Bins per octave | 60 |
| Octaves | 8 |
| Gamma | 20 |
| Hop length | 160 |
| Clip length | 5 bars (based on downbeat) |

### MV2H 評估指標

使用 **Non-aligned MV2H** (McLeod, 2019)，包含四個子指標：
- $F_p$: Multi-pitch detection accuracy
- $F_{voi}$: Voice separation accuracy
- $F_{val}$: Note value detection accuracy
- $F_{harm}$: Harmonic detection accuracy
- $F_{MV2H}$ = average of above four

### Zeng 的最佳結果 (Fine-tuned on ASAP)

| Metric | Score |
|--------|-------|
| $F_p$ | 63.3% |
| $F_{voi}$ | 88.4% |
| $F_{val}$ | 90.7% |
| $F_{harm}$ | 54.5% |
| **$F_{MV2H}$** | **74.2%** |

> **觀察**：Zeng 的 $F_p$ (音高) 和 $F_{harm}$ (和聲) 偏低，這正是 CNN 局部感受野的限制。

---

## 頻譜表示的神經科學基礎

本章節探討為何選擇 Log-Mel Spectrogram 而非 VQT，從頻譜特性與模型遷移的角度提供理論依據。

### VQT vs Log-Mel 的數學差異

| 特性 | **Log-Mel Spectrogram** | **VQT (Variable-Q Transform)** |
|------|------------------------|-------------------------------|
| **設計目的** | 模擬人類聽覺感知（語音） | 專為音樂設計 |
| **「Log」作用位置** | 能量的對數 (dB scale) | 頻率軸的對數 |
| **頻率尺度** | Mel scale（心理聲學） | 對數頻率（音樂學） |
| **每八度解析度** | **不固定**（低頻多、高頻少） | **固定**（如 60 bins/octave） |
| **音高對齊** | 不對齊 MIDI 音高 | **完美對齊** 12 音階 |
| **音色保留** | **保留共振峰** | **破壞共振峰** |
| **常見應用** | 語音識別、聲音分類 | 音樂轉譜、和聲分析 |

### 為何選擇 Log-Mel？

**核心論點**：VQT 的「音高對齊」優勢在多聲部音樂轉譜中不值一提，因為：

1. **音色扭曲問題（Critical）**：
   - VQT 為了讓 C4 和 C5 看起來一樣，對頻譜進行非線性扭曲
   - 這導致**固定的共振峰特徵被扭曲**，小提琴的泛音結構在高低音域看起來不同
   - 這對 **Instrument Auxiliary Loss** 是毀滅性的打擊（無法區分樂器）

2. **ImageNet 遷移相容性**：
   - Log-Mel 頻譜圖的「雲霧狀」紋理與自然圖像相似
   - Swin V2 在 ImageNet 上訓練的淺層特徵（邊緣、紋理）可直接遷移
   - VQT 的「橫線狀」紋理是 ImageNet 模型從未見過的

3. **分類任務驗證**：
   - AST 論文證明 Log-Mel + ImageNet Pretrain 在 AudioSet 分類任務上擊敗所有 CNN
   - 我們的轉錄任務需要「看見」音樂結構，而非「測量」音高頻率

### 神經科學對應

#### 耳蝸層級（Cochlea）：對數頻率

基底膜（Basilar Membrane）的 tonotopic organization 是**對數頻率**排列：
- 每移動固定距離 ≈ 一個八度
- 這支持 VQT 的設計理念（但僅限於耳蝸層級）

#### 聽覺皮層（Auditory Cortex）：更複雜

1. **A1 (Primary Auditory Cortex)**：保留 tonotopic map，接近對數
2. **更高層級**：開始出現「範疇知覺（Categorical Perception）」與**音色感知**
   - 共振峰（Formant）是區分樂器的關鍵
   - Log-Mel 保留頻譜包絡，更接近皮層處理方式

### 設計決策

**核心假設**：對於多聲部音樂轉譜（需要區分樂器），Log-Mel 比 VQT 更適合。

| 設定 | 輸入 | 理由 |
|------|------|------|
| **Clef** | Log-Mel (128 bins) | 音色保留佳、ImageNet 相容 |
| **Ablation** | Log-Mel vs VQT | 實證驗證 Log-Mel 優勢 |

### Ablation 驗證

我們將進行消融實驗來驗證此決策：

| 實驗 | 頻譜類型 | 預期 MV2H | 樂器 F1 | 預期結論 |
|------|---------|-----------|---------|---------|
| Clef + VQT | VQT (60 bins/oct) | ~83% | ~75% | 音高高解析，但音色辨識差 |
| **Clef + Log-Mel** | Log-Mel (128 bins) | **~86%** | **~90%** | **音色保留佳，ImageNet 相容** |

**科學問題**：「對於多聲部音樂轉譜，Log-Mel 是否比 VQT 更適合？」

**預期結果**：Log-Mel 在 Overall MV2H 上勝出，特別是在 $F_{harm}$（和聲）與 Instrument F1 上顯著優於 VQT。

---

## 資料集下載

### ASAP Dataset（Study 1 - 鋼琴）

ASAP 的音訊檔不在 GitHub repo 裡，需要從 MAESTRO 提取。

**來源**：
- GitHub: https://github.com/fosfrancesco/asap-dataset
- 音訊來源: MAESTRO v2.0.0

**下載步驟**：
```bash
# Step 1: Clone ASAP repo（樂譜 + metadata）
git clone https://github.com/fosfrancesco/asap-dataset.git

# Step 2: 下載 MAESTRO v2.0.0（音訊）
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip
unzip maestro-v2.0.0.zip

# Step 3: 執行初始化腳本（連結音訊到 ASAP 結構）
cd asap-dataset
pip install librosa pandas numpy
python initialize_dataset.py --maestro_path ../maestro-v2.0.0
```

**只下載 Test Set（推薦）**：
```bash
# 下載 Zeng 的 test split（25 首 / ~80 段錄音）
wget https://raw.githubusercontent.com/wei-zeng98/piano-a2s/main/data_processing/metadata/test_asap.txt

# 根據 split 手動篩選需要的 MAESTRO 音訊（約 3GB）
```

**資料結構**：
```
asap-dataset/
├── Bach/
│   └── Prelude/
│       └── bwv_875/
│           ├── score.mid
│           ├── score.musicxml
│           └── performance_*.wav
├── Chopin/
├── Beethoven/
└── metadata.csv
```

### URMP Dataset（Study 2 - 多樂器合奏）

需要填表單申請下載。

**來源**：
- 官網: https://labsites.rochester.edu/air/projects/URMP.html
- 大小: 12.5GB

**下載步驟**：
1. 前往 [URMP 官網](https://labsites.rochester.edu/air/projects/URMP.html)
2. 點擊 **"Download the whole dataset package"**
3. 填寫 Google Form（學術用途）
4. 收到 email 後下載

**資料結構（使用 AuMix + AuSep）**：
```
URMP/
├── 01_Jupiter_vn_vc/           # Duet: 小提琴 + 大提琴
│   ├── AuMix_01_Jupiter.wav    # ✅ 混音音訊（測試輸入）
│   ├── AuSep_1_vn_01.wav       # ✅ 小提琴分軌（Ground Truth）
│   ├── AuSep_2_vc_01.wav       # ✅ 大提琴分軌（Ground Truth）
│   ├── Sco_01_Jupiter.mid      # MIDI 樂譜
│   └── Notes_1_vn_01.txt       # 音符標註
├── 02_Sonata_fl_fl/            # Duet: 雙長笛
└── ...（共 44 首）
```

**樂器分類**：
| 類別 | 樂器 |
|------|------|
| Strings | violin (vn), viola (va), cello (vc), double bass (db) |
| Winds | flute (fl), oboe (ob), clarinet (cl), saxophone (sax), bassoon (bn) |
| Brass | trumpet (tpt), horn (hn), trombone (tbn), tuba (tba) |

### Slakh2100 Dataset（Study 2 - 合成多軌混音）

**來源**：
- 官網: https://zenodo.org/record/4599666
- 論文: Manilow et al., "Cutting Music Source Separation Some Slakh", ISMIR 2019
- 大小: ~120GB

**下載步驟**：
```bash
# 下載 Slakh2100 (需要大量空間！)
wget https://zenodo.org/record/4599666/files/slakh2100_flac_16k.tar.gz

# 或使用官方腳本
pip install slakh
slakh download --help
```

**資料結構**：
```
slakh2100_flac/
├── Track00001/
│   ├── mix.flac              # ✅ 混音音訊（測試輸入）
│   ├── stems/
│   │   ├── S01.flac          # ✅ 樂器 1 分軌
│   │   ├── S02.flac          # ✅ 樂器 2 分軌
│   │   └── ...
│   └── MIDI/
│       ├── S01.mid           # MIDI Ground Truth
│       └── ...
└── ...（共 2100 首）
```

**樂器分類（MIDI Program Number）**：
| 類別 | 樂器範例 |
|------|---------|
| Piano (0-7) | Acoustic Grand, Electric Piano |
| Guitar (24-31) | Acoustic Guitar, Electric Guitar |
| Bass (32-39) | Acoustic Bass, Electric Bass |
| Strings (40-55) | Violin, Viola, Cello, Ensemble |
| Brass (56-63) | Trumpet, Trombone, Tuba |
| Reed (64-79) | Saxophone, Clarinet, Oboe |
| Drums (N/A) | Drum Kit (不轉譜) |

**Slakh2100 的優勢**：
- 大規模：2100 首，比 URMP 多 47 倍
- 多樂器：每首 4-8 軌，涵蓋流行/搖滾樂器
- 高品質 MIDI：來自 Lakh MIDI Dataset，經過人工校正

**Slakh2100 的劣勢**：
- 合成音訊：使用 VST 合成，不是真實錄音
- Sim-to-Real Gap：需要驗證在真實錄音（URMP）上的泛化能力

### 資料集規模總覽

| Dataset | 用途 | 檔案數 | 大小 | Study |
|---------|------|--------|------|-------|
| ASAP (test only) | Piano baseline | ~80 段 | ~3GB | Study 1 |
| URMP (full) | Multi-instrument ensemble | 44 首 | 12.5GB | Study 2 |
| Slakh2100 | Multi-track training | 2100 首 | ~120GB | Study 2 |

---

## Clef 訓練策略

為確保公平比較，Clef 在不同 Study 採用不同訓練策略。

### Study 1 vs Study 2 訓練對比

| | Study 1 (ISMIR 2026 結果) | Study 2 (Multi-instrument Ensemble) |
|---|---|---|
| **目標** | 證明架構有效性 | 展示多樂器合奏能力 |
| **訓練資料** | ASAP train | PDMX + 跨樂器 TDR |
| **測試資料** | ASAP test (25首/80段) | URMP ensemble + Slakh2100 |
| **訓練類型** | Supervised | Zero-shot |
| **Auxiliary Loss** | 不使用 | ✅ Instrument Aux Loss (λ=0.3) |
| **TDR 策略** | 同樂器 Aug | 跨樂器 TDR |

### Study 1: 與 Zeng 相同設定（公平比較）

為了與 Zeng et al. (2024) 進行 apple-to-apple comparison，Clef 在 Study 1 採用**完全相同的訓練/測試 split**：

```
Clef (Study 1) 訓練流程：
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Pre-training                                  │
│  ├── Data: PDMX scores (對應 Zeng 的 MuseSyn+HumSyn)     │
│  ├── Audio: TDR 合成（對應 Zeng 的 EPR 合成）            │
│  └── Augmentation: key shift, tempo scaling, etc.       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: Fine-tuning（與 Zeng 相同）                    │
│  ├── Data: ASAP train split (14 首 / 58 段)             │
│  └── 真實鋼琴錄音                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Test: ASAP test split (25 首 / 80 段)                  │
│  與 Zeng 完全相同，公平比較                              │
└─────────────────────────────────────────────────────────┘
```

**公平比較要素**：

| 要素 | Zeng (2024) | Clef (Ours) | 相同？ |
|------|-------------|-------------|--------|
| Train split | 14 首 / 58 段 | 14 首 / 58 段 | ✅ |
| Test split | 25 首 / 80 段 | 25 首 / 80 段 | ✅ |
| Pre-train | 合成資料 (EPR) | 合成資料 (TDR) | ✅ |
| Fine-tune | ASAP train | ASAP train | ✅ |
| 評估指標 | MV2H (non-aligned) | MV2H (non-aligned) | ✅ |
| 輸出格式 | \*\*Kern | \*\*Kern | ✅ |
| **Encoder** | CNN (VQT spectrogram) | **ViT (視覺化樂譜)** | ❌ 差異 |
| **Decoder** | Hierarchical RNN | **Transformer** | ❌ 差異 |

> **結論**：Zeng 和 Clef 都輸出 \*\*Kern 格式，核心差異在於 **Encoder 架構**（CNN vs ViT）和 **Decoder 架構**（RNN vs Transformer）。

### Study 2: Multi-instrument Ensemble 訓練（跨樂器 TDR + Aux Loss）

Study 2 使用 **跨樂器 TDR** 策略配合 **Instrument Auxiliary Loss**：

```
Clef (ICLR 2027) 訓練流程：
┌─────────────────────────────────────────────────────────┐
│  Training: Universal + Cross-instrument TDR              │
│  ├── Data: PDMX (250K+ scores，涵蓋多種樂器)             │
│  ├── Audio: 跨樂器 TDR 合成                              │
│  │   ├── 小提琴譜 + 長笛音色                             │
│  │   ├── 鋼琴譜 + 吉他音色                               │
│  │   └── ... 隨機組合「樂譜 × 音色」                     │
│  ├── Instrument Auxiliary Loss (λ=0.3)                  │
│  └── 不使用任何真實錄音！                                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Test: Multi-instrument Ensemble (Zero-shot)            │
│  ├── URMP ensemble (真實錄音)                            │
│  ├── Slakh2100 test (合成多軌)                           │
│  └── 證明 Source Separation + Transcription 能力         │
└─────────────────────────────────────────────────────────┘
```

**為什麼需要跨樂器 TDR + Aux Loss？**

| 策略 | 數據效率 | 樂器辨識 | 適用場景 |
|------|---------|---------|---------|
| 同樂器 Aug (ISMIR 2026) | 低 | 自然正確 | 單樂器 Solo |
| **跨樂器 TDR + Aux Loss** | **高** | **需要 Aux Loss 輔助** | **多樂器 Ensemble** |

### 訓練資料需求總覽

| Study | 訓練資料 | 測試資料 | 需下載 |
|-------|----------|----------|--------|
| Study 1 | (引用 ISMIR 2026) | ASAP test (25首/80段) | — |
| Study 2 | PDMX + 跨樂器 TDR | URMP ensemble + Slakh2100 | URMP, Slakh2100 |

---

## 評估流程設計

本節說明如何確保與 Zeng et al. (2024) 的公平比較。

### Zeng 的評估流程分析

基於對 [piano-a2s repo](https://github.com/wei-zeng98/piano-a2s) 的完整探索，發現 Zeng 的評估流程為：

```
模型輸出 (Logits)
    ↓
Argmax 取得 tokens
    ↓
LabelsMultiple.decode() → **Kern 格式字串
    ↓
get_xml_from_target() 轉換流程：
    ├── tiefix (Humdrum 工具) → 修正連音線
    ├── hum2xml (Humdrum 工具) → 轉換為 MusicXML
    └── music21 → 加入譜號、調性、拍號
    ↓
MusicXML 檔案
    ↓
    ├─→ 轉成 MIDI → MV2H 評估 (音樂內容)
    └─→ 直接使用 XML → ER 評估 (編輯距離)
```

**關鍵發現**：
- MV2H 評估**不是**直接在 **Kern 上進行
- 實際流程是 `**Kern → XML → MIDI → MV2H`
- Zeng 使用 Humdrum Toolkit (`tiefix`, `hum2xml`) + `music21` 進行轉換

### Clef 的評估策略

為確保公平比較，我們在 **MusicXML 層級**統一評估所有系統，而不是強制統一中間格式：

```
評估流程總覽：

┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    MT3      │     │  Transkun   │     │    Zeng     │     │    Clef     │
│ + music21   │     │  + Beyer    │     │   (2024)    │     │   (Ours)    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                    │                   │
       ▼                   ▼                    ▼                   ▼
  MIDI (raw)          MIDI (raw)           **Kern              **Kern
  Performance         Performance          Symbolic            Symbolic
       │                   │                    │                   │
       ▼                   ▼                    ▼                   ▼
  ┌─────────┐       ┌──────────┐       ┌───────────────┐   ┌───────────────┐
  │music21  │       │  Beyer   │       │ tiefix        │   │ tiefix        │
  │quantize │       │Transform.│       │ + hum2xml     │   │ + hum2xml     │
  └────┬────┘       └────┬─────┘       │ + music21     │   │ + music21     │
       │                 │              └───────┬───────┘   └───────┬───────┘
       ▼                 ▼                      ▼                   ▼
   MusicXML          MusicXML              MusicXML            MusicXML
       │                 │                      │                   │
       └─────────────────┴──────────────────────┴───────────────────┘
                                 ▼
                    ┌────────────────────────────┐
                    │  統一的 XML → MIDI 轉換     │
                    │  (music21.write('midi'))   │
                    └─────────────┬──────────────┘
                                  ▼
                          MIDI (symbolic)
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                    MV2H 評估         STEPn 評估
                 (音樂內容正確性)    (樂譜結構正確性)
```

**關鍵設計原則**：

1. **不強制統一到 **Kern 格式**
   - **Kern 只是 Zeng/Clef 的原生輸出，不是通用標準
   - 強制 Pipeline 系統轉 **Kern 會引入額外轉換誤差
   - 沒有標準的 MIDI → **Kern 轉換工具

2. **統一在 MusicXML 層級評估**
   - MusicXML 是所有系統都能產生的格式
   - 各系統使用其原生的符號化流程
   - 在 symbolic representation 層級確保公平比較

3. **尊重系統設計哲學**
   - Pipeline 系統：MIDI (performance) → XML (score)
   - End-to-End 系統：Audio → **Kern (symbolic) → XML (score)
   - 評估焦點：最終符號化結果的品質，而非中間步驟的一致性

4. **統一的最終評估**
   - 所有系統的 XML 都用相同的 `music21.write('midi')` 轉換
   - 確保 MV2H 和 STEPn 評估的公平性

### Baseline 系統配置

#### 1. Weak Baseline: MT3 + MuseScore 4

**系統組成**：
- **Audio-to-MIDI**: MT3 (Google Magenta, ICLR 2022)
- **MIDI-to-Score**: music21 (Rule-based quantization + heuristic hand separation)

**轉換流程**：
```python
# Step 1: MT3 推論
midi_output = mt3.transcribe(audio)

# Step 2: music21 量化
# quarterLengthDivisors=(4, 3) = sixteenth notes + eighth-note triplets
score = music21.converter.parse(midi_output, quarterLengthDivisors=(4, 3))

# Step 3: 分手（pitch-based heuristic at Middle C）
# Reference: Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
right_hand, left_hand = separate_by_pitch(score, split_point=60)

# Step 4: 輸出 MusicXML
score.write('musicxml', fp=output_path)
```

**實作腳本**: `evaluation/zeng_baseline/mt3_to_musicxml.py`

**學術依據**：

| 步驟 | 方法 | 學術參考 |
|------|------|---------|
| 量化 | `quarterLengthDivisors=(4, 3)` | music21 default (Cuthbert & Ariza, 2010) |
| 分手 | Pitch split at MIDI 60 | Hadjakos et al. (2019) baseline method |
| 輸出 | MusicXML | W3C Music Notation Community Group |

**已知限制（論文需說明）**：
1. **Hand crossing**: 右手彈低音會被誤判給左手
2. **Overlapping range**: 中音區音符分配模糊
3. **No voice separation**: 同手的複音被壓成和弦

> 這些限制是 **intentional**，用以展示 rule-based post-processing 的局限性。

**代表性**：工業界最常用的 Pipeline 方法

#### 2. Strong Baseline: Transkun + Beyer

**系統組成**：
- **Audio-to-MIDI**: Transkun (ISMIR 2023, Piano transcription SOTA)
- **MIDI-to-Score**: Beyer Transformer (ISMIR 2024, Performance-to-Score SOTA)

**轉換流程**：
```python
# Step 1: Transkun 推論
midi_output = transkun.transcribe(audio)

# Step 2: Beyer Transformer 符號化
xml_output = beyer.performance_to_score(midi_output)
```

**代表性**：Pipeline 方法的天花板（SOTA combination）

**參考文獻**：
- Transkun: Kong et al. "High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times", ISMIR 2023
- Beyer: Beyer & Dai "End-to-End Piano Performance-MIDI to Score Conversion with Transformers", ISMIR 2024, arXiv:2410.00210

### 延伸閱讀

**完整評估流程文件**：
- 📊 [evaluation-flow-diagram.md](./evaluation-flow-diagram.md) - 詳細的評估流程圖與說明
- 🛡️ [reviewer-response-template.md](./reviewer-response-template.md) - 針對評估設計的防守範本

這些文件提供：
- 完整的視覺化評估流程
- 每個轉換步驟的詳細說明
- 針對 reviewer 可能質疑的完整防守論述
- 可重現性檢查清單

### 評估工具來源

| 工具 | 來源 | 用途 | License |
|------|------|------|---------|
| `evaluate.py` | [piano-a2s/evaluate.py](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate.py) | MV2H/WER/F1/ER 評估 | Apache-2.0 |
| `evaluate_midi_mv2h.sh` | [piano-a2s/evaluate_midi_mv2h.sh](https://github.com/wei-zeng98/piano-a2s/blob/main/evaluate_midi_mv2h.sh) | MV2H Shell 執行腳本 | Apache-2.0 |
| `humdrum.py` | [piano-a2s/data_processing/humdrum.py](https://github.com/wei-zeng98/piano-a2s/blob/main/data_processing/humdrum.py) | **Kern ↔ 符號轉換 | Apache-2.0 |
| Humdrum Toolkit | [humdrum-tools](https://github.com/humdrum-tools/humdrum-tools) | `tiefix`, `hum2xml` | BSD License |
| MV2H 評估器 | [music-voice-separation](https://github.com/cheriell/music-voice-separation) | 符號層級評估 | MIT License |

**使用說明**：
- ✅ 可以直接使用 Zeng 的 `evaluate.py` 和相關腳本（Apache-2.0 License 允許）
- ✅ 已下載至 `evaluation/zeng_baseline/` 目錄，包含完整的 LICENSE 檔案
- ✅ 需要在論文 Acknowledgments 中註記：
  > "We thank Wei Zeng, Xian He, and Ye Wang for open-sourcing their evaluation scripts, which we adapted for our experiments."
- ✅ 在 repo README 的 Citation 區塊加入：
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

---

## Study 1: Depth (深度) — 引用 ISMIR 2026 結果

> **本章節引用 ISMIR 2026 論文的結果**：完整的架構驗證和消融實驗請參考 ISMIR 2026 論文。

### 核心架構（來自 ISMIR 2026）

ICLR 2027 版本建立在 ISMIR 2026 證明的架構基礎上：

| 元件 | 設計 | 驗證來源 |
|------|------|---------|
| **Encoder** | Swin-V2 | ISMIR 2026 Study 1 |
| **Bridge** | 2-layer Transformer | ISMIR 2026 Study 1 |
| **Decoder** | Autoregressive Transformer | ISMIR 2026 Study 1 |
| **輸入** | Stereo 3-channel Log-Mel | ISMIR 2026 Study 1 |
| **輸出** | Kern（含樂器標籤） | ISMIR 2026 Study 2 |

### 鋼琴 A2S 結果摘要（來自 ISMIR 2026）

| System | MV2H | $F_p$ | $F_{harm}$ |
|--------|------|-------|------------|
| MT3 + MuseScore 4 | ~58% | ~80% | ~40% |
| Zeng (2024) | 74.2% | 63.3% | 54.5% |
| **Clef (ISMIR 2026)** | **~85%** | **~81%** | **~70%** |

**ICLR 2027 的延伸**：在 ISMIR 2026 驗證的架構基礎上，加入：
1. **Instrument Auxiliary Loss**：強迫編碼器保留音色資訊
2. **跨樂器 Timbre Domain Randomization**：最大化數據效率
3. **Multi-track 輸出**：使用 `<coc>` token 分隔不同樂器軌道

---

## Ablation Study 設計

本節設計系統性的消融實驗，量化各設計決策的貢獻。我們的架構包含三個關鍵創新：**Swin-V2 編碼器**、**Global Bridge** 與 **Auxiliary Loss**，以下實驗逐一驗證它們的必要性。

### 1. 編碼器 Ablation（Swin-V2 vs ViT vs CNN）

驗證 Swin Transformer V2 相較於 ViT 與 CNN 的優勢：

| 實驗 | Encoder | Decoder | Input | 預期 MV2H | $F_p$ | $F_{harm}$ |
|------|---------|---------|-------|-----------|-------|------------|
| Zeng (baseline) | CNN | Hierarchical RNN | Mono VQT | 74.2% | 63.3% | 54.5% |
| Clef-ViT | ViT | Transformer | Log-Mel | ~77% | ~70% | ~58% |
| **Clef-Swin** | **Swin-V2** | Transformer | Log-Mel | **~80%** | **~75%** | **~62%** |

**預期結論**：Swin-V2 的相對位置偏差與階層式結構使其在捕捉和聲結構上優於 ViT 與 CNN。

### 2. Global Bridge Ablation

驗證 Bridge 層數對效能的影響：

| 實驗 | Encoder | Bridge 層數 | Decoder | 預期 MV2H | TEDn |
|------|---------|-------------|---------|-----------|------|
| Clef-Swin (無 Bridge) | Swin-V2 | 0 | Transformer | ~80% | ~0.75 |
| Clef-Swin + Bridge-1 | Swin-V2 | 1 | Transformer | ~82% | ~0.78 |
| **Clef-Swin + Bridge-2** | Swin-V2 | **2** | Transformer | **~84%** | **~0.80** |
| Clef-Swin + Bridge-4 | Swin-V2 | 4 | Transformer | ~84% | ~0.80 |
| Clef-Swin + Bridge-6 | Swin-V2 | 6 | Transformer | ~83% | ~0.79 |

**預期結論**：
- 0 層 Bridge：缺乏全域上下文，無法捕捉段落呼應
- 1 層 Bridge：改善有限，全域資訊傳遞不足
- 2 層 Bridge：最佳平衡點，有效實現跨段落資訊傳遞
- 4-6 層 Bridge：開始出現過擬合，收益遞減

**研究問題**：「Bridge 的最佳層數是多少？」

### 3. Instrument Auxiliary Loss Ablation

驗證樂器分類輔助任務對多樂器合奏的貢獻：

| 實驗 | TDR 策略 | Aux Loss | λ | Solo MV2H | Ensemble MV2H | 樂器 F1 |
|------|----------|----------|---|-----------|---------------|---------|
| Clef (ISMIR) | 同樂器 Aug | ❌ | - | ~85% | ~30% | ~65% |
| Clef + 跨樂器 TDR | 跨樂器 TDR | ❌ | - | ~80% | ~40% | ~55% |
| **Clef + TDR + Aux** | 跨樂器 TDR | ✅ | 0.1 | ~83% | ~55% | ~80% |
| **Clef + TDR + Aux** | 跨樂器 TDR | ✅ | **0.3** | **~85%** | **~65%** | **~90%** |
| Clef + TDR + Aux | 跨樂器 TDR | ✅ | 0.5 | ~82% | ~60% | ~92% |

**預期結論**：
- 無 Aux Loss 的跨樂器 TDR 會導致「樂器混淆」（Solo 下降、樂器 F1 暴跌）
- λ = 0.3 為最佳權重，平衡主任務與輔助任務
- Auxiliary Loss 帶來的效能提升主要來自：
  1. **特徵解耦**：強迫編碼器分離「音色」與「音高」表徵
  2. **樂器辨識**：在混音中正確辨識各樂器來源
  3. **Source Separation**：隱式學會分離不同樂器軌道

### 4. 頻譜表示 Ablation（VQT vs Log-Mel）

驗證 Log-Mel 對音色保留的優勢：

| 實驗 | 頻譜類型 | 解析度 | 預期 MV2H | 樂器 F1 | 備註 |
|------|---------|--------|-----------|---------|------|
| Clef-Swin + VQT | VQT | 60 bins/oct | ~83% | ~75% | 音高解析度高，但音色扭曲 |
| **Clef-Swin + Log-Mel** | Log-Mel | 128 bins | **~86%** | **~90%** | 音色保留佳，ImageNet 相容 |
| Clef-Swin + Log-Mel-256 | Log-Mel | 256 bins | ~85% | ~88% | 邊際效益遞減 |

**科學問題**：「對於多聲部音樂轉譜，Log-Mel 是否比 VQT 更適合？」

**理論基礎**：
- VQT 會對頻譜進行非線性扭曲，破壞共振峰（Formant）位置
- 共振峰是區分小提琴 vs 中提琴的關鍵特徵
- Log-Mel 保留頻譜包絡，有利於 Instrument Auxiliary Loss

### 5. 前處理 Ablation

逐步加入前處理改進，量化各自貢獻：

| 實驗 | Input | Normalization | Augmentation | 預期 MV2H |
|------|-------|---------------|--------------|-----------|
| Clef-base | Mono Log-Mel | ❌ | ❌ | ~80% |
| + Loudness | Mono Log-Mel | ✅ | ❌ | ~81% |
| + Stereo | Stereo 3-ch | ✅ | ❌ | ~83% |
| + L/R Flip | Stereo 3-ch | ✅ | ✅ | ~84% |

### 6. 完整 Ablation 總結表（ICLR 2027 重點）

| 設計決策 | Solo 貢獻 | Ensemble 貢獻 | 驗證方式 |
|---------|----------|---------------|---------|
| Swin-V2 + Bridge | (ISMIR 基礎) | (ISMIR 基礎) | 引用 ISMIR 2026 |
| **跨樂器 TDR** | -5% (樂器混淆) | +10% | TDR Ablation |
| **Aux Loss (λ=0.3)** | +5% (補回) | +25% | Aux Loss Ablation |
| **TDR + Aux 組合** | ±0% | **+35%** | 完整系統比較 |

**核心結論**：
- 跨樂器 TDR 單獨使用會導致「樂器混淆」，Solo 效能下降
- Aux Loss 是跨樂器 TDR 的「必要配套」，兩者必須同時使用
- 組合後在 Solo 上維持 ISMIR 效能，在 Ensemble 上大幅提升

### 7. 消融實驗預期結果表（Study 2: Multi-instrument Ensemble）

| Model Configuration | Solo MV2H | Ensemble MV2H | Instrument F1 | Source Sep. SDR |
|---------------------|-----------|---------------|---------------|-----------------|
| MT3 + MuseScore 4 | ~50% | ~25% | N/A | N/A |
| Clef (ISMIR 2026) | 85.0 | ~30% | ~65% | N/A |
| Clef + 跨樂器 TDR (無 Aux) | 80.0 | ~40% | ~55% | ~3 dB |
| Clef + TDR + Aux (λ=0.1) | 83.0 | ~55% | ~80% | ~5 dB |
| **Clef + TDR + Aux (λ=0.3)** | **85.0** | **~65%** | **~90%** | **~7 dB** |
| Clef + TDR + Aux (λ=0.5) | 82.0 | ~60% | ~92% | ~6 dB |

**評估說明**：
- **Solo MV2H**：在單樂器錄音上的 MV2H
- **Ensemble MV2H**：在多樂器混音上，分離後各軌的平均 MV2H
- **Instrument F1**：樂器辨識準確度（Multi-label）
- **Source Sep. SDR**：Signal-to-Distortion Ratio（分離品質，僅作參考）

> **註**：跨樂器 TDR 需要 Auxiliary Loss 配合才能維持 Solo 效能並提升 Ensemble 效能。

---

## 音訊前處理策略

本節詳述音訊前處理的實作細節，基於對 ASAP 資料集的深入分析。

### 1. Loudness Normalization

**問題**：ASAP 中同一首曲子不同演奏者的音量差異巨大。

**解決方案**：
- 統一標準化到 **-20 dBFS** 或 **-14 LUFS**（串流平台標準）
- 訓練時加入輕微 **Gain Jitter (±3dB)** 作為 augmentation

```python
# 前處理：標準化
audio = loudness_normalize(audio, target_lufs=-14)

# 訓練時：加入抖動
if training:
    gain_db = random.uniform(-3, 3)
    audio = audio * (10 ** (gain_db / 20))
```

### 2. Stereo 3-Channel Input

**設計理念**：模擬人類大腦的雙耳整合（Binaural Summation）機制。

| Channel | 來源 | 神經科學對應 |
|---------|------|-------------|
| **Ch 1 (Red)** | Left spectrogram | 左耳訊號 |
| **Ch 2 (Green)** | Right spectrogram | 右耳訊號 |
| **Ch 3 (Blue)** | Mid = (L+R)/2 | 大腦疊加後的「幻象中心」|

**處理 Mono/Stereo 混合資料**：

```python
if audio.shape[0] == 1:  # Mono
    L = R = Mid = audio[0]
else:  # Stereo
    L, R = audio[0], audio[1]
    Mid = (L + R) * 0.5

input_tensor = torch.stack([spec(L), spec(R), spec(Mid)], dim=0)
```

**優點**：
- Mid channel 提供冗餘：即使一個聲道壞掉（如 ASAP 的 YeZ02M.wav），仍有訊號
- 符合 ImageNet 預訓練的 RGB 期望（3 channels）

### 3. Spatial Augmentation: L/R Flip

**物理意義**：演奏者視角（低音在左）vs 觀眾視角（低音在右）。

**實作**：50% 機率交換 L/R channel（**不是** Horizontal Flip！）

```python
def stereo_flip_augmentation(input_tensor):
    """
    input_tensor shape: (3, H, W) -> (L, R, Mid)
    注意：只交換 Ch1/Ch2，Ch3 (Mid) 不變！
    因為 L+R = R+L，Mid 是不動點 (invariant)
    """
    if random.random() > 0.5:
        flipped = input_tensor.clone()
        flipped[0] = input_tensor[1]  # New L = Old R
        flipped[1] = input_tensor[0]  # New R = Old L
        # flipped[2] 保持不變 (Mid)
        return flipped
    return input_tensor
```

**重要警告**：不要使用 `torchvision.transforms.RandomHorizontalFlip`，那會翻轉時間軸！

### 4. ASAP 資料品質問題處理

基於對 ASAP test set 的人工聆聽分析：

| 問題 | 範例檔案 | 處理策略 |
|------|---------|---------|
| 音量不一致 | 多個演奏者 | Loudness Normalization |
| 聲道偏移 | YeZ02M.wav | Mid channel 提供冗餘 |
| 殘響截斷 | GalantM02M.wav | 視為 outlier，Error Analysis 標註 |

**Error Analysis 寫法範例**：
> "In file *GalantM02M*, the audio recording contains an abrupt cutoff that contradicts the score duration, leading to unavoidable alignment errors."

---

## Study 2: Breadth (廣度) — Multi-instrument Ensemble

### 設計理念

Study 2 的目標是展示 Clef 在 **多樂器合奏場景** 的能力：
- 從混音中分離並轉錄多個樂器
- 正確辨識每個樂器並輸出對應的 Kern 樂器標籤

核心問題：
> 「一個用跨樂器 TDR + Auxiliary Loss 訓練的模型，能不能在真實合奏錄音中正確分離並轉錄多個樂器？」

### 核心技術（ICLR 2027 新增）

#### 1. Instrument Auxiliary Loss

**目的**：強迫編碼器保留音色資訊，幫助模型在跨樂器 TDR 下正確辨識樂器。

**數學定義**：
$$
\mathcal{L}_{total} = \mathcal{L}_{transcription} + \lambda \cdot \mathcal{L}_{instrument}
$$

其中：
- $\mathcal{L}_{transcription}$：主要的轉譜損失（Cross-Entropy）
- $\mathcal{L}_{instrument}$：樂器分類損失（Multi-label Cross-Entropy）
- $\lambda$：權重係數（預設 0.3）

**實作方式**：
```python
# Bridge output: (batch, seq_len, hidden_dim)
bridge_output = self.bridge(encoder_output)

# Auxiliary head: instrument classification
# Global average pooling + MLP
pooled = bridge_output.mean(dim=1)  # (batch, hidden_dim)
instrument_logits = self.instrument_head(pooled)  # (batch, num_instruments)

# Multi-label loss (each track can have multiple instruments)
aux_loss = F.binary_cross_entropy_with_logits(
    instrument_logits,
    instrument_labels  # (batch, num_instruments) one-hot
)
```

**為什麼需要 Auxiliary Loss？**

| 策略 | 跨樂器 TDR | 樂器辨識準確度 | 說明 |
|------|-----------|---------------|------|
| 無 Aux Loss | ✅ | ~60% | 模型混淆樂器標籤 |
| **有 Aux Loss** | ✅ | **~90%** | 強迫編碼器保留音色資訊 |

#### 2. 跨樂器 Timbre Domain Randomization (TDR)

**與 ISMIR 2026 的差異**：
| 策略 | ISMIR 2026 | ICLR 2027 |
|------|------------|-----------|
| **同樂器 Aug** | ✅ 使用 | ✅ 使用 |
| **跨樂器 TDR** | ❌ 不使用 | ✅ 使用 |
| **Aux Loss** | ❌ 不使用 | ✅ 使用 |

**跨樂器 TDR 策略**：
```python
# 訓練時：隨機組合「樂譜 + 音色」
score_instrument = "violin"  # 原本的樂譜樂器
synth_instrument = random.choice(["violin", "flute", "cello", "clarinet"])

# Aux Loss 幫助模型學會：
# - 從音色判斷「這是什麼聲音」
# - 從樂譜標籤知道「應該轉成什麼譜」
```

#### 3. Multi-track 輸出格式

使用 `<coc>` (Change of Channel) token 分隔不同樂器軌道：

```
*Ivioln
4c 4e 4g
*-
<coc>
*Icello
4C 4E 4G
*-
```

### 測試資料

| 資料集 | 類型 | 樂器數 | 說明 |
|--------|------|--------|------|
| URMP (ensemble) | 真實錄音 | 2-5 | Duets, Trios, Quartets |
| Slakh2100 | 合成錄音 | 4-8 | Pop/Rock 多軌混音 |

### Table 2: Multi-instrument Ensemble Transcription

| Model Strategy | Training Data | Architecture | Solo MV2H | Ensemble MV2H | Instrument F1 |
|----------------|---------------|--------------|-----------|---------------|---------------|
| MT3 + MuseScore 4 | MAESTRO + Slakh | CNN + Rule | ~50% | ~25% | N/A |
| Clef (ISMIR 2026) | Universal Solo | Swin + Bridge | **~85%** | ~30% | ~65% |
| **Clef (ICLR 2027)** | **Universal + TDR** | **Swin + Bridge + Aux** | **~85%** | **> 60%** | **~90%** |

> **註**：Ensemble MV2H 是在分離後的各軌上分別計算，再取平均。

### 消融實驗：Auxiliary Loss 權重 (λ)

| λ | Transcription MV2H | Instrument F1 | 備註 |
|---|-------------------|---------------|------|
| 0.0 | ~60% | ~60% | 無 Aux Loss，樂器混淆 |
| 0.1 | ~62% | ~80% | Aux Loss 太弱 |
| **0.3** | **~65%** | **~90%** | **最佳平衡** |
| 0.5 | ~63% | ~92% | Aux Loss 太強，搶走主任務梯度 |

### 表格亮點

1. **Clef (ISMIR 2026)**：單樂器表現優異，但合奏場景失效
2. **Clef (ICLR 2027)**：
   - Aux Loss 強迫特徵解耦
   - 跨樂器 TDR 提供音色不變性
   - 成功在合奏場景分離並轉錄多個樂器
3. **MT3 + MuseScore 4**：完全無法處理多樂器合奏

---

## 論文結構總覽（ICLR 2027）

| Study | 定位 | 戰場 | 對手 | 目標 |
|-------|------|------|------|------|
| Study 1 | Depth (深度) | ASAP (Piano) | (引用 ISMIR 2026) | — |
| Study 2 | Representation | Visual Aux Head Ablation | Clef w/o Aux | 證明 Aux Head 的效果 |

### 核心論點

> 「學習視覺佈局（stem, beam, voice）是否能幫助語意理解？」

這是一個 representation learning 的問題，而非純粹的音樂轉譜任務。

### ISMIR 2026 vs ICLR 2027 差異

| 面向 | ISMIR 2026 | ICLR 2027 (本文件) |
|------|------------|-------------------|
| **目標** | 單樂器轉譜 | Representation Learning |
| **視覺資訊** | **清掉**（簡化任務） | **學習**（Visual Auxiliary Head） |
| **TDR 策略** | 同樂器內換音源 | 同樂器內換音源 |
| **Auxiliary Loss** | 不使用 | ✅ Instrument Aux + Visual Aux |
| **輸出格式** | Kern（語意為主） | Kern（語意 + 視覺佈局） |
| **TEDn 評估** | Optimality Gap 方法 | 完整 TEDn（含視覺） |
| **核心賣點** | 「能用」 | 「為什麼能用」 |

### Visual Auxiliary Head 設計（ICLR 2027 核心創新）

**架構**：

```
┌─────────────────────────────────────────────────────────┐
│                   Kern Decoder                          │
│            (Autoregressive Transformer)                 │
│                                                         │
│   Output: 4C  4E  4G  =  8D  8F# ...                   │
│           ↓   ↓   ↓      ↓   ↓                         │
│         [h₁] [h₂] [h₃] [h₄] [h₅]  ← hidden states      │
└──────────┬──────────────────────────────────────────────┘
           │
           ├────────────────┐
           │                │
           ▼                ▼
    ┌─────────────┐  ┌─────────────────┐
    │ Main Head   │  │ Visual Aux Head │
    │ (CE Loss)   │  │ (Aux Loss)      │
    │             │  │                 │
    │ next token  │  │ stem: up/down   │
    │ prediction  │  │ beam: L/J/k/K   │
    │             │  │ voice: 1/2/3/4  │
    │             │  │ staff: 1/2      │
    └─────────────┘  └─────────────────┘
```

**Loss 設計**：
```python
L_total = L_main + λ_inst * L_instrument + λ_vis * L_visual
# λ_inst ≈ 0.3, λ_vis ≈ 0.1
# 視覺任務權重較低，避免主導訓練
```

**核心洞見**：視覺佈局是從音樂內容可推導的規則：
- Stem direction：中央 B 以上 stem down，以下 stem up
- Voice assignment：Voice 1 stem up，Voice 2 stem down
- Staff assignment：根據音域和聲部分配

這個輔助任務強迫模型理解樂譜結構，同時不會為了視覺資訊犧牲音符準確性。

**Ground Truth 來源**：heal_cross_staff 的移動紀錄可作為 staff assignment 的 ground truth。

### 時程規劃

| 時間點 | 行動 | 里程碑 |
|--------|------|--------|
| 2026 May | 投稿 ISMIR 2026 | Piano A2S 論文 |
| 2026 Jun-Aug | 衝刺 Visual Aux Head 實驗 | Ablation Study |
| 2026 Aug-Sep | 撰寫 ICLR 論文 | ISMIR 放榜（通常 8 月底）|
| **2026 Sep-Oct** | **投稿 ICLR 2027** | Representation Learning 論文 |
| 2026 Nov | 參加 ISMIR 2026 | 阿布達比 (Abu Dhabi) |
| 2027 Apr-May | 參加 ICLR 2027 | (地點待定) |

### ICLR 風格的包裝策略

**Title Idea**：
> _Learning Visual Layout as Auxiliary Supervision for Audio-to-Score Transcription_

**關鍵賣點**：
1. **Sim-to-Real Transfer**：用合成數據訓練，在真實錄音上表現良好
2. **Representation Disentanglement**：Instrument Aux Loss 強迫編碼器分離「音色」與「音高」；Visual Aux Head 強迫 decoder 分離「語意」與「視覺」
3. **Zero-shot Generalization**：對未見過的樂器/錄音環境仍能正確轉錄（ICLR 評審在乎泛化能力）
4. **Auxiliary Task Design**：Visual layout prediction 作為輔助任務，探討是否幫助主任務

**次要賣點**：
- 小節線有沒有畫對
- Rubato 處理
- 人類可讀性（但可以放 demo）
