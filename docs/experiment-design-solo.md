# Experiment Design: Universal Solo Transcription (ISMIR 2026)

本文件描述 Clef 針對 **通用單樂器轉譜 (Universal Solo Transcription)** 的實驗設計。

**目標定位**：為所有音樂家設計的通用單樂器轉譜系統 — 不只是「另一個鋼琴轉譜模型」。

> 「因為我不是鋼琴家，我是曼陀林演奏家... 你能想像鋼琴家在歡呼，然後隔壁的小提琴演奏家、長笛演奏家冷眼說：『喔，阿不就是另外一個鋼琴的轉錄模型？輪到我還久的』」

**核心實驗**：
- **Study 1（深度/Precision）**：鋼琴 A2S — 證明架構的深度能力
- **Study 2（廣度/Breadth）**：Universal Solo — 證明跨樂器的泛化能力

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

### Zeng 的兩個版本：Score vs EPR

Zeng et al. (2024) 的 codebase 支援兩種 MIDI 生成模式（`midi_syn` 參數）：

| 版本 | MIDI 來源 | Velocity | 特徵 |
|------|----------|----------|------|
| **Zeng-score** | verovio 直接轉換 | 90（固定） | 死板、無動態變化 |
| **Zeng-EPR** | VirtuosoNet (15 composers) | 變化的 | 模擬人類演奏表情 |

**EPR (Expressive Performance Rendering)** 的四個隨機化步驟：

1. **Random Key Shift**: ±4 semitones
2. **Random Composer**: 從 15 位虛擬作曲家隨機選擇（僅 EPR）
3. **Random Tempo Scaling**: 0.85-1.15x
4. **Random Soundfont**: 4 種鋼琴音色

> **關鍵差異**：Score 版本的 MIDI velocity 全為 90（verovio 預設），完全沒有動態變化。EPR 版本透過 VirtuosoNet 模擬人類演奏的 onset、duration、velocity、pedal 變化。

### 兩層比較策略 (Two-Tier Comparison)

為了進行**控制變因的公平比較**，我們將實驗分為兩個層級：

| 比較層級 | Clef | vs | Zeng | 控制變因 | 目的 |
|----------|------|-----|------|----------|------|
| **Tier 1: 架構比較** | clef-piano-base | vs | Zeng-score | 都用死板 MIDI（無 EPR） | 證明架構優越性 |
| **Tier 2: SOTA 比較** | clef-piano-full | vs | Zeng-finetune | 都用完整 augmentation | 打敗 SOTA |

### Table 0: 兩層比較總覽

| Model | Training Data | Augmentation | MV2H | 結論 |
|-------|---------------|--------------|------|------|
| **Tier 1: Architecture** | | | | |
| Zeng-score | Syn (Basic) | Transpose + Tempo + SF | TBD | 基準線 |
| **Clef-piano-base** | Syn (Basic) | Tempo + SF | **> Zeng-score** | **架構勝利** |
| | | | | |
| **Tier 2: SOTA** | | | | |
| Zeng-EPR | Syn (EPR) | VirtuosoNet + all | 69.4% (Real) | Pre-train only |
| Zeng-finetune | Syn (EPR) + Real | VirtuosoNet + all | **74.2%** (Real) | 目前 SOTA |
| **Clef-piano-full** | Syn + Real | Custom Rule-based | **> 75.0%** | **全面勝利** |

> **說明**：
> - **Tier 1** 比較的是「架構差異」：Swin + Transformer vs CNN + H-RNN
> - **Tier 2** 比較的是「完整系統」：我們的 augmentation vs VirtuosoNet
> - Clef-piano-base 不使用 EPR/VirtuosoNet，與 Zeng-score 公平比較
> - Clef-piano-full 使用自己設計的 rule-based augmentation（velocity variation、sustain、reverb 等）

### 兩階段訓練 (Two-Stage Training)

**Zeng 原始設定（EPR 版本）**：

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

**Clef-piano-base 設定（Score 版本對照）**：

```
Stage 1: Pre-training (Synthetic Data)
├── Data: MuseSyn (Pop) + HumSyn (Classical/Ragtime)
├── Audio: verovio 直接轉換（無 EPR）
├── MIDI Velocity: 90（固定，無動態）
├── Augmentation:
│   ├── Random tempo scaling (0.85-1.15x)
│   └── Random soundfont (4 種鋼琴)
└── 無 key shift（鋼琴 voicing 考量）

Stage 2: Fine-tuning (Real Recordings)
├── Data: ASAP train split
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
   - 這對跨樂器泛化是毀滅性的打擊（模型難以從音色區分樂器）

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

| 實驗 | 頻譜類型 | 預期 Piano MV2H | 預期跨樂器 MV2H | 預期結論 |
|------|---------|-----------------|-----------------|---------|
| Clef + VQT | VQT (60 bins/oct) | ~83% | ~45% | 音高高解析，但音色辨識差 |
| **Clef + Log-Mel** | Log-Mel (128 bins) | **~85%** | **~58%** | **音色保留佳，ImageNet 相容** |

**科學問題**：「對於跨樂器泛化，Log-Mel 是否比 VQT 更適合？」

**預期結果**：Log-Mel 在跨樂器泛化上顯著優於 VQT，因為它保留了音色特徵（共振峰）。

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

### GAPS Dataset（Study 2 - 吉他）

GAPS (Guitar Audio-to-Score) 是 2024 年發布的高品質古典吉他資料集，包含真實錄音與對齊的 MusicXML 樂譜。

**來源**：
- 論文: "GAPS: A Dataset for Guitar Audio-to-Score Transcription", ISMIR 2024
- 規模: 14 小時 / 200+ 演奏者

**資料結構**：
```
GAPS/
├── audio/              # 真實錄音 (WAV)
├── scores/             # MusicXML 樂譜
└── alignments/         # Audio-score 對齊資訊
```

**為什麼選 GAPS 而不是 GuitarSet？**
- GuitarSet 只有 JAMS 格式（MIDI-like），**沒有 MusicXML**
- GAPS 有完整的 MusicXML 樂譜，可直接用 MV2H 評估

### Bach Violin Dataset（Study 2 - 小提琴）

Bach Solo Violin 作品 BWV 1001-1006 的真實錄音與對齊樂譜。

**來源**：
- 規模: 6.5 小時 / 17 位演奏家
- 曲目: Bach Solo Violin Sonatas & Partitas (BWV 1001-1006)

**資料結構**：
```
BachViolin/
├── audio/              # 17 位演奏家的錄音
├── scores/             # MusicXML 樂譜
└── alignments/         # Performance-score 對齊
```

**特點**：
- 標準曲目，審稿人熟悉
- 單聲部（容易對齊）
- 高品質專業錄音

### GTSinger Dataset（Study 2 - 人聲）

NeurIPS 2024 發布的大規模人聲資料集，包含真實演唱與對齊樂譜。

**來源**：
- 論文: "GTSinger: A Global Multi-Technique Singing Corpus", NeurIPS 2024
- 規模: 80+ 小時 / 20 位歌手 / 9 語言

**資料結構**：
```
GTSinger/
├── audio/              # 真實演唱錄音
├── scores/             # MusicXML 樂譜
└── metadata/           # 語言、技巧標註
```

**特點**：
- 目前最大的有樂譜人聲資料集
- 多語言、多技巧
- 與 MTC-ANN（荷蘭民謠 Kern）互補

### MTC-ANN Dataset（Study 2 - 人聲/民謠，Kern 格式）

Meertens Tune Collections 的標註子集，包含荷蘭田野錄音與 Kern 樂譜。

**來源**：
- 官網: https://www.liederenbank.nl/mtc/
- 規模: 360 首 / 原生 Kern 格式
- 授權: CC BY-NC-SA 3.0

**特點**：
- 真正的田野錄音（1950s-1980s）
- 原生 Kern 格式（不需轉換）
- 單聲部旋律

**限制**：
- 規模較小
- 錄音品質參差（田野錄音）
- 荷蘭語民謠，風格偏離古典

### Evaluation Datasets 總覽

| 資料集 | 樂器 | 規模 | 樂譜格式 | 狀態 |
|--------|------|------|----------|------|
| **ASAP** | 🎹 鋼琴 | 92+ 小時 / 1,068 performances | MusicXML | ✅ 可用 |
| **GAPS** | 🎸 古典吉他 | 14 小時 / 200+ 演奏者 | MusicXML | ✅ 可用 |
| **Bach Violin** | 🎻 小提琴 | 6.5 小時 / 17 位演奏家 | MusicXML | ✅ 可用 |
| **GTSinger** | 🎤 人聲 | 80+ 小時 / 9 語言 | MusicXML | ✅ 可用 |
| **MTC-ANN** | 🎤 人聲（民謠） | 360 首 | Kern | ✅ 可用（規模小） |
| 中提琴 | — | — | — | ❌ 無 |
| 大提琴 | — | — | — | ❌ 無 |
| 長笛/木管 | — | — | — | ❌ 無 |

> **Limitation**: 目前沒有公開可用的 **中提琴、大提琴、長笛** 資料集同時包含真實錄音 + MusicXML/Kern 樂譜。這些樂器的評估留待未來研究。

### 資料集下載總覽

| Dataset | 用途 | 檔案數 | 大小 | Study |
|---------|------|--------|------|-------|
| ASAP (test only) | Piano baseline | ~80 段 | ~3GB | Study 1 & 2 |
| GAPS | Guitar 真實錄音 | ~200 段 | ~5GB | Study 2 |
| Bach Violin | Violin 真實錄音 | ~100 段 | ~2GB | Study 2 |
| GTSinger | Voice 真實錄音 | ~1000 段 | ~10GB | Study 2 |
| MTC-ANN | Voice (民謠) | 360 首 | ~500MB | Study 2 (補充) |

---

## Clef 訓練策略

為確保公平比較，Clef 在不同 Study 採用不同訓練策略。

### Study 1 vs Study 2 訓練對比

| | Study 1 (Piano) | Study 2 (Universal Solo) |
|---|---|---|
| **目標** | 公平比較架構差異 | 展示跨樂器泛化能力 |
| **訓練資料** | MuseSyn + HumSyn（與 Zeng 相同） | MuseSyn + HumSyn + 多樂器拆分 + PDMX 非古典 |
| **測試資料** | ASAP test (25首/80段) | ASAP + GAPS + Bach Violin + GTSinger |
| **訓練類型** | Supervised | Zero-shot |
| **樂器標籤** | `*Ipiano` (固定) | Kern 原生標籤 (`*Ipiano`, `*Iguitr`, etc.) |
| **Auxiliary Loss** | 不使用 | 不使用（留給 ICLR 2027） |

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

### Study 2: Universal Solo 訓練（同樂器 Augmentation）

Study 2 使用 **同樂器內 Augmentation** 策略，而非跨樂器 TDR：

```
Clef (Study 2) 訓練流程：
┌─────────────────────────────────────────────────────────┐
│  Training: Universal Solo Pre-training                   │
│  ├── Data: PDMX (250K+ scores，涵蓋多種樂器)             │
│  ├── Audio: 同樂器 Augmentation（不跨樂器！）            │
│  │   ├── Piano: Steinway, Yamaha, Upright, Electric      │
│  │   ├── Guitar: Classical, Steel-string, Nylon          │
│  │   ├── Violin: Stradivarius, Modern, Baroque           │
│  │   └── ...每種樂器用 3-5 種不同音源                    │
│  ├── Kern 樂器標籤: *Ipiano, *Iguitr, *Ivioln, etc.     │
│  └── 不使用任何真實錄音！                                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Test: Sim-to-Real Evaluation (Zero-shot)               │
│  ├── Piano: ASAP test (真實鋼琴錄音)                     │
│  ├── Guitar: GuitarSet (真實吉他錄音)                    │
│  ├── Strings/Winds: URMP solo tracks (真實錄音)          │
│  └── 證明 Sim2Real 泛化能力                              │
└─────────────────────────────────────────────────────────┘
```

**為什麼用「同樂器 Augmentation」而非「跨樂器 TDR」？**

| 策略 | 優點 | 缺點 | 適用場景 |
|------|------|------|---------|
| **同樂器 Augmentation** | 模型自然學會正確樂器標籤 | 需要每種樂器都有足夠訓練資料 | ISMIR 2026 (單樂器) |
| **跨樂器 TDR** | 最大化數據效率 | 需要 Aux Loss 解耦 | ICLR 2027 (多樂器合奏) |

### 訓練資料需求總覽

| Study | 訓練資料 | 測試資料 | 需下載 |
|-------|----------|----------|--------|
| Study 1 | MuseSyn + HumSyn (與 Zeng 相同) | ASAP test (25首/80段) | ASAP + MAESTRO |
| Study 2 | MuseSyn + HumSyn + 多樂器拆分 + PDMX 非古典 | ASAP + GAPS + Bach Violin + GTSinger | GAPS, Bach Violin, GTSinger |

### Study 2 訓練資料策略

**核心策略**：不使用 PDMX 的 50,000+ Piano Solo，避免樂器不平衡和 overfit。

| 來源 | Piano 數量 | Genre | 角色 |
|------|-----------|-------|------|
| MuseSyn | ~200 | Classical | Study 1 baseline |
| HumSyn | ~2,000 | Classical | Study 1 baseline |
| PDMX (Pop/Jazz/Rock) | ~2,000-3,000 | Non-classical | 多元化 |
| 拆分的伴奏 | ~5,000 | Mixed | 伴奏角色 |
| **Total Piano** | **~10,000** | **Balanced** | ✅ |

**為什麼這樣設計？**

1. **承接 Study 1**：Piano 資料與 Zeng 相同，確保公平比較
2. **防止 Overfit**：不讓 50,000+ Piano 主導訓練
3. **伴奏 Piano 的價值**：從 Piano-Voice、Piano-Violin 等拆出來的 Piano 是伴奏角色，音域、節奏、複雜度都不同於 Solo Piano

**其他樂器的資料策略**：

| 樂器 | Solo 數量估計 | 拆分補充 | 總計 |
|------|--------------|---------|------|
| **Violin** | ~3,000-5,000 | String Quartet | ~8,000 |
| **Voice** | ~2,000 | Piano-Voice Lieder | ~8,000 |
| **Cello** | ~500-1,000 | String Quartet | ~4,000 |
| **Viola** | ~200-500 | String Quartet | ~3,000 |
| **Flute** | ~500-1,000 | Chamber Music | ~3,000 |
| **Guitar** | ~2,000 | — | ~2,000 |

**Genre 多元化（好和弦策略）**：

PDMX rated subset (~14,182 首) 中約 40% 是非古典/民謠：
- Pop, Rock, Jazz, Blues, R&B, Latin, World, Soundtrack

```python
# 篩選非古典的 rated songs
non_classical = df[
    (df['is_rated'] == True) &
    (~df['genre'].isin(['classical', 'folk', None, '']))
]
```

**Paper 可以這樣寫**：
> "To ensure genre diversity and prevent classical music bias, we supplement MuseSyn and HumSyn (classical piano) with non-classical works from PDMX's rated subset, which contains approximately 40% non-classical/folk genres including pop, jazz, rock, and world music."

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

## Study 1: Depth (深度) — ASAP Dataset

### Clef 變體設計

為了區分各設計決策的貢獻，我們設計一系列 Clef 變體：

| 變體 | Input | Encoder | Bridge | 目的 |
|------|-------|---------|--------|------|
| **Zeng (2024)** | Mono VQT | CNN | N/A | Baseline |
| **Clef-ViT** | Log-Mel | ViT | N/A | **證明 Transformer > RNN** |
| **Clef-Swin** | Log-Mel | Swin-V2 | N/A | **證明 Swin > ViT** |
| **Clef-Swin + Bridge** | Log-Mel | Swin-V2 | 2 layers | **證明 Bridge 的必要性** |
| **Clef-Full** | Stereo 3-ch | Swin-V2 | 2 layers | **最佳性能**（含前處理改進） |

**Clef 變體說明**：
- **Clef-ViT**：與 Zeng 使用相同輸入（Log-Mel），驗證 Transformer Decoder 優於 Hierarchical RNN
- **Clef-Swin**：驗證 Swin-V2 優於 ViT（相對位置編碼 vs 絕對位置插值）
- **Clef-Swin + Bridge**：驗證 Global Transformer Bridge 對段落結構理解的貢獻
- **Clef-Full**：加入所有前處理改進（Stereo 3-ch + Loudness norm + L/R flip）

> **注意**：ISMIR 2026 版本不使用 Instrument Auxiliary Loss。Aux Loss 留給 ICLR 2027 的多樂器合奏版本。

### Table 1: Comparison of A2S Systems on Real-World Recordings (Piano)

| Approach | System | Audio Model | Score Model | MV2H | $F_p$ | $F_{harm}$ | 關鍵弱點 |
|----------|--------|-------------|-------------|------|-------|-----------|----------|
| Pipeline | MT3 + MuseScore 4 | MT3 (CNN) | music21 (Rule) | ~58% | ~80% | ~40% | **量化災難**：啟發式演算法無法處理 Rubato 與複雜節奏 |
| Pipeline | Transkun + Beyer | Transkun (Trans.) | Beyer (Trans.) | ~68% | ~92% | ~50% | **誤差傳播**：MIDI 層級的小誤差在符號化時被放大 |
| E2E | Zeng (2024) | CNN | H-RNN | 74.2% | 63.3% | 54.5% | **局部感受野**：CNN 無法捕捉長距離和聲結構 |
| E2E | Clef-ViT | ViT | Transformer | ~77% | 70% | 58% | **絕對位置**：ViT 對變長輸入支援不佳 |
| E2E | **Clef-Swin** | **Swin-V2** | Transformer | **~80%** | **75%** | **62%** | **缺 Bridge**：無全域段落結構理解 |
| E2E | **Clef-Swin + Bridge** | **Swin-V2** | **Transformer + Bridge** | **~84%** | **79%** | **68%** | **最佳架構** |
| E2E | **Clef-Full** | **Swin-V2** | **Transformer + Bridge** | **~85%** | **81%** | **70%** | **完整系統**（含前處理改進） |

**評估設定**：
- 資料集：ASAP test split (25 首 / 80 段錄音)
- 評估指標：MV2H (Non-aligned, McLeod 2019)
- 統一評估流程：所有系統 → (Slice to 5-bar if needed) → MusicXML → MIDI → MV2H

### 貢獻分解

```
總提升 = Clef-Full - Zeng = ~11%

├── ViT + Transformer vs CNN + RNN: ~3%
│
├── ViT → Swin-V2: ~3%
│
├── Swin-V2 → +Bridge: ~4%
│
└── 前處理改進: ~1%
    ├── Stereo 3-channel input
    ├── Loudness normalization
    └── L/R flip augmentation
```

> **注意**：Instrument Auxiliary Loss 不在 ISMIR 2026 版本使用，留給 ICLR 2027。

> **註**：Transkun 的 $F_p$ 設為 92% 是參考其 MAESTRO 數據，但轉成 XML 後 MV2H 通常會掉下來。Zeng 的數據來自其論文中的 ASAP 實測。

### Baseline 選擇理由

1. **為什麼選 MT3 + MuseScore 4？**
   - 這是 **Baseline of Baselines**
   - MT3 是目前引用率最高的 Audio-to-MIDI 模型
   - music21 是最多人用的處理庫
   - 目的：證明「工業標準」在轉譜任務上不及格

2. **為什麼選 Transkun + Beyer？**
   - 這是 **防禦性攻擊 (Defensive Attack)**
   - 預防審稿人說：「MT3 表現爛是因為它舊了」
   - 如果連這套 SOTA Combo 都輸，就證明了 **Pipeline 方法論本身的失敗**

3. **為什麼不交叉 (Cross-match)？**
   - MT3 + Beyer (爛頭+好尾) 和 Transkun + music21 (好頭+爛尾) 結果介於中間
   - 對論證「E2E vs Pipeline」的優劣沒有額外幫助

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

### 3. 同樂器 Augmentation vs 無 Augmentation（Study 2 專用）

驗證同樂器 Augmentation 對跨樂器泛化的貢獻：

| 實驗 | 訓練策略 | Piano MV2H | Guitar MV2H | Strings MV2H |
|------|----------|------------|-------------|--------------|
| Clef-Swin + Bridge (無 Aug) | 單一音源 | ~84% | ~40% | ~35% |
| **Clef-Swin + Bridge (同樂器 Aug)** | 同樂器多音源 | **~84%** | **~60%** | **~55%** |

**預期結論**：
- 同樂器 Augmentation 顯著提升跨音源泛化能力
- 對 Piano 效果有限（因為 ASAP 本身就有多種演奏者）
- 對 Guitar/Strings 效果顯著（因為測試資料是完全不同的音源）

> **注意**：Instrument Auxiliary Loss 不在本 Study 使用，留給 ICLR 2027 的多樂器合奏版本。

### 4. 頻譜表示 Ablation（VQT vs Log-Mel）

驗證 Log-Mel 對音色保留的優勢：

| 實驗 | 頻譜類型 | 解析度 | 預期 Piano MV2H | 預期 Guitar MV2H | 備註 |
|------|---------|--------|-----------------|------------------|------|
| Clef-Swin + VQT | VQT | 60 bins/oct | ~83% | ~45% | 音高解析度高，但音色扭曲 |
| **Clef-Swin + Log-Mel** | Log-Mel | 128 bins | **~85%** | **~60%** | 音色保留佳，ImageNet 相容 |
| Clef-Swin + Log-Mel-256 | Log-Mel | 256 bins | ~84% | ~58% | 邊際效益遞減 |

**科學問題**：「對於跨樂器泛化，Log-Mel 是否比 VQT 更適合？」

**理論基礎**：
- VQT 會對頻譜進行非線性扭曲，破壞共振峰（Formant）位置
- 共振峰是區分小提琴 vs 中提琴的關鍵特徵
- Log-Mel 保留頻譜包絡，有利於跨樂器泛化

### 5. 前處理 Ablation

逐步加入前處理改進，量化各自貢獻：

| 實驗 | Input | Normalization | Augmentation | 預期 MV2H |
|------|-------|---------------|--------------|-----------|
| Clef-base | Mono Log-Mel | ❌ | ❌ | ~80% |
| + Loudness | Mono Log-Mel | ✅ | ❌ | ~81% |
| + Stereo | Stereo 3-ch | ✅ | ❌ | ~83% |
| + L/R Flip | Stereo 3-ch | ✅ | ✅ | ~84% |

### 6. 完整 Ablation 總結表

| 設計決策 | 預期貢獻 | 驗證方式 |
|---------|---------|---------|
| ViT → Swin-V2 | +2~3% | 編碼器 Ablation |
| Swin → +Bridge | +2~3% | Bridge Ablation |
| Bridge-0 → Bridge-2 | +2% | Bridge Ablation |
| VQT → Log-Mel | +2~3% | 頻譜 Ablation |
| Loudness Norm | +1% | 前處理 Ablation |
| Stereo 3-ch | +1~2% | 前處理 Ablation |
| L/R Flip | +1% | 前處理 Ablation |
| 同樂器 Augmentation | +15~20% (非鋼琴) | Study 2 (Universal Solo) |

> **注意**：Instrument Auxiliary Loss 不在此版本使用，留給 ICLR 2027 的多樂器合奏版本。

### 7. 消融實驗預期結果表（Study 1: Piano）

| Model Configuration | MV2H | $F_p$ | $F_{voi}$ | $F_{val}$ | $F_{harm}$ | TEDn |
|---------------------|------|-------|-----------|-----------|------------|------|
| Zeng (2024) | 74.2 | 63.3 | 88.4 | 90.7 | 54.5 | 0.72 |
| Clef-ViT + Transformer | 77.0 | 70.0 | 86.0 | 89.0 | 58.0 | 0.75 |
| Clef-Swin + Transformer | 80.0 | 75.0 | 87.0 | 90.0 | 62.0 | 0.77 |
| Clef-Swin + Bridge-0 | 80.0 | 75.0 | 87.0 | 90.0 | 62.0 | 0.77 |
| Clef-Swin + Bridge-1 | 82.0 | 77.0 | 88.0 | 91.0 | 65.0 | 0.78 |
| **Clef-Swin + Bridge-2** | **84.0** | **79.0** | **89.0** | **92.0** | **68.0** | **0.80** |
| **Clef-Full (+ 前處理)** | **85.0** | **81.0** | **90.0** | **93.0** | **70.0** | **0.81** |

### 8. Study 2 預期結果表（Universal Solo — 4 Instrument Categories）

| Model | Piano (ASAP) | Guitar (GAPS) | Violin (Bach) | Voice (GTSinger) | Avg |
|-------|--------------|---------------|---------------|------------------|-----|
| Clef (Study 1, Piano Only) | 85.0 | ~25% | ~20% | ~20% | ~38% |
| **Clef (Study 2, Universal)** | **85.0** | **~60%** | **~55%** | **~50%** | **~63%** |

**評估資料集對應**：
- Piano: ASAP test split (與 Study 1 相同)
- Guitar: GAPS (古典吉他，MusicXML)
- Violin: Bach Violin Dataset (BWV 1001-1006，MusicXML)
- Voice: GTSinger (多語言人聲，MusicXML)

> **註**：Study 2 使用同樂器 Augmentation，不使用 Instrument Auxiliary Loss。Aux Loss 留給 ICLR 2027 的多樂器合奏版本。

> **Limitation**：由於缺乏公開的 Cello/Viola/Flute + MusicXML 資料集，這些樂器的定量評估留待未來研究。

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

## Study 2: Breadth (廣度) — Universal Solo Benchmark

### 設計理念

Study 2 的目標是證明 Clef 能夠成為 **通用單樂器轉譜系統**，而不是「另一個鋼琴專用模型」。

核心問題：
> 「一個訓練在各種單樂器譜（用同樂器不同音源做 augmentation）的模型，能不能在真實錄音上正確轉譜各種樂器？」

### 策略：使用 Kern 原生的樂器標籤

Kern 格式本身就有內建的樂器標籤（Tandem Interpretation），不需要自己設計 token：

| Kern Code | 樂器 | 譜表格式 |
|-----------|------|---------|
| `*Ipiano` | 鋼琴 | Grand Staff (大譜表) |
| `*Iguitr` | 吉他 | 單譜表 + 8va |
| `*Ivioln` | 小提琴 | 單譜表 (G clef) |
| `*Iviola` | 中提琴 | 中音譜號 (Alto clef) |
| `*Icello` | 大提琴 | 低音譜號 (Bass clef) |
| `*Iflt` | 長笛 | 單譜表 (G clef) |
| `*Iclars` | 單簧管 | 單譜表 (移調樂器) |
| `*Ioboe` | 雙簧管 | 單譜表 |
| `*Imandol` | 曼陀林 | 單譜表 + 8va |

**模型輸出**：正確的 Kern（含 `*I` 樂器標籤）→ 自動產生正確的譜表格式

### 訓練策略：同樂器內 Augmentation

| 策略 | 說明 |
|------|------|
| **同樂器 Augmentation** | 鋼琴譜只用不同鋼琴音源（Steinway, Yamaha, Upright, Electric）|
| **不跨樂器** | 不會出現「吉他譜 + 鋼琴音色」的組合 |
| **Kern 原生標籤** | `*Iguitr` 自動對應吉他記譜法（單譜表 + 8va）|

**為什麼不用「跨樂器 TDR」？**
- 跨樂器 TDR 需要 Instrument Auxiliary Loss 來幫助模型解纏（Disentangle）
- 這個策略留給 ICLR 2027 的多樂器合奏版本
- ISMIR 2026 版本專注於「單樂器」場景，Auxiliary Loss 不是必要的

**為什麼不用「家族內 TDR」（例如弦樂家族內互換）？**
- 曼陀林和小提琴音域一樣，用幾何特徵無法區分
- 模型必須從音色中學會區分樂器，而不是從幾何結構
- 如果用「小提琴音色 + 曼陀林譜」訓練，模型會學錯樂器標籤

### 測試資料：Sim-to-Real Evaluation

使用真實錄音測試，驗證合成訓練資料的泛化能力：

| 樂器類別 | 資料集 | 規模 | 樂譜格式 | 說明 |
|---------|--------|------|----------|------|
| 🎹 Piano | ASAP test split | 80 段 | MusicXML | 與 Study 1 相同 |
| 🎸 Guitar | GAPS | 14 小時 | MusicXML | 古典吉他真實錄音 |
| 🎻 Violin | Bach Violin | 6.5 小時 | MusicXML | BWV 1001-1006 |
| 🎤 Voice | GTSinger | 80+ 小時 | MusicXML | 9 語言多元人聲 |

> **Limitation**: 目前沒有公開可用的 **中提琴、大提琴、長笛** 資料集同時包含真實錄音 + MusicXML 樂譜。這些樂器的評估留待未來研究。

### Table 2: Cross-instrument Zero-Shot Transfer (4 Instrument Categories)

| Model Strategy | Training Data | Piano | Guitar | Violin | Voice |
|----------------|---------------|-------|--------|--------|-------|
| MT3 + MuseScore 4 | MAESTRO | ~58% | ~30% | ~25% | ~20% |
| Clef (Study 1) | Piano Only | **~85%** | < 25% | < 20% | < 20% |
| **Clef (Study 2)** | **Universal Solo** | **~85%** | **~60%** | **~55%** | **~50%** |

**評估說明**：
- 所有指標為 MV2H (Non-aligned)
- **4 個樂器類別**：Piano, Guitar, Violin, Voice — 涵蓋大譜表、撥弦、弓弦、人聲
- MT3 + MuseScore 4 在非鋼琴樂器上的「量化災難」更嚴重
- Clef (Study 1) 只練鋼琴，遇到非鋼琴樂器完全失效
- Clef (Study 2) 使用 Universal Solo 訓練策略，展現跨樂器泛化能力

**Contribution Statement**：
> "To the best of our knowledge, Clef is the first end-to-end audio-to-score system validated across **4 distinct instrument categories** (keyboard, plucked string, bowed string, voice) on real-world recordings."

### 表格亮點

1. **Clef (Study 1)**：證明「專用模型」的侷限性（只練鋼琴，其他樂器完全失效）
2. **Clef (Study 2)**：證明 Kern 原生樂器標籤 + 同樂器 Augmentation 的有效性
   - Swin 的階層式結構學習音色紋理
   - Bridge 捕捉曲式結構
   - 不需要 Auxiliary Loss 也能達到 **4 種樂器類別** 的跨樂器泛化
3. **MT3 + MuseScore 4**：Pipeline 在非鋼琴樂器上的「量化災難」更嚴重（缺乏樂器特定記譜規則）

### Study 2 的「Cute Killer」策略

ISMIR 審稿人是音樂學者，他們在乎的是：
- 這個模型對**我的樂器**有用嗎？
- 轉出來的譜**能不能讀**？

你不需要在 Study 2 強調技術細節（Swin、Bridge），而是強調：
> "Clef is not just another piano transcription model — it's designed for **all musicians**."

技術細節留給 ICLR 2027。

---

## 論文結構總覽（ISMIR 2026）

| Study | 定位 | 戰場 | 對手 | 目標 |
|-------|------|------|------|------|
| Study 1 | Depth (深度) | ASAP (Piano) | Zeng 2024, MT3 + MuseScore 4 | MV2H > 78% |
| Study 2 | Breadth (廣度) | ASAP + GAPS + Bach Violin + GTSinger | MT3 + MuseScore 4 | Cross-instrument MV2H > 55% |

### 樂器覆蓋總結

| 類別 | 樂器 | Evaluation Dataset | 狀態 |
|------|------|-------------------|------|
| 大譜表（鍵盤） | 鋼琴 | ASAP | ✅ 充足 |
| 撥弦 | 古典吉他 | GAPS | ✅ 充足 |
| 弓弦 | 小提琴 | Bach Violin | ✅ 可用 |
| 人聲 | 獨唱 | GTSinger | ✅ 充足 |
| 木管/銅管 | 長笛、單簧管等 | — | ⚠️ 缺口 |
| 其他弦樂 | 中提琴、大提琴 | — | ⚠️ 缺口 |

> **Paper Limitation Statement**: "Due to the lack of publicly available datasets with aligned audio and musical scores for viola, cello, and wind instruments, we leave their evaluation to future work."

### 核心論點

> 「Clef 不只是另一個鋼琴轉譜模型 — 它是為所有音樂家設計的通用單樂器轉譜系統。」

### ISMIR 2026 vs ICLR 2027 差異

| 面向 | ISMIR 2026 (本文件) | ICLR 2027 |
|------|---------------------|-----------|
| **目標** | 單樂器轉譜 | Representation Learning |
| **視覺資訊** | **清掉**（stem, beam 等） | **學習**（Visual Auxiliary Head） |
| **TDR 策略** | 同樂器內換音源 | 同樂器內換音源 |
| **Auxiliary Loss** | 不使用 | Instrument Aux + Visual Aux |
| **輸出格式** | Kern（語意為主） | Kern（語意 + 視覺佈局） |
| **TEDn 評估** | Optimality Gap 方法 | 完整 TEDn（含視覺） |
| **核心賣點** | 「能用」 | 「為什麼能用」 |

### 視覺資訊處理策略（ISMIR 2026）

ISMIR 版本採用**清除視覺資訊**策略，原因如下：

1. **TEDn 的侷限**：TEDn 評估 stem direction、voice assignment 等視覺元素，但 A2S 模型無法從音訊推斷這些資訊
2. **簡化任務**：讓模型專注於語意正確性（音高、時值、和聲）
3. **Optimality Gap 評估**：用 `XML→Kern→XML` 的 TEDn 作為 upper bound，模型真正的錯誤 = TEDn_model - TEDn_conversion
4. **Demo 友善**：截圖選擇視覺乾淨的段落即可

**clean_kern.py 的處理**：
- 移除 stem direction（`/` `\`）
- 移除 above/below staff markers（`>` `<`，僅在有 duration 的 token 中）
- 保留語意必要的資訊（pitch, duration, ties）

### 後續研究方向（ICLR 2027 預告）

ISMIR 2026 證明了 Clef 在單樂器場景的能力後，ICLR 2027 將探討 representation learning：

1. **Visual Auxiliary Head**：從 decoder hidden states 預測視覺屬性（stem, beam, voice, staff assignment）
   - 核心問題：「學習視覺佈局是否幫助語意理解？」
   - Loss = L_main + λ_vis * L_visual（λ_vis ≈ 0.1，避免視覺任務主導）

2. **Instrument Auxiliary Loss**：強迫編碼器保留音色資訊

> 詳見：`experiment-design-tutti.md`
