# Experiment Design: Clef 實驗設計

本文件描述 Clef 研究的兩個核心實驗：Study 1（深度）與 Study 2（廣度）。

---

## 實驗策略：稻草人與鋼鐵人

採用 **「攻擊稻草人與鋼鐵人 (The Straw Man and The Steel Man)」** 策略，不需做 $2 \times 2$ 的交叉實驗，只需挑出兩組最具代表性的 Pipeline：

1. **Standard Baseline (稻草人)**：**MT3 + music21**
   - **角色**：代表「一般大眾/工程師」最常用的解法
   - **目的**：證明「傳統做法」完全不可行 (MV2H < 60%)，凸顯題目價值

2. **Strong Baseline (鋼鐵人)**：**Transkun + Beyer**
   - **角色**：代表「目前學術界最強」的拼裝車
   - **目的**：證明「即使把最強的零件拼起來」，還是會有 **誤差傳播 (Error Propagation)**，依然輸給 End-to-End

---

## Zeng et al. (2024) 實驗設定參考

為確保公平比較，本研究採用與 Zeng et al. (2024) 相同的實驗設定。

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

### URMP Dataset（Study 2 - 多樂器）

需要填表單申請下載。

**來源**：
- 官網: https://labsites.rochester.edu/air/projects/URMP.html
- 大小: 12.5GB

**下載步驟**：
1. 前往 [URMP 官網](https://labsites.rochester.edu/air/projects/URMP.html)
2. 點擊 **"Download the whole dataset package"**
3. 填寫 Google Form（學術用途）
4. 收到 email 後下載

**資料結構**：
```
URMP/
├── 01_Jupiter_vn_vc/           # Duet: 小提琴 + 大提琴
│   ├── AuMix_01_Jupiter.wav    # 混音音訊
│   ├── AuSep_1_vn_01.wav       # 小提琴分軌
│   ├── AuSep_2_vc_01.wav       # 大提琴分軌
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

### 資料集規模總覽

| Dataset | 用途 | 檔案數 | 大小 |
|---------|------|--------|------|
| ASAP (full) | Study 1 完整版 | 1,067 段 | ~30GB |
| ASAP (test only) | Study 1 baseline | ~80 段 | ~3GB |
| URMP | Study 2 | 44 首 | 12.5GB |

---

## Clef 訓練策略

為確保公平比較，Clef 在不同 Study 採用不同訓練策略。

### Study 1 vs Study 2 訓練對比

| | Study 1 (ASAP) | Study 2 (URMP) |
|---|---|---|
| **目標** | 公平比較架構差異 | 展示泛化能力 |
| **訓練資料** | 與 Zeng 完全相同 | PDMX + TDR |
| **測試資料** | ASAP test (25首/80段) | URMP (44首) |
| **訓練類型** | Supervised | Zero-shot |

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

### Study 2: Zero-Shot 泛化（展示能力）

Study 2 不需要與任何人比較，目標是展示 Clef 的泛化能力：

```
Clef (Study 2) 訓練流程：
┌─────────────────────────────────────────────────────────┐
│  Training: Universal Pre-training                        │
│  ├── Data: PDMX (250K+ scores，涵蓋多種樂器)             │
│  ├── Audio: TDR 合成（多種樂器音色）                     │
│  └── 不使用任何真實錄音！                                │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  Test: URMP (Zero-shot)                                  │
│  ├── 真實錄音（從未見過）                                │
│  ├── 多種樂器（小提琴、長笛、單簧管...）                 │
│  └── 證明 Sim2Real 泛化能力                              │
└─────────────────────────────────────────────────────────┘
```

### 訓練資料需求總覽

| Study | 訓練資料 | 測試資料 | 需下載 |
|-------|----------|----------|--------|
| Study 1 | ASAP train (14首/58段) | ASAP test (25首/80段) | ASAP + MAESTRO |
| Study 2 | PDMX + TDR (合成) | URMP (44首) | URMP |

---

## Study 1: Depth (深度) — ASAP Dataset

### Table 1: Comparison of A2S Systems on Real-World Recordings

| Approach | System | Role | MV2H | $F_p$ (音高) | $F_{harm}$ (和聲) | 弱點分析 |
|----------|--------|------|------|-------------|------------------|----------|
| Pipeline | MT3 + music21 | Industry Std. | ~58.0% | ~80.0% | ~40.0% | **量化災難**：music21 的啟發式算法無法處理 Rubato |
| Pipeline | Transkun + Beyer | SOTA Combo | ~68.0% | ~92.0% | ~50.0% | **語義鴻溝**：Beyer 模型仍無法完美修復 MIDI 的語義缺失 |
| End-to-End | Zeng et al. (2024) | E2E Baseline | 74.2% | 63.3% | 54.5% | **聽覺失聰**：CNN 架構無法處理真實錄音 |
| End-to-End | **Clef (Ours)** | Proposed | **> 78.0%** | **> 85.0%** | **> 65.0%** | Sim2Real + ViT 全面勝出 |

> **註**：Transkun 的 $F_p$ 設為 92% 是參考其 MAESTRO 數據，但轉成 XML 後 MV2H 通常會掉下來。Zeng 的數據來自其論文中的 ASAP 實測。

### Baseline 選擇理由

1. **為什麼選 MT3 + music21？**
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

## Study 2: Breadth (廣度) — URMP Dataset

### 設計理念

Study 2 的目標不是「換個戰場繼續卷分數」，而是 **「廣度的展現 (Generalization)」**。

核心問題：
> 「如果我把全世界的譜 (PDMX 250k)，用電腦合成出各種聲音 (TDR) 餵給模型，它能不能學會『聽懂音樂』，而不只是『聽懂鋼琴』？」

### 為什麼不跟其他人比？

| 對手 | 風險 | 問題 |
|------|------|------|
| Alfaro-Contreras (2024) 弦樂四重奏 | 變成「另一個做特定樂器轉譜的人」 | 邊際效應遞減 |
| Zhang (2024) 流行歌 | 指標不同 (WER vs MV2H) | 難以直接比較 |

### Table 2: Zero-Shot Generalization on Unseen Instruments

**Dataset**: URMP (University of Rochester Multi-Modal Musical Performance)
- 包含多種樂器（小提琴、長笛、單簧管...）的真實錄音
- Zeng 沒測過，MT3 測過但效果普普

| Model Strategy | Training Data | Piano | Strings | Winds | Ensemble |
|----------------|---------------|-------|---------|-------|----------|
| MT3 + music21 | MAESTRO + Slakh | ~75% (MV2H) | ~35% (MV2H) | ~30% (MV2H) | ~25% (MV2H) |
| Clef (Study 1) | Piano Only | **> 78% (MV2H)** | < 20% (Fail) | < 20% (Fail) | < 20% (Fail) |
| Clef (Study 2) | **Universal (TDR)** | **> 76% (MV2H)** | **> 60% (MV2H)** | **> 60% (MV2H)** | **> 55% (MV2H)** |

> **註**：MT3 + music21 的 MV2H 預估值基於 Study 1 的「量化災難」現象。實際數據需實驗驗證。

### 表格亮點

1. **Clef (Study 1)**：證明「專用模型」的侷限性（只練鋼琴，遇到小提琴就掛了）
2. **Clef (Study 2)**：Big Data + TDR 展現 **Zero-Shot 能力**
3. **對比 MT3 + music21**：Pipeline 在非鋼琴樂器上的「量化災難」更嚴重

---

## 論文結構總覽

| Study | 定位 | 戰場 | 對手 | 目標 |
|-------|------|------|------|------|
| Study 1 | Depth (深度) | ASAP (Piano) | Zeng 2024, MT3 + music21 | MV2H > 78% |
| Study 2 | Breadth (廣度) | URMP (Multi-instrument) | MT3 + music21 | Zero-shot MV2H > 60% |

### 核心論點

> 「只要有譜，我就能生成訓練資料；只要有訓練資料，我的 VLM 就能學會轉任何樂器。」

這才是「通用轉譜 AI」的真正價值。
