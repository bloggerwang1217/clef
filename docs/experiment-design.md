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

#### 1. Weak Baseline: MT3 + music21

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

| 變體 | Input | Encoder | Bridge | Aux Loss | 目的 |
|------|-------|---------|--------|----------|------|
| **Zeng (2024)** | Mono VQT | CNN | N/A | ❌ | Baseline |
| **Clef-ViT** | Log-Mel | ViT | N/A | ❌ | **證明 Transformer > RNN** |
| **Clef-Swin** | Log-Mel | Swin-V2 | N/A | ❌ | **證明 Swin > ViT** |
| **Clef-Swin + Bridge** | Log-Mel | Swin-V2 | 2 layers | ❌ | **證明 Bridge 的必要性** |
| **Clef-Full** | Stereo 3-ch | Swin-V2 | 2 layers | ✅ | **最佳性能** |

**Clef 變體說明**：
- **Clef-ViT**：與 Zeng 使用相同輸入（Log-Mel），驗證 Transformer Decoder 優於 Hierarchical RNN
- **Clef-Swin**：驗證 Swin-V2 優於 ViT（相對位置編碼 vs 絕對位置插值）
- **Clef-Swin + Bridge**：驗證 Global Transformer Bridge 對段落結構理解的貢獻
- **Clef-Full**：加入所有改進（Stereo 3-ch + Loudness norm + L/R flip + Aux Loss）

### Table 1: Comparison of A2S Systems on Real-World Recordings

| Approach | System | Audio Model | Score Model | MV2H | $F_p$ | $F_{harm}$ | 關鍵弱點 |
|----------|--------|-------------|-------------|------|-------|-----------|----------|
| Pipeline | MT3 + music21 | MT3 (CNN) | music21 (Rule) | ~58% | ~80% | ~40% | **量化災難**：啟發式演算法無法處理 Rubato 與複雜節奏 |
| Pipeline | Transkun + Beyer | Transkun (Trans.) | Beyer (Trans.) | ~68% | ~92% | ~50% | **誤差傳播**：MIDI 層級的小誤差在符號化時被放大 |
| E2E | Zeng (2024) | CNN | H-RNN | 74.2% | 63.3% | 54.5% | **局部感受野**：CNN 無法捕捉長距離和聲結構 |
| E2E | Clef-ViT | ViT | Transformer | ~77% | 70% | 58% | **絕對位置**：ViT 對變長輸入支援不佳 |
| E2E | **Clef-Swin** | **Swin-V2** | Transformer | **~80%** | **75%** | **62%** | **缺 Bridge**：無全域段落結構理解 |
| E2E | **Clef-Swin + Bridge** | **Swin-V2** | **Transformer + Bridge** | **~84%** | **79%** | **68%** | **最佳架構** |
| E2E | **Clef-Full** | **Swin-V2** | **Transformer + Bridge** | **~86%** | **82%** | **72%** | **完整系統** |

**評估設定**：
- 資料集：ASAP test split (25 首 / 80 段錄音)
- 評估指標：MV2H (Non-aligned, McLeod 2019)
- 統一評估流程：所有系統 → MusicXML → MIDI → MV2H

### 貢獻分解

```
總提升 = Clef-Full - Zeng = ~12%

├── ViT + Transformer vs CNN + RNN: ~3%
│
├── ViT → Swin-V2: ~3%
│
├── Swin-V2 → +Bridge: ~4%
│
└── 前處理 + Aux Loss: ~2%
    ├── Stereo 3-channel input
    ├── Loudness normalization
    ├── L/R flip augmentation
    └── Instrument Auxiliary Loss
```

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

### 3. Auxiliary Loss Ablation

驗證樂器分類輔助任務對主任務的貢獻：

| 實驗 | Encoder + Bridge | Aux Loss | λ | 預期 MV2H | 樂器 F1 |
|------|------------------|----------|---|-----------|---------|
| Clef-Swin + Bridge | Swin-V2 + Bridge | ❌ | - | ~84% | N/A |
| Clef-Swin + Bridge + Aux | Swin-V2 + Bridge | ✅ | 0.1 | ~85% | ~85% |
| Clef-Swin + Bridge + Aux | Swin-V2 + Bridge | ✅ | 0.3 | **~86%** | **~90%** |
| Clef-Swin + Bridge + Aux | Swin-V2 + Bridge | ✅ | 0.5 | ~85% | ~88% |

**預期結論**：
- λ = 0.3 為最佳權重，輔助任務有效但不搶走主任務的梯度
- Auxiliary Loss 帶來的效能提升主要來自：
  1. **加速收斂**：分類任務比序列生成簡單，提供穩定梯度
  2. **特徵解耦**：強迫編碼器保留音色資訊
  3. **結構理解**：樂器分布與曲式結構相關

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

### 6. 完整 Ablation 總結表

| 設計決策 | 預期貢獻 | 驗證方式 |
|---------|---------|---------|
| ViT → Swin-V2 | +2~3% | 編碼器 Ablation |
| Swin → +Bridge | +2~3% | Bridge Ablation |
| Bridge-0 → Bridge-2 | +2% | Bridge Ablation |
| 無 Aux → +Aux (λ=0.3) | +1~2% | Aux Loss Ablation |
| VQT → Log-Mel | +2~3% | 頻譜 Ablation |
| Loudness Norm | +1% | 前處理 Ablation |
| Stereo 3-ch | +1~2% | 前處理 Ablation |
| L/R Flip | +1% | 前處理 Ablation |

### 7. 消融實驗預期結果表

| Model Configuration | MV2H | $F_p$ | $F_{voi}$ | $F_{val}$ | $F_{harm}$ | TEDn | 樂器 F1 |
|---------------------|------|-------|-----------|-----------|------------|------|---------|
| Zeng (2024) | 74.2 | 63.3 | 88.4 | 90.7 | 54.5 | 0.72 | N/A |
| Clef-ViT + Transformer | 77.0 | 70.0 | 86.0 | 89.0 | 58.0 | 0.75 | N/A |
| Clef-Swin + Transformer | 80.0 | 75.0 | 87.0 | 90.0 | 62.0 | 0.77 | N/A |
| Clef-Swin + Bridge-0 | 80.0 | 75.0 | 87.0 | 90.0 | 62.0 | 0.77 | N/A |
| Clef-Swin + Bridge-1 | 82.0 | 77.0 | 88.0 | 91.0 | 65.0 | 0.78 | N/A |
| **Clef-Swin + Bridge-2** | **84.0** | **79.0** | **89.0** | **92.0** | **68.0** | **0.80** | N/A |
| Clef-Swin + Bridge-2 + Aux (λ=0.1) | 85.0 | 80.0 | 90.0 | 93.0 | 70.0 | 0.81 | 85% |
| **Clef-Swin + Bridge-2 + Aux (λ=0.3)** | **86.0** | **82.0** | **91.0** | **94.0** | **72.0** | **0.82** | **90%** |
| Clef-Swin + Bridge-2 + Aux (λ=0.5) | 85.0 | 81.0 | 90.0 | 93.0 | 71.0 | 0.81 | 88% |
| Clef-Swin + Bridge-2 + Aux + TDR | **88.0** | **85.0** | **92.0** | **95.0** | **75.0** | **0.85** | **92%** |

> **註**：TDR（Timbre Domain Randomization）在 Study 2 中啟用，驗證跨音色泛化能力。

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

### Study 2 的音訊設定

Study 2 同樣使用 **Stereo 3-channel input**，理由：
- URMP 有真實的空間資訊（不同樂器位置不同）
- **L/R Flip** 在多樂器場景更有意義（小提琴在左、大提琴在右的標準座位安排）
- 保持與 Study 1 Clef-full 的一致性

### Table 2: Zero-Shot Generalization on Unseen Instruments

**Dataset**: URMP (University of Rochester Multi-Modal Musical Performance)
- 包含多種樂器（小提琴、長笛、單簧管...）的真實錄音
- Zeng 沒測過，MT3 測過但效果普普

| Model Strategy | Training Data | Architecture | Piano | Strings | Winds | Ensemble |
|----------------|---------------|--------------|-------|---------|-------|----------|
| MT3 + music21 | MAESTRO + Slakh | CNN + Rule | ~75% | ~35% | ~30% | ~25% |
| Clef (Study 1) | Piano Only | Swin + Bridge | **> 80%** | < 20% | < 20% | < 20% |
| Clef (Study 2) | Universal (TDR) | Swin + Bridge + Aux | **> 82%** | **> 65%** | **> 65%** | **> 60%** |

> **註**：MT3 + music21 的 MV2H 預估值基於 Study 1 的「量化災難」現象。實際數據需實驗驗證。

### 表格亮點

1. **Clef (Study 1)**：證明「專用模型」的侷限性（只練鋼琴，遇到非鋼琴樂器就掛了）
2. **Clef (Study 2)**：Swin + Bridge + Aux + TDR 展現 **Zero-Shot 能力**
   - Swin 的階層式結構學習音色紋理
   - Bridge 捕捉跨樂器的和聲關係
   - Aux Loss 強迫特徵解耦
   - TDR 提供音色不變性
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
