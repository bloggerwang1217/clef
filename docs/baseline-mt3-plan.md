# Plan: Baseline 驗證 — MT3 + music21 on ASAP & URMP

## 目標

驗證 Pipeline baseline (MT3 + music21) 的 MV2H 表現：

| Study | Dataset | 預期 MV2H | 目的 |
|-------|---------|----------|------|
| Study 1 | ASAP (Piano) | ~58% | 證明「量化災難」 |
| Study 2 | URMP (Multi-instrument) | ~25-35% | 證明 Pipeline 在非鋼琴樂器上更慘 |

---

## 背景分析：Zeng et al. (2024) 的實驗設定

### ASAP Dataset 使用方式
| Split | Zeng 用量 | ASAP 全集 |
|-------|----------|-----------|
| Train | 14 首 / 58 段 | - |
| Test | 25 首 / 80 段 | 236 首 / 1,067 段 |

### 關鍵發現
Zeng 的 split 檔案位於：
- `data_processing/metadata/train_asap.txt` (14 首)
- `data_processing/metadata/test_asap.txt` (25 首)

### Test Split (25 pieces, ~80 recordings)
| Composer | Pieces |
|----------|--------|
| Bach | BWV 875, 891 |
| Beethoven | Sonata 9/1, 21/1, 22/1, 27/1, 28/1 |
| Chopin | Ballade 2,3; Etude Op.10 No.2,4,5; Sonata 2/4 |
| Haydn | Sonata 50/1 |
| Liszt | Concert Etude S145/1, Paganini 6 |
| Mozart | Sonata 12/1, 12/3 |
| Schubert | Impromptu D.899 No.1,2,4; Moment Musical 1; D.664/3, D.894/2 |
| Schumann | Toccata |

### 建議策略
**使用 Zeng 完全相同的 test split**，理由：
1. 公平比較，審稿者無話可說
2. 可以直接引用 Zeng 的數據作為對照
3. Split 檔案已公開，可完全複製

---

## 實驗設計（Colab + 本地混合版）

### Step 1: 選擇 MT3 實作版本

| 版本 | 框架 | 學術可信度 | 說明 |
|------|------|-----------|------|
| [magenta/mt3](https://github.com/magenta/mt3) | T5X (JAX) | ⭐⭐⭐ 最高 | **官方版，論文可直接引用** |
| [mimbres/YourMT3](https://github.com/mimbres/YourMT3) | PyTorch | ⭐⭐ | 2024 改良版，審稿者可能質疑 |
| [gudgud96/MR-MT3](https://github.com/gudgud96/MR-MT3) | PyTorch | ⭐⭐ | 第三方實作，結果可能不同 |

**決定：使用官方 Google MT3**
- 論文寫法：`We use the official MT3 implementation (Gardner et al., 2022)`
- 審稿者無話可說

### Step 2: 混合策略（Colab + 本地）

由於官方 MT3 使用 T5X/JAX（安裝複雜），採用混合策略：

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   Google Colab (免費)    │     │      本地環境           │
│                         │     │                         │
│  MT3 官方 notebook      │ ──► │  music21 量化           │
│  輸出 MIDI 檔案         │     │  MV2H 評估              │
│                         │     │                         │
│  ~124 檔案 × 2 min      │     │  不需 GPU               │
│  ≈ 4 小時（免費額度內）  │     │  完全離線               │
└─────────────────────────┘     └─────────────────────────┘
```

### Step 2.1: Colab 批次推論
```python
# 使用官方 Colab notebook
# https://colab.research.google.com/github/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb

# 修改成批次處理：
import os
from google.colab import drive
drive.mount('/content/drive')

input_dir = '/content/drive/MyDrive/clef_baseline/audio/'
output_dir = '/content/drive/MyDrive/clef_baseline/midi/'

for audio_file in os.listdir(input_dir):
    if audio_file.endswith('.wav'):
        # 使用 MT3 transcribe function (notebook 內已定義)
        midi = transcribe(os.path.join(input_dir, audio_file))
        midi.save(os.path.join(output_dir, audio_file.replace('.wav', '.mid')))
```

### Step 2.2: 本地 music21 + MV2H
```bash
# 本地環境只需要 music21 和 MV2H
conda create -n baseline python=3.10
conda activate baseline

pip install music21
# MV2H 安裝見 Step 5
```

### Step 3: 準備資料集

#### 3.1 ASAP Dataset (Study 1 - Piano)
```bash
# Clone ASAP dataset
git clone https://github.com/fosfrancesco/asap-dataset data/asap-dataset

# 下載 Zeng 的 split 檔案
mkdir -p data/splits
wget -O data/splits/test_asap.txt \
  https://raw.githubusercontent.com/wei-zeng98/piano-a2s/main/data_processing/metadata/test_asap.txt
wget -O data/splits/train_asap.txt \
  https://raw.githubusercontent.com/wei-zeng98/piano-a2s/main/data_processing/metadata/train_asap.txt
```

#### 3.2 URMP Dataset (Study 2 - Multi-instrument)
```bash
# URMP: University of Rochester Multi-Modal Musical Performance Dataset
# 下載連結：http://www2.ece.rochester.edu/projects/air/projects/URMP.html

# 資料集結構
# - 44 pieces (duets, trios, quartets, quintets)
# - 樂器：小提琴、中提琴、大提琴、低音提琴、長笛、雙簧管、單簧管、薩克斯風、巴松管、小號、法國號、長號、大號
# - 每首有：混音音訊、分軌音訊、MIDI、樂譜

mkdir -p data/urmp
# 需手動下載或使用學術申請
```

#### 資料集規模總覽
| Dataset | 檔案數 | 用途 |
|---------|--------|------|
| ASAP (Zeng split) | ~80 段錄音 | Study 1 |
| URMP | 44 首 | Study 2 |
| **總計** | **~124 個檔案** | 需跑 MT3 + music21 |

### Step 4: Pipeline 流程

```
Audio (WAV)
    ↓ MT3 (Colab)
MIDI (predicted)
    ↓ 下載到本地
    ↓ music21.quantize()
MusicXML (quantized)
    ↓ MV2H evaluation
Score vs Ground Truth
```

#### 4.1 Colab 批次 MT3 推論
使用修改版的官方 notebook：`notebooks/mt3_batch_transcribe.ipynb`

詳見 Step 2.1 的批次處理程式碼。

#### 4.2 批次 music21 量化
```python
# experiments/asap_baseline/batch_quantize.py
import music21
from pathlib import Path

def batch_quantize(input_dir: Path, output_dir: Path):
    """Batch quantize all MIDI files using music21."""
    output_dir.mkdir(parents=True, exist_ok=True)

    midi_files = list(input_dir.glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files")

    for midi_path in midi_files:
        xml_path = output_dir / f"{midi_path.stem}.musicxml"
        if xml_path.exists():
            print(f"Skipping {midi_path.name} (already exists)")
            continue

        print(f"Quantizing {midi_path.name}...")
        try:
            score = music21.converter.parse(str(midi_path))
            # This is where "quantization disaster" happens
            quantized = score.quantize()
            quantized.write('musicxml', fp=str(xml_path))
        except Exception as e:
            print(f"Error processing {midi_path.name}: {e}")

if __name__ == "__main__":
    batch_quantize(
        input_dir=Path("results/mt3_midi"),
        output_dir=Path("results/mt3_music21_xml")
    )
```

#### 4.3 批次 MV2H 評估
```bash
# 使用 McLeod 的 MV2H 工具
# https://github.com/apmcleod/MV2H

# 安裝 MV2H (需要 Java)
git clone https://github.com/apmcleod/MV2H.git
cd MV2H && mvn package

# 批次評估
for xml in results/mt3_music21_xml/*.musicxml; do
    name=$(basename "$xml" .musicxml)
    gt="data/asap-dataset/**/score.xml"  # 需要對應的 ground truth
    java -jar MV2H/target/MV2H.jar -g "$gt" -t "$xml" >> results/mv2h_scores.txt
done
```

### Step 5: 評估指標
與 Zeng 一致（使用 **Non-aligned MV2H**）：
- **$F_p$** (Multi-pitch detection)
- **$F_{voi}$** (Voice separation)
- **$F_{val}$** (Note value detection)
- **$F_{harm}$** (Harmonic detection)
- **$F_{MV2H}$** = average of above

---

## Repo 結構

```
clef/                            # 主專案
├── notebooks/
│   └── mt3_batch_transcribe.ipynb  # Colab notebook (上傳到 Colab 執行)
├── scripts/
│   ├── batch_quantize.py        # 本地 music21 量化
│   └── batch_evaluate.sh        # 本地 MV2H 評估
├── data/
│   ├── asap-dataset/            # git submodule
│   ├── urmp/                    # URMP dataset
│   └── splits/
│       ├── train_asap.txt       # Zeng's train split
│       └── test_asap.txt        # Zeng's test split
├── results/
│   ├── mt3_midi/                # MT3 輸出的 MIDI (從 Colab 下載)
│   ├── mt3_music21_xml/         # music21 量化後的 MusicXML
│   └── mv2h_scores/             # MV2H 評估結果
└── docs/
    └── baseline-mt3-plan.md     # 本文件
```

---

## 預期結果

### Study 1: ASAP (Piano)

| System | $F_{MV2H}$ | $F_p$ | $F_{harm}$ | 問題 |
|--------|-----------|-------|-----------|------|
| MT3 + music21 | ~58% | ~80% | ~40% | 量化災難 |
| Zeng (2024) | 74.2% | 63.3% | 54.5% | CNN 聽覺失聰 |

### Study 2: URMP (Multi-instrument)

| Instrument | MT3 + music21 (預估) | 問題 |
|------------|---------------------|------|
| Piano | ~75% | 量化災難（同 Study 1） |
| Strings | ~35% | 音高偏移 + 量化災難 |
| Winds | ~30% | 泛音複雜 + 量化災難 |
| Ensemble | ~25% | 多聲部混亂 + 量化災難 |

### 論證邏輯

如果 MT3 + music21 的 MV2H：
- **Study 1 (ASAP) < 60%**：證明 Pipeline 在鋼琴上就不行
- **Study 2 (URMP) < 40%**：證明 Pipeline 在其他樂器上更慘

> **Pipeline 方法論本身有根本缺陷，即使 MT3 音高準（~80% $F_p$），量化後和聲結構全毀（~40% $F_{harm}$）**

---

## 執行策略

### Phase 1: Pilot（驗證 pipeline）
- 先跑 ASAP 5 首（~20 段錄音）
- 確認整個流程跑得通
- 預估完整實驗時間

### Phase 2: Study 1 Full Run (ASAP)
- 跑完全部 25 首（~80 段錄音）
- 收集完整 MV2H 數據
- 產出 Table 1 數據

### Phase 3: Study 2 Full Run (URMP)
- 跑完全部 44 首
- 分類統計：Piano / Strings / Winds / Ensemble
- 產出 Table 2 數據

---

## 決策摘要

| 問題 | 決策 |
|------|------|
| MT3 推論 | **Google Colab**（官方 notebook，免費額度足夠） |
| MT3 實作 | **官方 magenta/mt3**（學術可信度最高） |
| music21 量化 | **本地執行**，先用預設參數 |
| MV2H 版本 | **Non-aligned**（與 Zeng 一致） |
| Test split | **Zeng 的 25 首 split** |
| 成本估計 | **免費 ~ $10**（Colab Free/Pro） |

---

## 硬體需求

### Colab (MT3 推論)
| 資源 | Free Tier | Colab Pro |
|------|-----------|-----------|
| GPU | T4 (運氣) | V100/A100 |
| 時間 | ~4-5 hr | ~1 hr |
| 成本 | $0 | $10/月 |

### 本地 (music21 + MV2H)
| 資源 | 需求 |
|------|------|
| GPU | 不需要 |
| RAM | 8GB+ |
| 磁碟 | 20GB（含資料集） |

> **總時間估計**：Colab ~4 hr + 本地 ~1 hr = **~5 小時**

---

## 風險與備案

| 風險 | 備案 |
|------|------|
| Colab 斷線 | Notebook 有 skip 邏輯，重跑即可續傳 |
| Colab 免費額度不夠 | 買 Colab Pro ($10)，或分多天跑 |
| GPU 記憶體不足 | 降低 batch_size（預設 8） |
| MV2H 安裝問題 | 用 Python 版本 `mir_eval` |
| 結果比預期好 | 檢查是否用錯參數，或重新評估假設 |

---

## 工作流程總覽

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 準備資料                                                     │
│     - Clone ASAP dataset                                        │
│     - 下載 URMP dataset                                         │
│     - 上傳音訊檔到 Google Drive                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Colab 執行 MT3                                               │
│     - 開啟 notebooks/mt3_batch_transcribe.ipynb                  │
│     - 批次轉錄所有音訊                                           │
│     - MIDI 檔案存回 Google Drive                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 本地執行 music21 + MV2H                                      │
│     - 下載 MIDI 到 results/mt3_midi/                             │
│     - 執行 scripts/batch_quantize.py                             │
│     - 執行 scripts/batch_evaluate.sh                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 分析結果                                                     │
│     - 彙整 MV2H 分數                                             │
│     - 產出 Table 1 & Table 2                                     │
│     - 更新 docs/experiment-results.md                            │
└─────────────────────────────────────────────────────────────────┘
```
