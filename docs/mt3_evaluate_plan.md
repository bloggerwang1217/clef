# MT3 Baseline MV2H 評估計畫

## 目標

在 clef repo 實作 MT3 baseline 的 MV2H 評估，支援：
1. **Full Song 評估** — 整首曲子評估一次
2. **5-bar Chunk 評估** — 與 Zeng 相同粒度，apple-to-apple comparison

---

## 現有資源

### Clef Repo
```
/home/bloggerwang/clef/
├── src/inference/mt3_to_musicxml.py    # 待移動並改名
├── data/experiments/mt3/
│   ├── full_midi/                       # MT3 原始輸出（單軌、未量化）
│   └── full_musicxml/                   # 量化 + 分手後 MusicXML（80 個檔案）
```

### Piano-A2S Repo（參考）
```
/home/bloggerwang/piano-a2s/
├── evaluate.py                          # Zeng 的評估腳本
├── evaluate_midi_mv2h.sh                # MV2H 呼叫腳本
├── zeng_test_chunk_set.csv              # Chunk 邊界資訊（9363 chunks）
└── MV2H/bin/                            # 編譯好的 MV2H Java
```

---

## 實作計畫

### Step 1: 移動並重構 mt3_to_musicxml.py → mt3_evaluate.py

**操作**：
```bash
mv /home/bloggerwang/clef/src/inference/mt3_to_musicxml.py \
   /home/bloggerwang/clef/src/evaluation/mt3_evaluate.py
```

**修改內容**：
1. 同時輸出 MusicXML + MIDI（量化後）
2. 新增 Full Song MV2H 評估功能
3. 新增 5-bar Chunk 評估功能

```python
# 核心功能整合到一個檔案
def convert_and_evaluate(
    pred_midi_path: str,       # MT3 原始 MIDI
    gt_path: str,              # ASAP ground truth (xml_score.musicxml 或 midi_score.mid)
    output_dir: str,
    mv2h_bin: str,
    mode: str = 'full'         # 'full' 或 'chunks'
) -> dict:
    """
    完整評估流程：
    1. MT3 MIDI → 量化 + 分手 → MusicXML + MIDI
    2. GT → MIDI (如果是 MusicXML)
    3. MV2H 評估
    """
    ...
```

---

### Step 2: 安裝/連結 MV2H

**選項 A**：直接使用 piano-a2s 已編譯的 MV2H
```python
MV2H_BIN = "/home/bloggerwang/piano-a2s/MV2H/bin"
```

**選項 B**：安裝到 Clef Repo
```bash
cd /home/bloggerwang/clef
git clone https://github.com/apmcleod/MV2H.git
cd MV2H && make && cd ..
```

---

### Step 3: 複製 evaluate_midi_mv2h.sh

```bash
cp /home/bloggerwang/piano-a2s/evaluate_midi_mv2h.sh \
   /home/bloggerwang/clef/src/evaluation/
```

---

## 檔案結構（預計）

```
/home/bloggerwang/clef/
├── src/
│   ├── inference/
│   │   └── (mt3_to_musicxml.py 已移走)
│   └── evaluation/
│       ├── mt3_evaluate.py              # [移動+重構] 核心評估腳本
│       └── evaluate_midi_mv2h.sh        # [複製] MV2H 呼叫腳本
├── data/experiments/mt3/
│   ├── full_midi/                       # MT3 原始輸出
│   ├── full_musicxml/                   # 量化後 MusicXML
│   ├── full_midi_quantized/             # [新增] 量化後 MIDI
│   ├── ground_truth_midi/               # [新增] ASAP GT 的 MIDI
│   └── results/
│       ├── full_song/                   # [新增] Full song 評估結果
│       └── chunks/                      # [新增] Chunk 評估結果
└── MV2H/                                # [選項 B] MV2H 工具
```

---

## 已確認事項

1. **Ground Truth 來源**：✅ 已確認
   - 使用 ASAP 的 `xml_score.musicxml` 或 `midi_score.mid`
   - 路徑範例：`asap-dataset/Bach/Prelude/bwv_875/xml_score.musicxml`

2. **Measure Number**：✅ 已確認
   - MT3 MusicXML：從 1 開始（music21 預設）
   - ASAP MusicXML：需要檢查，但應該也是從 1 開始

3. **重要提醒**：
   - MT3 原始 MIDI 是**未量化**的，不能直接用於 MV2H 評估
   - 必須用 `mt3_evaluate.py` 輸出的**量化後 MIDI**

---

## 驗證方法

1. **Full Song 評估**：
   ```bash
   python src/evaluation/mt3_evaluate.py \
       --mode full \
       --pred_dir data/experiments/mt3/full_midi \
       --gt_dir /home/bloggerwang/asap-dataset \
       --mv2h_bin MV2H/bin \
       --output data/experiments/mt3/results/full_song.csv
   ```

2. **Chunk 評估**：
   ```bash
   python src/evaluation/mt3_evaluate.py \
       --mode chunks \
       --pred_dir data/experiments/mt3/full_midi \
       --gt_dir /home/bloggerwang/asap-dataset \
       --chunk_csv /home/bloggerwang/piano-a2s/zeng_test_chunk_set.csv \
       --mv2h_bin MV2H/bin \
       --output data/experiments/mt3/results/chunks.csv
   ```

3. **與 Zeng 結果比較**：
   - Zeng MV2H = 74.2%
   - 預期 MT3 + music21 MV2H < 60%（Straw Man baseline）

---

## MV2H 評估流程說明

```
┌─────────────────────────────────────────────────────────────────┐
│  evaluate_midi_mv2h.sh                                          │
│                                                                  │
│  # Step 1: 轉換 MIDI → MV2H 中間格式                            │
│  java -cp $MV2H_BIN mv2h.tools.Converter -i target.mid          │
│  java -cp $MV2H_BIN mv2h.tools.Converter -i pred.mid            │
│                                                                  │
│  # Step 2: 評估（使用 DTW 對齊）                                │
│  java -cp $MV2H_BIN mv2h.Main -g target.conv.txt -t pred.conv.txt -a │
│                                                                  │
│  # 輸出 6 個指標：                                               │
│  Multi-pitch: 0.633  ← 音高準確度                               │
│  Voice: 0.884        ← 聲部分離準確度                           │
│  Meter: 0.950        ← 拍子結構準確度                           │
│  Value: 0.907        ← 音值準確度                               │
│  Harmony: 0.545      ← 和聲結構準確度                           │
│  MV2H: 0.742         ← 綜合分數（上述 5 項平均）                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 量化設定說明

```python
# mt3_evaluate.py 中的量化設定

class QuantizationConfig:
    # 量化網格：16分音符 + 8分音符三連音
    QUARTER_LENGTH_DIVISORS: Tuple[int, ...] = (4, 3)

    # 學術參考：music21 default settings
    # https://www.music21.org/music21docs/moduleReference/moduleMidiTranslate.html

class HandSeparationConfig:
    # 分手切分點：Middle C (MIDI 60)
    SPLIT_POINT: int = 60

    # 學術參考：Hadjakos et al. "Detecting Hands from Piano MIDI Data" (2019)
    # 這是「工業界預設」的 naive baseline
```
