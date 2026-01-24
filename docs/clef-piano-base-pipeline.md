# clef-piano-base Data Pipeline

> 本文件完整比較 Zeng et al. (2024) 的 pipeline 與 clef-piano-base 的 pipeline，
> 明確指出哪些步驟維持一致、哪些步驟有修改及修改原因。

## 設計原則

- **clef-piano-base**: 公平比較 Zeng，修正明顯的 bug，但不增加額外優勢
- **clef-piano-full**: 展示完整能力，使用自己的字典和更豐富的音樂資訊

---

## Phase 1: Raw Data → Kern

### 資料來源

| 資料集 | 原始格式 | 數量 | 來源 |
|--------|----------|------|------|
| MuseSyn | MusicXML | 210 | Zenodo (restricted) |
| HumSyn | Kern | 471 | GitHub (6 repos) |

### Step 1.1: MuseSyn XML → Kern

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **工具** | `verovio -f musicxml-hum` | `music21` → `sanitize_score()` → `converter21` | ⚠️ 不同 |
| **成功率** | 170/210 (81%) | 待驗證 (預期 ~100%) | ⚠️ 不同 |
| **Token 清理** | 無 | `clean_kern_sequence()` | ⚠️ 新增 |

**修改原因**:
1. verovio 有 40 個檔案轉換失敗，導致 train/test/valid 資料不完整
2. converter21 需要 `sanitize_score()` 處理 MusicXML 的各種問題
3. converter21 輸出需要 `clean_kern_sequence()` 移除 visual layout tokens

**公平性說明**:
- `clean_kern_sequence()` 只移除 Zeng 字典不包含的 visual tokens (rGG→r, /\→remove)
- 音樂語意 (pitch, duration, voice) 完全保留
- 最終進入模型的 token sequence 語意等價

### Step 1.2: HumSyn Kern 處理

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **Chopin filter** | `selected_chopin.txt` | 同 | ✅ 相同 |
| **Joplin reformat** | `hum2xml` → `verovio` | `clean_kern_sequence()` | ⚠️ 不同工具 |
| **其他 HumSyn** | 直接使用 | `clean_kern_sequence()` | ⚠️ 新增清理 |

**Chopin filter**: 使用 Zeng 的 `selected_chopin.txt`，排除未選中的 Chopin 曲目。

**Zeng 明確排除的曲目**:
- `joplin#school.krn` - 程式碼明確刪除

**Joplin reformat 修改原因**:
- Zeng 用 `hum2xml` → `verovio` 來正規化 Joplin 的 Kern 格式
- 原始 Joplin Kern 包含額外資訊：`**dynam` spine、repeat markers、stem directions (`>/<`)、accents (`^^`)、rest positions (`rGG`)
- clef 改用 `clean_kern_sequence()` 統一處理，避免依賴 humextra 和 verovio 工具鏈
- 需要在 `clean_kern_sequence()` 補充處理 Joplin 特有的格式：
  - 移除 `**dynam` spine
  - 移除 repeat expansion markers (`*>[...]`, `*>norep[...]`)
  - 移除 stem directions (`>`, `<`)
  - 移除 accent marks (`^^`)

**其他 HumSyn**: 統一用 `clean_kern_sequence()` 確保格式一致。

---

## Phase 2: Kern → Training Samples

### Step 2.1: Staff 分離

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **工具** | `extractx -s 1/2` | `extract_staff_from_kern()` | ⚠️ 不同工具 |
| **邏輯** | 按 spine 編號分離 | 按 `*staff1/*staff2` 標記分離 | ✅ 語意相同 |

**說明**: 兩者都是從 2-spine piano kern 分離出 upper/lower staff，只是實作方式不同。

### Step 2.2: Kern.clean()

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **移除裝飾符號** | ✅ `[pTtMmWwS$O:]` | ✅ 同 | ✅ 相同 |
| **移除 slurs/beaming** | ✅ `[(){}JKkL\\/]` | ✅ 同 | ✅ 相同 |
| **移除 grace notes** | ✅ 移除 `[qQP]` 行 | ❌ **保留** | ⚠️ **不同** |

**Grace Notes 修改原因**:
- Zeng 移除 grace notes 導致 audio-label mismatch
- Audio 聽得到裝飾音，但 label 沒有對應 token
- 這是 Zeng pipeline 的 bug，不是應該複製的「特色」
- clef-piano-base 修正此問題，需要擴充字典包含 grace note tokens

### Step 2.3: Chunk 切分

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **方式** | 5-bar chunks | **Full song** | ⚠️ **不同** |
| **Stride (train)** | 2 | N/A | ⚠️ **不同** |
| **Stride (test)** | 5 | N/A | ⚠️ **不同** |

**修改原因**:
- clef 的模型架構是 full-song encoder-decoder，不是 chunk-based
- 這是模型架構的根本差異，不是 preprocessing 的選擇

### Step 2.4: tiefix

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **使用** | ✅ `humextra/bin/tiefix` | ❌ 跳過 | ⚠️ 不同 |

**修改原因**:
- converter21 產生正確的 tie markers，不需要 tiefix
- verovio 有時產生不完整的 ties，需要 tiefix 修復

### Step 2.5: Voice 處理

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **process_voices()** | ✅ 合併 sub-spines | ✅ 同 | ✅ 相同 |
| **sort_chords()** | ✅ 排序 chord 內音符 | ❌ 跳過 | ⚠️ 不同 |
| **sort_voices()** | ✅ 排序 sub-spines | ❌ 跳過 | ⚠️ 不同 |

**修改原因**:
- converter21 已經產生排序好的輸出
- 跳過 sort 不影響最終 token sequence

### Step 2.6: tosequence() + encode()

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **tosequence()** | ✅ | ✅ 同 | ✅ 相同 |
| **LabelsMultiple.encode()** | ✅ Zeng 字典 | ✅ Zeng 字典 + grace notes | ⚠️ 字典擴充 |

---

## Phase 3: Audio Synthesis (Data Augmentation)

### Step 3.1: Kern → MIDI

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **工具** | `verovio` 或 `VirtuosoNet` (EPR) | 待定 | ❓ 待確認 |
| **EPR styles** | 隨機選 composer | 待定 | ❓ 待確認 |

### Step 3.2: Transpose (Data Augmentation)

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **層級** | Per-chunk | **Per-song** | ⚠️ **不同** |
| **Train** | 隨機 transpose | 同 | ✅ 相同 |
| **Test** | 不 transpose | 同 | ✅ 相同 |

**修改原因**:
- clef 使用 full-song，所以 augmentation 在 song level
- 維持相同的 augmentation 數量和範圍

### Step 3.3: MIDI → Audio

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **Synthesizer** | FluidSynth | 同 | ✅ 相同 |
| **Soundfonts** | 4-5 種隨機選 | 同 | ✅ 相同 |
| **Loudness norm** | `pyloudnorm` | 同 | ✅ 相同 |

### Step 3.4: Audio → Spectrogram

| | Zeng | clef-piano-base | 差異 |
|--|------|-----------------|------|
| **Feature** | VQT | **Log-Mel** | ⚠️ **不同** |
| **Sample rate** | 16kHz | 待定 | ❓ 待確認 |

**修改原因**:
- clef 使用 Log-Mel spectrogram（業界標準，Whisper 也用）
- 這是模型設計的選擇，不影響公平性

---

## Phase 4: Vocabulary

### Zeng 的 LabelsMultiple

```python
# 基本 tokens
durations = ["1","1.","2","2.","4","4.","8","8.","16","16.","32","32.","64","64.","3","6","12","24","48","96"]
extended = ["128","20","40","176","112"]
pitches = ['C','D','E','F','G','A','B','c','d','e','f','g','a','b'] + octave variants
accidentals = ['#', '-', 'n']
ties = ['[', ']', '_']
special = ['<sos>', '<eos>', '<pad>', 'r', '=']
```

### clef-piano-base 字典擴充

```python
# 新增 grace note tokens
grace_notes = ['q', 'Q', 'P']  # 加在 duration prefix
# 例如: 8qc, 16Qd, 32Pe
```

### clef-piano-full 字典

```python
# 完整字典，包含:
# - 所有 Zeng tokens
# - Grace notes
# - Dynamics (p, f, mp, mf, cresc, dim, etc.)
# - Articulations (staccato, accent, etc.)
# - 更多...
```

---

## 總結：差異清單

### 必須相同（公平比較的核心）

| 項目 | 說明 |
|------|------|
| Train/Test/Valid split | 使用 Zeng 的 split files |
| Data augmentation 數量 | 維持相同的 augmentation 倍數 |
| Soundfonts | 使用相同的 soundfont 集合 |
| Transpose 範圍 | 使用相同的 transpose 邏輯 |
| 基礎 vocabulary | Zeng 的 LabelsMultiple（擴充但不刪減） |

### 允許不同（模型架構差異）

| 項目 | Zeng | clef | 原因 |
|------|------|------|------|
| Chunk vs Full-song | 5-bar | Full-song | 模型架構 |
| Spectrogram | VQT | Log-Mel | 模型設計 |
| Augmentation 層級 | Per-chunk | Per-song | 配合模型架構 |

### 修正的 Bug

| 項目 | Zeng 的問題 | clef 的修正 |
|------|-------------|-------------|
| Grace notes | Audio 有但 label 沒有 | 保留 grace notes，擴充字典 |
| MuseSyn 轉換失敗 | 40/210 失敗 | 用 converter21 確保全部成功 |

### 工具鏈簡化

| 項目 | Zeng | clef | 原因 |
|------|------|------|------|
| Joplin reformat | `hum2xml` → `verovio` | `clean_kern_sequence()` | 避免依賴 humextra/verovio |
| HumSyn 清理 | 無 | `clean_kern_sequence()` | 統一處理所有來源 |

---

## Random Seeds

為確保可重複性並公平比較 Zeng，使用相同的 seed 設定：

| 階段 | Seed | 說明 |
|------|------|------|
| Data Augmentation | `0` | 用於 transpose、soundfont 選擇等 |
| Training | `1234` | 用於 model init、dataloader shuffle 等 |

```python
from src.utils import set_seed, SEED_DATA_AUGMENTATION, SEED_TRAINING

# Data preprocessing
set_seed(SEED_DATA_AUGMENTATION)  # 0

# Training
set_seed(SEED_TRAINING)  # 1234
```

---

## 待確認事項

- [x] ~~Seed 設定~~ (已完成: `src/utils/seed.py`)
- [ ] converter21 轉換全部 210 個 MuseSyn 的成功率
- [ ] Grace note tokens 的具體格式（`8qc` 或 `q8c`？）
- [ ] EPR synthesis 的實作細節
- [ ] Log-Mel vs VQT 的參數對齊
- [ ] `clean_kern_sequence()` 補充 Joplin 特有格式處理：
  - [ ] 移除 `**dynam` spine
  - [ ] 移除 repeat expansion markers (`*>[...]`)
  - [ ] 移除 stem directions (`>`, `<`)
  - [ ] 移除 accent marks (`^^`)
- [ ] 檢查其他 HumSyn (beethoven, haydn, mozart, scarlatti) 是否需要額外清理

---

## 參考文件

- `piano-a2s/PIPELINE_COMPARISON.md` - converter21 vs verovio 詳細比較
- `piano-a2s/clean_kern.py` - Token 清理邏輯
- `piano-a2s/sanitize_piano_score.py` - MusicXML 修復邏輯
