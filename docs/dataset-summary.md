## clef-piano-base 資料統計

**重要！目前進展中**

**資料路徑**: `data/experiments/clef_piano_base/`

### Kern 檔案 (Phase 1 輸出)

| 來源 | 數量 | 備註 |
|------|------|------|
| humdrum_chopin_first_editions | 205 | Chopin 首版 |
| beethoven_piano_sonatas | 103 | Beethoven 鋼琴奏鳴曲 |
| mozart_piano_sonatas | 69 | Mozart 鋼琴奏鳴曲 |
| scarlatti_keyboard_sonatas | 65 | Scarlatti 鍵盤奏鳴曲 |
| joplin | 42 | Joplin Ragtime |
| haydn_piano_sonatas | 25 | Haydn 鋼琴奏鳴曲 |
| musesyn_* | 214 | MuseScore 流行歌 |
| **總計** | **723** | |

### 失敗的 Kern (無法生成 MIDI)

**記錄**: `src/datasets/syn/skip_files.txt`

| 來源 | 失敗數 | 原因 |
|------|--------|------|
| humdrum_chopin | 8 | offsetInScore 錯誤 (6) + tuplet 對齊錯誤 (2) |
| beethoven | 0 | - |
| mozart | 0 | - |
| **總計** | **8** | |

**失敗原因詳細**：
- **offsetInScore 錯誤 (6 首)**：原始 kern 檔案的 spine timing 問題，導致 negative delta time
  - 009-1-KI-003, 023-1-BH, 028_1-12-1a-C-005, 028_13-24-1a-C-013, 055-1-BH-002, 060-1-BH
- **tuplet 對齊錯誤 (2 首)**：`*tuplet` 和 `*Xtuplet` 標記未正確追蹤 spine 分裂/合併
  - 021-1a-BH-001, 021-1a-BH-002

**2026-01-28 更新**:
- `fix_kern_spine_timing` 改進：48 → 14（救回 34 首）
- 移除 Phase 1 的 `expand_tuplets_to_zeng_vocab`：14 → 5（再救回 9 首，含 Beethoven/Mozart 全部）
- 深入調查後發現額外 3 首有問題：5 → 8

**成功的 Kern**: 723 - 8 = **715**

### Split 分配

Split 檔案: `src/datasets/syn/test_split.txt`, `src/datasets/syn/valid_split.txt`

| Split | Kern 數 | 失敗數 | 成功 Kern | 音檔版本數 | 音檔數 |
|-------|--------|--------|-----------|------------|--------|
| Train | 591 | 7 | 584 | ×4 | 2336 |
| Valid | 66 | 0.5 | 65 | ×1 | 65 |
| Test | 66 | 0.5 | 66 | ×1 | 66 |
| **總計** | **723** | **8** | **715** | | **2467** |

### 實際音檔數量 (Phase 2 完成)

**2026-02-01 更新**：
- MIDI: **2467** (全部成功)
- Audio: **2467** (全部成功)
- Mel: **2467** (全部成功)
- Metadata entries: **2467** (全部欄位完備)
- Alignment mismatches: **0**

### 已知問題

- **Validation 已改用 ChunkedDataset**（2026-02-01 修正）：
  之前 valid set 直接用 ManifestDataset（完整 mel + tokens[:max_seq_len]），
  導致長曲目 OOM 風險且 mel-kern 對齊不一致。已修正為與 train 相同的 chunking。

### Pipeline 修正記錄 (2026-02-01)

1. **kern/ 不再 strip cue passages**（`humsyn_processor.py`）：
   `strip_cue_passages` 移至 Phase 1b 的 kern_gt/ 生成。
   原因：cue notes 被替換為 rests 後，converter21 對複雜 spine 結構（如 Chopin Op.13）
   計算 offsetInScore 出錯，導致 MIDI 寫入卡死。
2. **MetronomeMark number=None 容錯**（`extract_measure_times`）：
   converter21 將 `!!!OMD` 段落標題（如 "TRIO"）誤建為 MetronomeMark，導致 82 首
   `float(None)` 崩潰。修正：skip `qpm is None` 的標記，沿用前一個 tempo。
3. **空小節過濾**（`extract_kern_measures`）：
   barline 之間若只有註解/interpretation（如 `!!!OMD: TRIO` + `*k[]` + `*M3/4`），
   不算作小節。修正前 76 個 alignment mismatch，修正後剩 1 個。

### Pipeline 修正記錄 (2026-02-02)

4. **Final barline 處理**（`extract_kern_measures`）：
   `==` double barline 可能是 final barline（曲尾）或 section double barline（如 DaCapo 段落分界）。
   Pre-scan 找到檔案中最後一個 `==` barline，僅該處停止 measure tracking，其餘 `==` 視為一般 barline。
   修正 MuseScore tie resolution 問題（1 首）及 DaCapo 段落遺漏。Alignment mismatch: 1 → 0。
5. **Repeat 函式整併至 `src/score/expand_repeat.py`**：
   將 `sanitize_kern.py`、`musesyn_processor.py` 中的 repeat 相關函式統一移至 `expand_repeat.py`，
   消除重複程式碼（~600 行）。`sanitize_kern.py` 使用 lazy import 避免循環依賴。
6. **模型 `_init_weights`**（`model.py`）：
   新增 `trunc_normal_(std=0.02)` 初始化（GPT-2/BART 慣例），跳過 Swin pretrained weights。
   初始 loss 從 ~5.6 降至接近理論值 `ln(V)=5.49`。
7. **Gradient accumulation loss 報告修正**（`train.py`）：
   `accumulated_loss` 報告時未除以 `gradient_accumulation_steps`，導致 wandb/progress bar
   顯示的 train loss 為實際值的 N 倍。`accumulation_steps=1` 時不受影響。

### Tokenizer 特殊 Token（2026-02-02 更新）

| ID | Token | 用途 |
|----|-------|------|
| 0 | `<pad>` | Padding |
| 1 | `<sos>` | Start of sequence |
| 2 | `<eos>` | End of sequence |
| 3 | `<coc>` | Change of Column（spine 分隔） |
| 4 | `<bar>` | Barline（小節線） |
| 5 | `<continue>` | Chunk 邊界（曲目跨 chunk） |
| 6 | `<nl>` | Newline（同小節內換行） |
| 7 | `<split>` | Spine split（`*^`，聲部分裂） |
| 8 | `<merge>` | Spine merge（`*v`，聲部合併） |
| 9 | `<*>` | Null interpretation（split/merge 行中不變的 spine） |

`<nl>` 用於同一小節內多個 data line 的分隔（`<bar>` 已隱含換行，不需要額外 `<nl>`）。
`<split>`/`<merge>` 保留聲部分裂/合併的結構資訊，77% (560/723) 的 kern_gt 有此操作。

### Kern Reconstruction Round-trip 測試（2026-02-02 更新）

**測試腳本**：`tests/test_reconstruct_kern.py`
**測試範圍**：test set 66 unique kern_gt files
**流程**：kern_gt → tokenize → reconstruct_kern_from_tokens (with metadata injection) → converter21 parse → write

**第一輪（補 `<nl>`/`<split>`/`<merge>` 前）**：

| 路徑 | Parse | 輸出 | 成功率 |
|------|-------|------|--------|
| **A** (原始 kern_gt → MIDI) | 65/66 | 65/66 | **98.5%** |
| **B** (reconstructed → MIDI) | 62/66 | 58/66 | **87.9%** |
| **C** (reconstructed → MusicXML) | 62/66 | 61/66 | **92.4%** |

**第二輪（補 `<nl>`/`<split>`/`<merge>` 後）**：

| 路徑 | Parse | 輸出 | 成功率 |
|------|-------|------|--------|
| **A** (原始 kern_gt → MIDI) | 65/66 | 65/66 | **98.5%** |
| **B** (reconstructed → MIDI) | 65/66 | 61/66 | **92.4%** |
| **C** (reconstructed → MusicXML) | 65/66 | 64/66 | **97.0%** |

**結論**：`<nl>`/`<split>`/`<merge>` 修好了 3 首 Joplin parse 失敗。
Path C (MusicXML) 成功率 97%，建議 inference 時用 MusicXML 輸出。
**<unk> tokens: 0**（Zeng vocab 完整覆蓋）。

#### 剩餘失敗

**A 失敗（1 首，原始 kern_gt 即有問題）**：

| 檔案 | 錯誤 |
|------|------|
| `chopin_066-1axx-COM` | `cannot convert quarterLength 0.6875 exactly to type` |

**B/C 比 A 多失敗（reconstruct 引入的問題）**：

| 檔案 | B (MIDI) | C (MusicXML) | 原因 |
|------|----------|--------------|------|
| `beethoven_sonata11-1` | negative delta time | duplex-maxima duration too long | 重建後時值累積超出小節範圍 |
| `beethoven_sonata26-1` | negative delta time | OK | MIDI 更敏感，MusicXML 可容忍 |
| `chopin_070-1-MEIf-002` | negative delta time | OK | 同上 |
| `musesyn_Spectre` | negative delta time | OK | 同上 |

**失敗分類**：
- **negative delta time (4 首)**：tokenize round-trip 後小節內時值微小偏差，converter21 計算 offsetInScore 為負。MusicXML 路徑可容忍其中 3 首。
- **duplex-maxima (1 首)**：重建後某小節時值過長，music21 無法轉換為 MusicXML duration type。

### clef-piano-full 需要的 Pipeline 改動（規劃中）

相比 clef-piano-base（Zeng vocab），clef-piano-full 擴展 vocab 並保留更多音樂資訊。

#### Vocab 擴展（消除 lossy quantization）

| 新增 token 類別 | 範例 | 效果 |
|----------------|------|------|
| Dotted triplets | `3.`, `6.`, `12.`, `24.`, `48.`, `96.` | 不再需要 `12.→8` 等價轉換（消除符號歧義） |
| Breve/Longa | `0`, `00` | 不再需要 `split_breves_in_sequence`（`0c` → `[1c 1c]`） |
| Quintuplet | `10` | 不再需要 `expand_quintuplets`（Zeng 缺 `10`，hack 為 `[20cc 20cc]` tied） |
| Tuplet ratios | 常見 `X%Y` | 減少 `quantize_tuplet_ratios` 的 lossy quantization |
| Natural accidental | `n` | 不再需要 `strip_natural_accidentals`（`cn` → `c`） |
| Slur/Phrase | `(`, `)` | 不再需要 `remove_slur_phrase_markers` |
| Articulation/Ornament | `'`, `~`, `^`, `;`, `T`, `M`, `t`, `S` | `clean_kern_token(preserve_articulation=True)` |

#### Spine 保留

- `keep_dynam=True`：保留 `**dynam` spine（`pp`, `ff`, `cresc.` 等）
- 抽取有意義的 `**text` spine 內容（表情術語如 "dolce", "espressivo"）
- 目前 `strip_non_kern_spines` 已支援 `keep_dynam` 參數

#### Layout Comments Tokenization

- `!!LO:TX:t=dolce`（文字表情）、`!!LO:HP`（hairpin）、`!!LO:DY`（動態位置）
- 這些包含豐富的音樂表情資訊，clef-piano-full 應 tokenize 而非丟棄

#### 安全移除（clef-piano-full 也不需要）

| 項目 | 原因 |
|------|------|
| Editorial markers (`X`, `y`, `z`, `N`, `?`) | 版本考證用，無演奏語意 |
| `*rscale:` regions | 標記 cue passage 時值倍率，cue 已被 strip 後失去意義 |
| Cue passages (`*cue...*Xcue`) | 不發聲，非該聲部的音 |
| Rest position (`rGG`) | 純排版，可從 staff/clef 推導 |

### 音檔命名規則

- **Train**: `{kern_stem}_v{N}~{soundfont}.wav` (N=0,1,2,3)
- **Valid/Test**: `{kern_stem}~{soundfont}.wav` (無 `_v{N}`)

### Soundfonts

| 版本 | Soundfont |
|------|-----------|
| v0 | TimGM6mb |
| v1 | FluidR3_GM |
| v2 | UprightPianoKW-20220221 |
| v3 | SalamanderGrandPiano-V3+20200602 |