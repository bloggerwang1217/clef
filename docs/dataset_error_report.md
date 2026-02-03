# ASAP 錯誤檔案分析報告

---

## 🟢 全局狀態總覽

### Pipeline 架構

```
MusicXML ──▶ Full Score Kern ──▶ Upper/Lower Kern ──▶ PKL (Zeng)
            (階段 1)            (階段 2)             (階段 3)
            converter21         譜表分離+切分         編碼
```

### 階段 1 處理狀態

| Split | 成功率 | 失敗原因 |
|-------|--------|----------|
| **Test Set** | **25/25 曲目 (100%)** ✅ | 無 |
| **Training Set** | 24/25 曲目 (96%) | converter21 v4.0.0 bug |

### ⚠️ 唯一剩餘的階段 1 錯誤

| 曲目 | Split | 錯誤類型 | 說明 |
|------|-------|----------|------|
| Chopin#Scherzos#31 | Training | `list assignment index out of range` | converter21 v4.0.0 已知 bug |

### Performance 層級統計

```
                    ┌───────────────────┐
                    │   161 performances │ (Test Set 總數)
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐         ┌───────────▼───────────┐
    │ 階段 1 失敗        │         │ 階段 1 成功            │
    │ 0 performances    │         │ 161 performances      │
    │ (0 曲目) ✅        │         │ (25 曲目) ✅           │
    └───────────────────┘         └─────────┬─────────────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                          │                                   │
              ┌───────────▼───────────┐         ┌─────────────▼─────────────┐
              │ 無音檔（非 MAESTRO）   │         │ 有音檔（MAESTRO）          │
              │ → 資料集限制          │         │ → ✅ 可用於 Zeng          │
              └───────────────────────┘         └───────────────────────────┘
```

### 「無音檔」說明

ASAP 資料集包含兩種 performances：
- **MAESTRO 子集**（519 個）：有音檔，來自 MAESTRO 鋼琴錄音資料集
- **非 MAESTRO**（548 個）：只有 MIDI + 對齊標註，無錄音

非 MAESTRO 的 performances 是 ASAP 作者手動對齊的額外資料，本來就沒有音檔。
這不是 pipeline 錯誤，是資料集設計的限制。

---

## 處理結果統計（2026-01-17 run_asap_new.sh）

### 生成結果

| 類型 | 數量 |
|------|------|
| Test PKL | 9,363 個 |
| Train PKL | 5,613 個 |
| [ERROR] | 1 個 |
| [SKIP] | ~8,269 個（正常過濾） |

### [SKIP] 類型說明

這些是 Zeng pipeline 後續處理階段的正常過濾，不是 converter21 錯誤：

| 原因 | 說明 |
|------|------|
| Audio duration out of range | Zeng 要求 chunk 長度 4-12 秒 |
| process_voices() failed | 解析 dotted duration 如 `4.` 時的 int() 錯誤 |
| kern.clean() failed | clean_kern 未處理的 notation |
| tosequence() returned None | 序列化失敗（可能是 vocab 不支援的 token） |
| Wrong measure count | 上下譜表小節數對不上 |

---

## 歷史問題分析

### converter21 v4.0.0 Index Bug（無法修復）

#### 檔案資訊
- **曲目**: Chopin - Scherzo No. 3, Op. 39
- **檔案路徑**: `/home/bloggerwang/asap-dataset/Chopin/Scherzos/31/xml_score.musicxml`
- **錯誤類型**: `list assignment index out of range`
- **錯誤位置**: converter21 `m21utilities.py:4542`

#### Bug 本質

這是一個在 **converter21 v4.0.0** 中發現的索引越界錯誤，屬於新版本引入的迴歸性錯誤。

#### 問題根源

錯誤發生在 MusicXML 到 Kern 轉換過程中的「小節平衡」機制：

1. **固定長度列表**: `partMeasures` 在處理前預先建立，長度基於原始 MusicXML 各聲部的小節總數
2. **動態插入邏輯**: 當偵測到某個 Part 的小節數少於最大值時，會嘗試插入空白小節
3. **索引越界**: 插入新小節到 music21 Stream 後，嘗試用固定索引寫回 `partMeasures` 列表時發生錯誤

#### 程式碼問題

**錯誤位置**: `converter21/shared/m21utilities.py:4542`

**原始錯誤程式碼**:
```python
# 第 4508-4510 行：建立固定長度列表
partMeasures: list[list[m21.stream.Measure]] = [
    list(part[m21.stream.Measure]) for part in parts
]

# 第 4523 行：檢查小節數不足
if msIdx >= numMeasuresInParts[partIdx]:
    # 第 4531 行：正確地 append 到 parts
    parts[partIdx].append(emptyMeas)

# 第 4542 行：BUG！嘗試用索引賦值到固定長度列表
partMeasures[partIdx][msIdx] = emptyMeas  # IndexError!
```

**問題分析**:
1. `partMeasures` 在第 4508-4510 行建立為固定長度列表
2. 當需要補充空白小節時，`parts[partIdx]` 正確地使用 `append()`
3. 但 `partMeasures[partIdx]` 嘗試用索引直接賦值，導致越界錯誤

**正確的修復方式**:
```python
# 修正第 4542 行
if msIdx < len(partMeasures[partIdx]):
    partMeasures[partIdx][msIdx] = emptyMeas
else:
    partMeasures[partIdx].append(emptyMeas)
```

#### 觸發條件

**為何是這首 Scherzo？**

根據執行日誌顯示：
- 系統偵測到 measure 780 需要被加到 part 0
- 但 `partMeasures[0]` 的長度只有 779
- 嘗試 `partMeasures[0][780] = emptyMeas` 時發生 IndexError

**樂曲特徵**:
1. **樂曲規模**: 長達 780 小節的大型作品
2. **最後一小節內容**:
   - 包含 2 個 staves 的正常鋼琴譜
   - 有正確的結束標記：`<barline location="right"><bar-style>light-heavy</bar-style>`
   - 包含 grace notes、fermata、octave shifts、pedal markings
   - 正常的 4/4 小節時值（240 duration）
3. **聲部不對齊**: MusicXML 導出時，不同 Staff 間的小節數存在微小差異

#### 解決方案

- 等待 converter21 v4.0.1 修復
- 向 converter21 作者 Greg Chapman 回報此問題

---

### converter21 SMUFL TextExpression Bug（已修復）

#### 發現日期
2026-01-21

#### 檔案資訊
- **資料集**: MuseSyn
- **受影響檔案**: `The_Glorious_State_Anthem_of_the_Soviet_Union.xml`
- **錯誤類型**: `IndexError: string index out of range`
- **錯誤位置**: converter21 `humdrum/m21convert.py:2064` (`translateSMUFLNotesToNoteNames`)

#### Bug 本質

converter21 在處理 SMUFL（Standard Music Font Layout）字元時，如果 SMUFL 字元位於字串末尾，會發生索引越界錯誤。

#### SMUFL 說明

SMUFL 是標準化的音樂符號字體規範，將音樂符號編碼到 Unicode Private Use Area (U+E000 - U+F8FF)。
常見於 MuseScore 匯出的 MusicXML 中，用於節拍器標記等視覺符號。

converter21 支援的 SMUFL 節拍器符號對應：
| Unicode | Humdrum 名稱 |
|---------|-------------|
| U+ECA0 | double-whole |
| U+ECA2 | whole |
| U+ECA3 | half |
| U+ECA5 | quarter |
| U+ECA7 | 8th |
| U+ECA9 | 16th |

#### 問題根源

**錯誤位置**: `converter21/humdrum/m21convert.py:2064`

**原始錯誤程式碼**:
```python
@staticmethod
def translateSMUFLNotesToNoteNames(text: str) -> str:
    # ...
    for i, char in enumerate(text):
        if char in SharedConstants.SMUFL_METRONOME_MARK_NOTE_CHARS_TO_HUMDRUM_NOTE_NAME:
            output += '[' + SharedConstants.SMUFL_METRONOME_MARK_NOTE_CHARS_TO_HUMDRUM_NOTE_NAME[char]
            j = i + 1
            while text[j] in (...):  # BUG: 沒有邊界檢查！
                # ...
```

**問題分析**:
1. 當 SMUFL 字元在字串末尾時，`j = i + 1` 會等於字串長度
2. `while text[j]` 會存取超出範圍的索引，導致 `IndexError`

#### 觸發條件

**復現測試**:
```python
from converter21.humdrum.m21convert import M21Convert

# 這些會 crash
M21Convert.translateSMUFLNotesToNoteNames('\ueca5')        # 只有 SMUFL
M21Convert.translateSMUFLNotesToNoteNames('tempo = \ueca5') # SMUFL 在末尾

# 這個正常
M21Convert.translateSMUFLNotesToNoteNames('\ueca5 = 120')   # SMUFL 後面有字元
# 輸出: '[quarter] = 120'
```

**受影響的 MuseSyn 檔案**:
該檔案包含一個 TextExpression，內容只有單一 SMUFL 字元 `'\ueca5'`（四分音符符號）。

#### 解決方案

**我們的 Workaround**（在 `sanitize_piano_score.py`）:
在 TextExpression 末尾的 SMUFL 字元後面加空格，繞過 converter21 的邊界檢查 bug。

```python
def fix_smufl_text_expressions(score):
    """Fix SMUFL characters at end of TextExpression to avoid converter21 bug."""
    for el in score.recurse():
        if isinstance(el, m21.expressions.TextExpression):
            if el.content and is_smufl_char(el.content[-1]):
                el.content = el.content + ' '  # 加空格繞過 bug
```

這樣：
- `'\ueca5'` → `'\ueca5 '` → converter21 輸出 `'[quarter] '`
- 資訊保留，不會遺失速度標記

**正確的修復方式**（應由 converter21 修復）:
```python
# m21convert.py:2064
j = i + 1
while j < len(text) and text[j] in (...):  # 加入邊界檢查
    # ...
```

#### 後續行動

- [x] 在 `sanitize_piano_score.py` 實作 workaround
- [ ] 向 converter21 作者 Greg Chapman 回報此問題

---

# Humdrum Chopin First Editions — 資料品質錯誤報告

> 調查日期：2026-02-03
> 資料來源：[Humdrum Chopin First Editions](https://github.com/pl-wnifc/humdrum-chopin-first-editions)
> 跳過清單：`src/datasets/syn/skip_files.txt`

## 摘要

8 首 Chopin kern_gt 檔案因**原始 Humdrum 編碼的品質問題**無法通過 converter21 轉換為 MIDI。
問題分為兩大類：

| 類別 | 數量 | 本質 |
|------|------|------|
| Spine split 區段小節溢出（negative delta time） | 6 | `*^` split 後的 sub-spine 音符總時值超過拍號 |
| 不可表示的 quarterLength | 2 | 原始 kern 使用非標準 duration，converter21 無法精確轉換 |

這些是 Humdrum 手動編碼的錯誤，非 pipeline 問題。8/723 = **1.1% 損失率**。

---

## 類別一：Negative Delta Time（6 首）

### 問題描述

converter21 將 kern 轉為 music21 Score 再寫入 MIDI 時，計算出負的 `offsetInScore`。
根本原因是原始 kern 檔案在 `*^` spine split 區段中，sub-spine 的音符總時值超過了小節拍號允許的長度，
導致下一個事件的 offset 比前一個事件更早。

### 受影響檔案

| 檔案 | Chopin 作品 |
|------|------------|
| `009-1-KI-003` | Nocturne Op. 9 No. 1 |
| `023-1-BH` | Ballade No. 1, Op. 23 |
| `028_1-12-1a-C-005` | Prelude Op. 28 No. 5 |
| `028_13-24-1a-C-013` | Prelude Op. 28 No. 13 |
| `055-1-BH-002` | Nocturne Op. 55 No. 1 |
| `060-1-BH` | Barcarolle, Op. 60 |

### Pipeline 處理歷程

- `fix_kern_spine_timing` 曾經修復 48 → 14 首的 spine timing 問題（救回 34 首）
- 移除 Phase 1 的 `expand_tuplets_to_zeng_vocab` 再救回 9 首
- 這 6 首的 timing 錯誤存在於原始編碼中，無法自動修正

---

## 類別二：不可表示的 quarterLength（2 首）

### 問題描述

原始 kern 使用的 duration 值（如 `0.6875` quarterLength）無法被 music21 精確轉換為標準音符時值類型。
這通常是編碼者用 triplet duration（`12`, `6`）近似複雜節奏所致，留下不完整的 tuplet group。

### 受影響檔案

| 檔案 | Chopin 作品 | 錯誤訊息 |
|------|------------|----------|
| `021-1a-BH-001` | Nocturne Op. 21 (posth.) | `cannot convert quarterLength 0.6875 exactly to type` |
| `021-1a-BH-002` | Nocturne Op. 21 (posth.) | 同上 |

---

## 結論

- 8 首全部是 Humdrum 手動編碼的品質問題，非 clef pipeline 的 bug
- 成功率 715/723 = **98.9%**
- 這些檔案已列入 `src/datasets/syn/skip_files.txt`，Phase 2 自動跳過
