# Data Pipeline: Clef 數據生成與增強策略

本研究採用 **"Synthetic-to-Real" (合成到真實)** 的訓練策略。由於真實世界缺乏完美的「音訊-樂譜」對齊數據，我們構建了一套自動化的數據生成管線，利用高強度的 **數據增強 (Data Augmentation)** 來迫使模型學習音樂的幾何結構，而非死記音色紋理。

---

## 0. 資料量比較與設計哲學

### 與其他系統的資料量比較

| 系統 | 訓練資料 | 時長 | 標註品質 |
|------|---------|------|---------|
| **Whisper** | 網路弱標註音訊 | 680,000 小時 | 弱標註（YouTube 字幕） |
| **MT3 (Multi-instrument)** | 6 個資料集混合 | 1,750 小時 | MIDI-level |
| **Zeng (2024)** | MuseSyn + HumSyn | ~50 小時 | Score-level |
| **Clef** | PDMX + TDR | **~2,000+ 小時** | **Score-level (Kern)** |

**Clef 的資料量已超過 MT3**，且標註品質更高（完美的樂譜對齊）。

### MT3 Multi-instrument 的資料組成

| 資料集 | 時長 | 類型 |
|--------|------|------|
| Slakh2100 | 969 小時 | 合成（MIDI 渲染） |
| Cerberus4 | 543 小時 | 合成（4 樂器組合） |
| MAESTROv3 | 199 小時 | 真實錄音 |
| MusicNet | 34 小時 | 真實錄音（標註較差） |
| GuitarSet | 3 小時 | 真實錄音 |
| URMP | 1 小時 | 真實錄音 |

MT3 的真實錄音只有 ~238 小時，其餘都是合成。Clef 的策略相似，但資料量更大。

### 核心哲學：質勝於量

Whisper 靠 **暴力 scale（680,000 小時）** 取勝，但標註品質差。

Clef 的策略是：
1. **完美標註的合成資料**（~2,000+ 小時）
2. **有意義的 Data Augmentation**（不是 naive 增強）
3. **Sim-to-Real 設計**（讓合成逼近真實）

> **關鍵洞見**：資料量不是瓶頸，**音源多樣性和 Sim-to-Real 策略**才是關鍵。

---

## 0.1 ISMIR 2026 vs ICLR 2027 的增強策略差異

| 面向 | ISMIR 2026 (單樂器) | ICLR 2027 (多樂器合奏) |
|------|---------------------|----------------------|
| **TDR 策略** | 同樂器內隨機化 | 同樂器內隨機化（相同） |
| **例子** | 鋼琴譜 → 不同鋼琴音源 | 鋼琴譜 → 不同鋼琴音源 |
| **Aux Loss** | 不使用 | ✅ Instrument Auxiliary Loss |
| **目的** | 音源魯棒性 | 音源魯棒性 + 樂器解纏 |

**兩者都使用「同樂器內隨機化」**：
- 鋼琴譜只用不同鋼琴音源（Steinway, Yamaha, Upright, Electric）
- 小提琴譜只用不同小提琴音源（Stradivarius, Modern, Baroque）
- **永遠不會出現「吉他譜 + 鋼琴音色」的組合**

**為什麼不做跨樂器 TDR？**
- 跨樂器 TDR 會讓模型混淆樂器標籤
- 用 **Auxiliary Loss** 取代，更乾淨地幫助模型從音色學會辨識樂器
- 模型必須從音色中學會區分樂器，而不是從幾何結構

**ICLR 2027 的額外武器**：
- Instrument Auxiliary Loss 強迫 encoder 保留音色資訊
- 讓模型在多樂器合奏場景中正確辨識各樂器

---

## 0.2 有意義的 Augmentation vs Naive Augmentation

### ❌ Naive Augmentation（不採用）

| 方法 | 為什麼沒用 |
|------|-----------|
| **Key Shift (±4 semitones)** | 對 Sim-to-Real 沒幫助，模型已經能處理不同調性 |
| **Random Pitch Shift** | 破壞音樂結構，產生不自然的音訊 |
| **Time Stretching（大幅度）** | 產生 artifact，不像真實演奏 |

### ✅ 有意義的 Augmentation（採用）

| 方法 | 目的 | 說明 |
|------|------|------|
| **Rubato 模擬** | 模擬真人演奏的時間變化 | VirtuosoNet 或自設計演算法 |
| **Room IR Convolution** | 模擬不同錄音環境 | 音樂廳、錄音室、小房間 |
| **麥克風模擬** | 頻率響應差異 | EQ + mild coloration |
| **動態變化** | 模擬 pp → ff | Velocity curves + compression |
| **多音源切換** | 同樂器不同音色 | 每種樂器 3-5 種音源 |

### 混音師哲學

> **「每一個 Data Augmentation 都必須聽起來是『音樂』，而不是『噪音』。」**

這是 Clef 與其他研究的核心差異：我們不做會破壞音樂性的增強。

## 1. 數據來源 (Data Sources)

### Study 1 vs Study 2 的訓練資料策略

| | Study 1 (Piano) | Study 2 (Universal Solo) |
|---|---|---|
| **訓練資料** | MuseSyn + HumSyn（與 Zeng 相同） | MuseSyn + HumSyn + 多樂器拆分 + PDMX 非古典 |
| **Piano 資料** | ~2,200 首（與 Zeng 相同） | ~10,000 首（控制比例，避免 overfit） |
| **其他樂器** | — | Violin ~8k, Voice ~8k, Cello ~4k, etc. |
| **目的** | 公平比較 | 跨樂器泛化 |

### Study 2: Piano 資料策略

**核心決策**：不使用 PDMX 的 50,000+ Piano Solo，避免樂器不平衡和 overfit。

| 來源 | Piano 數量 | Genre | 角色 |
|------|-----------|-------|------|
| MuseSyn | ~200 | **Pop** | Study 1 baseline |
| HumSyn | ~2,000 | Classical/Ragtime | Study 1 baseline |
| PDMX (Jazz/Rock/etc.) | ~2,000-3,000 | Non-classical | 多元化 |
| 拆分的伴奏 | ~5,000 | Mixed | 伴奏角色 |
| **Total Piano** | **~10,000** | **Balanced** | ✅ |

**伴奏 Piano 的價值**：從 Piano-Voice、Piano-Violin 等拆出來的 Piano 是伴奏角色，讓模型學會「不要假設 piano 一定有旋律」。

### Genre 多元化：好和弦策略

PDMX rated subset (~14,182 首) 中約 **40% 是非古典/民謠**，包含 pop, jazz, rock, blues, latin, world, soundtrack 等。

### 資料集總覽

| **資料集用途** | **Study 1 (Piano)** | **Study 2 (Universal Solo)** |
|----------------|---------------------|------------------------------|
| **訓練 (Training)** | MuseSyn + HumSyn (~2,200 首，與 Zeng 相同) | MuseSyn + HumSyn + 多樂器拆分 + PDMX 非古典 (~40,000 首) |
| **測試 (Sim2Real)** | ASAP test split (80 段) | ASAP + GAPS + Bach Violin + GTSinger (~500 段) |

------

## 2. 數據增強金字塔 (The Augmentation Pyramid)

為了讓模型學會 **"Structure Invariance" (結構不變性)**，我們設計了三層增強機制。這不是隨機的噪音，而是有層次地剝離模型的依賴。

### Level 1: 訊號層增強 (Signal-Level Augmentation)

> **目的**：模擬不同的錄音環境與設備，解決 **"Channel Robustness"** 問題。

- **聲學環境模擬 (Acoustic Simulation)**：
  - **Impulse Response (IR) Convolution**：隨機掛載「大教堂」、「浴室」、「錄音室」、「小房間」的空間殘響 (Reverb)。
- **訊號劣化 (Signal Degradation)**：
  - **Additive Noise**：加入白噪音、粉紅噪音、街道環境音、錄音帶底噪。
  - **Frequency Cutoff**：隨機 Low-pass / High-pass filter（模擬爛麥克風或手機錄音）。
  - **Compression**：動態壓縮，模擬現代流行音樂的響度戰爭。

### Level 2: 音源層增強 (Source-Level Augmentation)

> **目的**：解決 **"Intra-class Variance" (類內變異)**，讓模型適應同一種樂器的不同狀態。

- **多樣化採樣 (Multi-Sampling)**：
  - 針對同一種樂器（如鋼琴），隨機切換不同的 SoundFont (如 Steinway, Yamaha, Upright, Honky-tonk)。
- **物理缺陷模擬 (Physical Imperfections)**：
  - **Detuning**：隨機對音高進行微小偏移 ($\pm 10$ cents)，模擬沒調準的樂器。
  - **Timing Jitter**：在生成 Audio 時加入微小的時間抖動，模擬人類演奏的不完美。

------

## 3. 實作架構 (Pipeline Implementation)

這部分是你寫程式的邏輯，展示給老師看會非常有說服力：

Code snippet

```
graph LR
    A[Raw MusicXML (PDMX)] --> B{Augmentation Logic};
    
    subgraph "The 'Cover' Generator"
    B --> C1[Random Instrument Selector];
    B --> C2[Random Tempo/Rubato];
    end
    
    C1 --> D[FluidSynth / MIDI Renderer];
    C2 --> D;
    
    D --> E[Audio Waveform];
    
    subgraph "Signal Processor (Pedalboard)"
    E --> F1[Add Reverb (IR)];
    E --> F2[Add Noise / EQ];
    end
    
    F2 --> G[Log-Mel Spectrogram];
    G --> H[Input to ViT];
    
    A --> I[Text Tokenizer];
    I --> J[Target Labels (**Kern Notation)];
```

### 關鍵技術棧 (Tech Stack)

- **渲染引擎**: `FluidSynth` + 大量開源 `.sf2` (SoundFonts)。
- **訊號處理**: `Spotify Pedalboard` (Python) 或 `TorchAudio`。
- **頻譜轉換**: `Librosa` 或 `nnAudio` (GPU 加速)。
- **對齊**: 由於 Audio 是我們自己生成的，**Time Alignment 是完美的**（這是做 Synthetic Data 最大的優勢）。

------

### 給老師的 Pitch 重點

當你展示這張表時，請強調：

> 「老師，這套 Data Pipeline 的核心在於 **Sim-to-Real 策略**。
>
> 傳統研究只做 naive augmentation（加噪音、key shift），這些對 Sim-to-Real 沒幫助。
>
> 我引入了『混音師思維』——每一個 augmentation 都必須聽起來是『音樂』，而不是『噪音』。
>
> 通過 **Rubato 模擬**、**Room IR Convolution**、**多音源切換**，我讓合成資料逼近真實錄音的分佈。
>
> 這保證了我的 Clef 模型在真實錄音上也能保持高精度，而不是只會處理合成資料。」
