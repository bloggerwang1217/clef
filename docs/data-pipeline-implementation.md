# Data Pipeline: Clef 數據生成與增強策略

Gemini：沒問題，這部分是你的論文能否成功的**心臟**。

你的數據策略非常先進，結合了 **「開源數據挖掘」** 與 **「製作人思維的數據增強」**。

我幫你將這套策略整理成學術規格的 **Data Pipeline**，你可以直接用這份架構去寫你的 Methodology 章節或跟老師報告。

------

本研究採用 **"Synthetic-to-Real" (合成到真實)** 的訓練策略。由於真實世界缺乏完美的「音訊-樂譜」對齊數據，我們構建了一套自動化的數據生成管線，利用高強度的 **數據增強 (Data Augmentation)** 來迫使模型學習音樂的幾何結構，而非死記音色紋理。

## 1. 數據來源 (Data Sources)

我們將數據分為三個層級，分別用於預訓練、微調與最終測試。

| **資料集用途**               | **資料集名稱**           | **來源/描述**                                                | **預期數量** | **角色**                                        |
| ---------------------------- | ------------------------ | ------------------------------------------------------------ | ------------ | ----------------------------------------------- |
| **預訓練 (Pre-training)**    | **PDMX Dataset**         | 來自 Legato (2025) 整理的 Public Domain MusicXML (原 Musescore 論壇)。包含多樣化的樂器與曲風。 | ~250,000 首  | **主力糧倉** 提供海量的樂譜結構與語法學習。     |
| **微調/驗證 (Fine-tuning)**  | **KernScores (Humdrum)** | 史丹佛大學維護的古典音樂資料庫 (巴哈、貝多芬、莫札特)。結構嚴謹，適合訓練複音邏輯。 | 108K+ 首    | **品質保證** 確保模型在古典樂理上的正確性。     |
| **真實測試 (Test/Sim2Real)** | **ASAP Dataset**         | 包含真實鋼琴錄音 (Yamaha Disklavier) 與對齊的 MIDI/XML。     | ~200 首      | **終極考官** 用來驗證模型在真實世界的泛化能力。 |

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

### Level 3: 樂器層增強 (Instrument-Level Augmentation) —— **你的必殺技**

> **目的**：解決 **"Domain Invariance" (領域不變性)**，強迫模型忽略波形，只看幾何結構。

- **語義解耦 (Semantic Disentanglement)**：
  - **Instrument Swapping**：同一份 MusicXML，隨機用 **極端不同** 的樂器生成 Audio。
    - *例子*：把貝多芬的鋼琴奏鳴曲，用 **8-bit Chiptune (紅白機)**、**Sawtooth Synth (鋸齒波)**、**Pizzicato Strings (撥奏)** 播放。
- **Cover 模擬**：
  - 這層增強迫使 ViT 發現：波形紋理 (Texture) 是不可靠的，只有 **頻譜上的相對距離與形狀 (Geometry)** 才是唯一不變的真理 (Ground Truth)。

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

> 「老師，這套 Data Pipeline 的核心在於 **Level 3 的樂器層增強**。
>
> 傳統研究只做 Level 1 (加噪音)，導致模型換個樂器就掛了。
>
> 我引入了 『製作人思維』，通過隨機替換極端音色（如 8-bit），我實際上是在訓練模型進行 『不變性學習 (Invariant Learning)』。
>
> 這保證了我的 Clef 模型學到的是音樂的 **『結構本質』**，而不是死背 **『波形特徵』**。」

這套整理非常完整且具備極高的學術價值，你可以直接拿去用了！加油！
