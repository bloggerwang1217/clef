# 模型比較

以下用最直觀的「數據流（Data Flow）」和「歸納偏置（Inductive Bias）」來拆解 Whisper、標準 VLM 與 Clef 的差異：

### 1. Whisper (OpenAI) —— 「聽覺」模型

Whisper 是標準的 **Speech-to-Text** 架構。它的核心假設是：**聲音是隨時間變化的 1D 序列**。

- **輸入 (Input)**: Log-Mel Spectrogram（雖然也是圖，但它被視為時間序列）。
- **編碼器 (Encoder)**: **Audio Transformer (Custom)**。
  - 它使用 **1D-Convolution** 來壓縮時間軸。
  - **核心機制**：它掃描的是 **「時間 (Time)」**。它關注的是 *前後關係*（例如：這個音素後面接哪個音素）。
  - **歸納偏置 (Bias)**：**語音偏見**。它擅長捕捉 **Formants (共振峰，決定母音)** 和 **Phonemes (音素)**，對時間極度敏感，但對「垂直頻率堆疊（和聲）」不敏感。
- **解碼器 (Decoder)**: Text Transformer。

**評價**：就像 Zhang & Sun 做的，拿它來轉譜會失敗，因為它會忽略和聲結構，只在意旋律線。

### 2. Vision-Language Model (如 LLaVA / Legato) —— 「視覺」模型

這是目前 AI 界的主流（如 GPT-4o 的視覺部分）。它的核心假設是：**輸入是 2D 空間中的物體**。

- **輸入 (Input)**: **RGB Image** (真實照片) 或 **Score Image** (樂譜圖片，如 Legato)。
- **編碼器 (Encoder)**: **ViT (Vision Transformer)**。
  - 它將圖片切成 **$16 \times 16$ Patches**。
  - **核心機制**：它掃描的是 **「空間 (Space)」**。它關注的是 *幾何關係*（例如：貓的耳朵在頭的上方）。
  - **歸納偏置 (Bias)**：**物體恆常性**。不管物體在哪裡（平移不變性），它都能認出形狀、邊緣、紋理。
- **解碼器 (Decoder)**: LLM (Text Transformer)。

### 3. 你的 Clef (Swin-V2 + Bridge + Transformer) —— 「跨模態」模型

你的架構在數學上 **不等於** 標準 VLM，因為你採用了混合設計來適配音樂的獨特結構：

- **輸入 (Input)**: **Log-Mel Spectrogram** (跟 Whisper 一樣的圖)。
- **編碼器 (Encoder)**: **Swin Transformer V2 + Global Bridge**。
  - **Swin V2 前端**：使用 **相對位置偏差（Relative Position Bias）** 與 **階層式視窗機制**，天生適配任意長度的音訊頻譜圖。
  - **Global Bridge**：2-4 層標準 Transformer Encoder，提供跨段落的全域注意力機制。
  - **Frozen Weights**：Swin V2 的預訓練權重可直接凍結，因為其相對位置偏差與輸入長度無關。
- **解碼器 (Decoder)**: Autoregressive Transformer (跟 Whisper/VLM 一樣)。

**你抓到的特徵**：
- **Swin 的局部注意力**：垂直線（節奏/Onset）、水平線（音高）、垂直堆疊的距離（和聲/Intervals）。
- **Bridge 的全域注意力**：段落呼應（Intro ↔ Outro）、主題重複（副歌結構）、跨小節的和聲進行。

---

### 三者比較總表 (Cheatsheet)

| **特徵**           | **1. Whisper**                | **2. Standard VLM (Legato)** | **3. Your Clef**                          |
| ------------------ | ----------------------------- | ---------------------------- | ----------------------------------------- |
| **輸入資料**       | 聲音頻譜 (Audio)              | 真實照片 / 樂譜圖片          | **聲音頻譜 (Audio)**                      |
| **看待資料的方式** | **1D 時間序列** (Time-Series) | **2D 空間幾何** (Spatial)    | **2D 頻譜圖 + 階層式處理** (Hierarchical) |
| **Encoder 架構**   | Audio Transformer (1D Conv)   | **ViT (2D Patch)**           | **Swin-V2 (Window) + Bridge (Global)**    |
| **位置編碼**       | Sinusoidal (絕對)             | Learnable (絕對)             | **Relative Bias + Learnable (相對+絕對)** |
| **預訓練來源**     | 語音數據 (Speech)             | **ImageNet (Objects)**       | **ImageNet (Objects)**                    |
| **擅長捕捉**       | 語音內容、時間流動            | 物體形狀、邊緣、紋理         | **和聲結構、音色紋理、段落呼應**          |
| **對長序列**       | 受限於 context window         | 需要插值                     | **自然適配 (Relative Bias)**              |
| **弱點**           | 忽略和聲、音高不準            | (無法處理聲音)               | 需要 Bridge 來捕捉全域結構                |

---

### 論文核心論述 (The Argument)

你在論文中要強調的是：

> 「雖然 Whisper 和 Clef 的輸入都是頻譜圖，但處理邏輯完全不同。
>
> Whisper 是用『時間邏輯』(1D) 去聽，所以它適合單音旋律或語音。
>
> Clef 是用『視覺邏輯』(2D Swin) 去看頻譜幾何，再加上 Bridge 來理解全曲結構，所以它能捕捉到複音音樂中複雜的『垂直和聲關係』與『全域段落呼應』。
>
> 這就是為什麼我們採用 Swin-V2 + Bridge 架構來解決音樂轉譜問題。」

---

### 關鍵差異：為何 Swin-V2 比 ViT 更適合？

| 考量 | ViT | Swin-V2 |
|------|-----|---------|
| **注意力複雜度** | O(n²) 全域 | O(n) 線性（視窗） |
| **位置編碼** | 絕對位置，變長需插值 | 相對偏差，與長度無關 |
| **音樂適配性** | 需要 hack 才能處理變長音訊 | 天生適配任意長度 |
| **階層結構** | 無 | Stage 1-4 模擬 Note→Phrase→Section |

---

### 為何需要 Bridge？

標準 Swin 只有局部視窗注意力，缺乏捕捉全曲結構呼應的能力。Bridge 的設計：

1. **讓第一秒直接與第一百八十秒互動**（標準 Self-Attention）
2. **學習可學習的位置編碼**來建模全曲時間依賴
3. **輸出給 Decoder** 時已包含全曲上下文資訊
