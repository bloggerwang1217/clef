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
| MT3 (Pipeline) | MAESTRO + Slakh | ~80% (F1) | ~40% (F1) | ~35% (F1) | ~30% (F1) |
| Clef (Study 1) | Piano Only | **> 78% (MV2H)** | < 20% (Fail) | < 20% (Fail) | < 20% (Fail) |
| Clef (Study 2) | **Universal (TDR)** | **> 76% (MV2H)** | **> 60% (MV2H)** | **> 60% (MV2H)** | **> 55% (MV2H)** |

### 表格亮點

1. **Clef (Study 1)**：證明「專用模型」的侷限性（只練鋼琴，遇到小提琴就掛了）
2. **Clef (Study 2)**：Big Data + TDR 展現 **Zero-Shot 能力**
3. **對比 MT3**：MT3 輸出 MIDI Token，在轉譜任務上結構性依然輸

---

## 論文結構總覽

| Study | 定位 | 戰場 | 對手 | 目標 |
|-------|------|------|------|------|
| Study 1 | Depth (深度) | ASAP (Piano) | Zeng 2024, MT3 Pipeline | MV2H > 78% |
| Study 2 | Breadth (廣度) | URMP (Multi-instrument) | 模型極限 (Self-Challenge) | Zero-shot MV2H > 60% |

### 核心論點

> 「只要有譜，我就能生成訓練資料；只要有訓練資料，我的 VLM 就能學會轉任何樂器。」

這才是「通用轉譜 AI」的真正價值。
