# Plan: clef-piano-base Training Pipeline (Complete)

整合 `docs/clef-piano-base-model-design.md` 和 `docs/clef-piano-base-deformable-transformer.md` 的所有細節。

**2026-01 更新**：根據認知科學分析（Auditory Scene Analysis, Bregman 1990），改用 content-dependent reference points + 方形採樣。

### 關鍵設計變更摘要

| 項目 | 舊設計 | 新設計 |
|------|--------|--------|
| freq_base | 固定 0.5 | **content-dependent** `freq_prior(tgt)` |
| 採樣形狀 | 垂直長條 6×2 | **方形 2×2** |
| n_points per level | 12 | **4** |
| 總採樣點 | 48 | **16** |
| freq_offset_scale | ±50% | **±15%** (統一方形) |
| 設計理念 | 覆蓋全頻率泛音 | **Stream tracking**（prior 定位） |

---

## 目標

完成 clef-piano-base 的模型訓練 pipeline：
- ~~Tokenizer（Factorized encoding）~~ ✅ 已完成
- ~~SynDataset~~ ✅ 已完成
- Model（Swin V2 + Deformable Bridge + **ClefAttention** Decoder）
  - **CLEF** = Content-aware Learned-prior Event Focusing Attention
- Audio Transform（Log-Mel）
- Collate Function（ChunkedDataset + BucketSampler + Valid Ratios）
- Training Loop（DDP + AMP）
- Inference

**現有進度**：
- `src/score/kern_tokenizer.py` ✅ — Factorized encoding (~220 vocab)
- `src/datasets/syn_dataset.py` ✅ — 支援 offline/online 模式
- Phase 2 Audio 合成 ✅ — 完成 (2,080 音檔)
- `src/clef/config.py` ✅ — ClefConfig base class
- `src/clef/piano/config.py` ✅ — ClefPianoConfig
- `src/clef/attention.py` ✅ — ClefAttention (方形 2×2 採樣)
- `src/clef/bridge.py` ✅ — DeformableBridge
- `src/clef/decoder.py` ✅ — ClefDecoder + DeformableDecoderLayer
- `src/clef/piano/model.py` ✅ — ClefPianoBase (59M params, 31M trainable)
- `src/clef/data.py` ✅ — ChunkedDataset, ManifestDataset
- `src/clef/collate.py` ✅ — BucketSampler, ClefCollator
- `src/clef/piano/train.py` ✅ — Training script with DDP + AMP

**記憶體實測** (RTX 3090):
- 30 sec + batch=2 + grad = **11.82 GB**
- 1 min + batch=2 (forward only) = **23.36 GB**

## 設計理念

**認知科學依據**：
- 人腦不是「掃描固定頻率」，而是進行 **stream segregation**（Bregman, 1990）
- 聽鋼琴時，可以選擇性關注「右手旋律」或「左手和聲」
- 這是 **content-dependent attention**，不是固定採樣

**對模型設計的啟示**：
- `freq_prior`：從 decoder hidden state 預測「要看高頻還是低頻」
- `time_prior`：從 position embedding 預測「要看哪個時間點」
- 方形採樣 (2×2)：prior 已經做了粗定位，offset 只需要看局部細節

**與 Stripe-Transformer / hFT-Transformer 的關係**：
- 他們用 full attention 分離 freq/time 處理
- 我們用 **learned spatial priors** + sparse sampling 達成類似效果
- 複雜度更低，且保留彈性

---

## 架構總覽

### 完整架構圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Observation Space (X)                            │
│                    Log-Mel Spectrogram [B, 1, 128, T]                   │
│                         ↓  (複製成 3-channel)                           │
│                    [B, 3, 128, T] for Swin input                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   Feature Extraction: Swin V2 (Frozen)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Stage 1 (High Res)  ──→ F1 [B, 96,  32, 3750]  (~16ms/token) ← grace  │
│  Stage 2             ──→ F2 [B, 192, 16, 1875]  (~32ms/token) ← onset  │
│  Stage 3             ──→ F3 [B, 384, 8,  937]   (~64ms/token) ← note   │
│  Stage 4 (Semantic)  ──→ F4 [B, 768, 4,  468]   (~128ms/token)← chord  │
│                                                                         │
│  Swin-tiny dims: C1=96, C2=192, C3=384, C4=768                         │
│  全部 4 個 scale 都用！Deformable Attention 處理稀疏採樣               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              Multi-scale Deformable Bridge (Encoder)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  各 Scale 投影到統一維度：                                               │
│    F1 → Linear(96→512)  + level_embed[0]                               │
│    F2 → Linear(192→512) + level_embed[1]                               │
│    F3 → Linear(384→512) + level_embed[2]                               │
│    F4 → Linear(768→512) + level_embed[3]                               │
│                                                                         │
│  Flatten & Concat: [B, N_total, 512]                                   │
│    N_total = 32×3750 + 16×1875 + 8×937 + 4×468                         │
│            = 120,000 + 30,000 + 7,496 + 1,872 = 159,368 tokens         │
│                                                                         │
│  ⚠️ 但我們不做 full attention！                                         │
│                                                                         │
│  Deformable Self-Attention × 2 layers:                                 │
│    每個 token 只採樣 K×L = 4×4 = 16 個位置（方形採樣）                   │
│    複雜度: O(159,368 × 16) ≈ 2.5M ops << O(159,368²)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│     Autoregressive Decoder with ClefAttention             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Token Embedding [vocab_size, 512]                                      │
│            ↓                                                            │
│  + Learnable Positional Embedding [max_seq_len, 512]                   │
│            ↓                                                            │
│  Transformer Decoder × 6 層，每層：                                     │
│    ┌─────────────────────────────────────────────────────────────┐     │
│    │ 1. Causal Self-Attention (標準)                             │     │
│    │    Q, K, V = decoder tokens                                 │     │
│    │    Mask = causal (只看已生成的 token)                       │     │
│    ├─────────────────────────────────────────────────────────────┤     │
│    │ 2. ClefAttention (關鍵創新！)             │     │
│    │    Query = decoder hidden state                             │     │
│    │    Keys/Values = multi-scale encoder features (F1~F4)       │     │
│    │                                                             │     │
│    │    Content-Dependent Reference Points:                      │     │
│    │      time_base = time_prior(tgt_pos).sigmoid()  [B, S, 1]  │     │
│    │      freq_base = freq_prior(tgt).sigmoid()      [B, S, 1]  │     │
│    │      refine = reference_refine(tgt).tanh() * 0.1            │     │
│    │      reference_points = (base + refine).clamp(0, 1)         │     │
│    │                                                             │     │
│    │    方形採樣 (2 freq × 2 time = 4 points per level):        │     │
│    │      freq_offset: ±15% (統一方形採樣)                      │     │
│    │      time_offset: ±15% (統一方形採樣)                      │     │
│    │                                                             │     │
│    │    總採樣: S × H × L × K = 4096 × 8 × 4 × 4 = 524K 次     │     │
│    │    vs Full Attention: 4096 × 159,368 = 652M 次              │     │
│    ├─────────────────────────────────────────────────────────────┤     │
│    │ 3. FFN                                                      │     │
│    └─────────────────────────────────────────────────────────────┘     │
│            ↓                                                            │
│  Output Head: Linear(512 → vocab_size)                                 │
│            ↓                                                            │
│  **Kern Tokens                                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 與原 Deformable DETR 的關鍵差異

| 項目 | Deformable DETR (原論文) | Clef (音樂專用設計) |
|------|-------------------------|-------------------|
| Reference Points | 每個 level 獨立 `[B, S, L, 2]` | 共用 + **Content-Dependent Prior** |
| freq_base | 固定或從 query 預測 | **`freq_prior(tgt)`** — 從內容預測 |
| time_base | 固定或從 query 預測 | **`time_prior(tgt_pos)`** — 從位置預測 |
| 採樣點分布 | 放射狀 | **方形** (prior 已定位) |
| n_points | 4 (均勻) | **4** (2 freq × 2 time) |
| Offset 預測 | 合併 `[..., 2]` | 分離 `time_offset_proj` + `freq_offset_proj` |
| Offset 範圍 | 均勻 | **freq ±20%, time ±10%** |
| Valid Ratios | 無 | 新增，處理 padding |
| 設計理念 | 物體邊界 | **Stream tracking** (Bregman, 1990) |

### 與 Stripe-Transformer / hFT-Transformer 的關係

| 項目 | Stripe/hFT | ClefAttention |
|------|------------|---------------|
| Freq 處理 | Full attention over F | `freq_prior` 定位 + sparse sampling |
| Time 處理 | Full attention over T | `time_prior` 定位 + sparse sampling |
| 複雜度 | O(F² + T²) | O(N × K × L)，K=4 |
| 分離方式 | 架構分離（兩個 Transformer） | **Prior 分離**（同一個 Attention） |
| 命名 | - | **C**ontent-aware **L**earned-prior **E**vent **F**ocusing |

---

## 實作步驟

### Step 0: 前置驗證

#### 0.1 Swin V2 Hidden States 格式確認

```bash
python -c "
from transformers import Swinv2Model
import torch

model = Swinv2Model.from_pretrained(
    'microsoft/swinv2-tiny-patch4-window8-256',
    output_hidden_states=True
)

x = torch.randn(1, 3, 128, 256)
out = model(x)

print(f'Number of hidden states: {len(out.hidden_states)}')
for i, h in enumerate(out.hidden_states):
    print(f'  hidden_states[{i}].shape = {h.shape}')
"
```

**預期**：`[0]=embedding, [1]=stage1, [2]=stage2, [3]=stage3, [4]=stage4`

---

### ~~Step 1: Tokenizer~~ ✅ 已完成

**檔案**: `src/score/kern_tokenizer.py`

- Factorized encoding: `"4c#"` → `["4", "c#"]`
- Vocab size: ~220 tokens
- 支援 ties `[`, `]`, grace notes `q`, `Q`, `P`, bar lines 等

---

### Step 2: Audio Transform

**檔案**: `src/audio/transforms.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class LogMelTransform(nn.Module):
    """
    Waveform → Log-Mel Spectrogram

    參數設定（與 Zeng 一致）：
    - sample_rate: 16000
    - n_mels: 128
    - n_fft: 2048
    - hop_length: 256  # 16ms/frame @ 16kHz
    - f_min: 20.0
    - f_max: 8000.0
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 256,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.normalize = normalize

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [1, samples] or [samples]

        Returns:
            mel: [1, n_mels, T]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Mel spectrogram
        mel = self.mel_spec(waveform)  # [1, n_mels, T]

        # Log scale (add small epsilon for numerical stability)
        mel = torch.log(mel + 1e-9)

        # Normalize (optional, per-sample)
        if self.normalize:
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        return mel


def pad_mel_for_swin(mel: torch.Tensor, multiple: int = 32) -> tuple:
    """
    Pad mel spectrogram to be divisible by Swin's window size.

    Args:
        mel: [B, 1, 128, T]
        multiple: Swin window_size (8) × patch_size (4) = 32

    Returns:
        padded_mel: [B, 1, H_pad, W_pad]
        original_size: (H, W) for computing valid_ratios
    """
    B, C, H, W = mel.shape

    pad_h = (multiple - H % multiple) % multiple  # 128 通常已經是 32 的倍數
    pad_w = (multiple - W % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        mel = F.pad(mel, (0, pad_w, 0, pad_h), mode='constant', value=0)

    return mel, (H, W)
```

**時間對應關係**：
- `hop_length=256` @ 16kHz = 16ms/frame
- 1 分鐘音訊 = 3750 frames
- 4 分鐘音訊 = 15000 frames

---

### Step 3: Collate Function + ChunkedDataset

**檔案**:
- `src/datasets/chunked_dataset.py` — ChunkedDataset
- `src/datasets/collate.py` — ClefCollator + BucketSampler

#### 3.1 ChunkedDataset

```python
class ChunkedDataset(Dataset):
    """
    將長曲子切成重疊的 chunks，每個 chunk 當成獨立樣本

    8 分鐘曲子 → 3 個 chunks（每個 4 min，重疊 2 min）
    10 分鐘曲子 → 4 個 chunks
    4 分鐘以下 → 1 個 chunk（原曲）

    優點：
    - 每個 epoch 整首曲子都練到
    - 重疊區域學到「如何銜接」
    - 不浪費任何資料
    - 樣本數 +35%
    """

    def __init__(
        self,
        base_dataset: SynDataset,
        chunk_frames: int = 15000,      # 4 min @ 16ms/frame
        overlap_frames: int = 7500,     # 2 min overlap
        min_chunk_ratio: float = 0.5,   # 最後 chunk 至少 2 min
    ):
        self.base_dataset = base_dataset
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.stride = chunk_frames - overlap_frames  # 7500 frames = 2 min
        self.min_chunk_frames = int(chunk_frames * min_chunk_ratio)

        # 預計算所有 chunks
        self.chunks = self._create_chunks()

    def _create_chunks(self) -> List[Tuple[int, int, int]]:
        """
        Returns:
            List of (base_idx, start_frame, end_frame)
        """
        chunks = []
        for idx in range(len(self.base_dataset)):
            length = self.base_dataset.get_audio_length(idx)

            if length <= self.chunk_frames:
                # 短曲：整首當一個 chunk
                chunks.append((idx, 0, length))
            else:
                # 長曲：切成重疊的 chunks
                start = 0
                while start + self.min_chunk_frames <= length:
                    end = min(start + self.chunk_frames, length)
                    chunks.append((idx, start, end))

                    if end >= length:
                        break
                    start += self.stride

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        base_idx, start, end = self.chunks[idx]
        audio, kern, meta = self.base_dataset[base_idx]

        # 切 audio
        audio = audio[..., start:end]

        # 切對應的 kern（需要時間對齊）
        kern = self._slice_kern(kern, start, end, meta)

        return audio, kern, meta

    def get_audio_length(self, idx) -> int:
        """For BucketSampler"""
        _, start, end = self.chunks[idx]
        return end - start
```

#### 3.2 BucketSampler

```python
class BucketSampler(Sampler):
    """
    按音檔長度分組，減少 padding 浪費

    Bucket 設計：
      Bucket 0: < 1 min   (< 3750 frames)
      Bucket 1: 1-2 min   (3750-7500 frames)
      Bucket 2: 2-3 min   (7500-11250 frames)
      Bucket 3: 3-4 min   (11250-15000 frames)
      Bucket 4: > 4 min   (> 15000 frames, 需 crop)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        bucket_boundaries: List[int] = [3750, 7500, 11250, 15000],
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 預先計算每個樣本的 bucket
        self.buckets = self._assign_buckets()
```

#### 3.3 ClefCollator（含 valid_ratios）

```python
class ClefCollator:
    """
    處理變長序列的 collate function

    功能：
    - Bucket 內的樣本長度相近，padding 少
    - Pad to multiple of 32（Swin window constraint）
    - 計算 mel_valid_ratios 處理 padding
    """

    def __init__(
        self,
        tokenizer: KernTokenizer,
        audio_transform: LogMelTransform,
        max_seq_len: int = 4096,
        pad_to_multiple: int = 32,
    ):
        self.tokenizer = tokenizer
        self.audio_transform = audio_transform
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        mels = []
        input_ids_list = []
        labels_list = []

        for audio, kern, meta in batch:
            # Audio → Mel
            mel = self.audio_transform(audio)  # [1, 128, T]
            mels.append(mel)

            # Kern → Token IDs
            tokens = self.tokenizer.encode(kern)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            input_ids_list.append(tokens[:-1])  # 不含 <eos>
            labels_list.append(tokens[1:])      # 不含 <sos>

        # Pad mels to same length
        mel_lengths = [m.shape[-1] for m in mels]
        max_mel_len = max(mel_lengths)
        # Pad to multiple of 32
        max_mel_len = ((max_mel_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)

        padded_mels = []
        for mel in mels:
            pad_len = max_mel_len - mel.shape[-1]
            padded_mel = F.pad(mel, (0, pad_len))
            padded_mels.append(padded_mel)

        # 計算 valid ratios（處理 padding）
        mel_valid_ratios = torch.tensor([
            mel_len / max_mel_len for mel_len in mel_lengths
        ])  # [B]

        # Pad token sequences
        label_lengths = [len(ids) for ids in labels_list]
        max_seq_len = max(label_lengths)

        padded_input_ids = []
        padded_labels = []
        for inp, lab in zip(input_ids_list, labels_list):
            pad_len = max_seq_len - len(inp)
            padded_input_ids.append(inp + [self.tokenizer.pad_id] * pad_len)
            padded_labels.append(lab + [self.tokenizer.pad_id] * pad_len)

        return {
            'mel': torch.stack(padded_mels),
            'mel_lengths': torch.tensor(mel_lengths),
            'mel_valid_ratios': mel_valid_ratios,  # 新增：處理 padding
            'input_ids': torch.tensor(padded_input_ids),
            'labels': torch.tensor(padded_labels),
            'label_lengths': torch.tensor(label_lengths),
        }
```

---

### Step 4: Model 實作（核心）

#### 4.1 檔案結構

```
src/clef/
├── __init__.py
├── config.py                    # ClefConfig base class
├── attention.py                 # ClefAttention（Content-aware Learned-prior Event Focusing）
├── bridge.py                    # DeformableBridge
├── decoder.py                   # DeformableDecoderLayer
│
└── piano/
    ├── __init__.py
    ├── config.py                # ClefPianoConfig
    ├── model.py                 # ClefPianoBase
    ├── train.py
    └── inference.py

third_party/
└── deformable_detr/             # CUDA kernel（Phase 2）
```

#### 4.2 ClefAttention（關鍵模組）

**檔案**: `src/clef/attention.py`

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClefAttention(nn.Module):
    """
    CLEF Attention: Content-aware Learned-prior Event Focusing Attention

    命名由來：
    - Content-aware: 從 decoder hidden state 預測關注區域
    - Learned-prior: freq_prior / time_prior 學習「去哪裡看」
    - Event: 聚焦在音樂事件（音符、和弦）
    - Focusing: 稀疏採樣，只看重要的位置

    設計理念（基於認知科學）：
    - 人腦的 stream segregation（Bregman, 1990）是 content-dependent
    - freq_prior + time_prior 做「粗定位」
    - offset 只需要小範圍「局部細節」
    - 方形採樣 (2×2) 足夠，因為 prior 已經選對區域

    與 Stripe-Transformer / hFT-Transformer 的關係：
    - 他們用 full attention 分離 freq/time
    - 我們用 learned spatial priors + sparse sampling
    - 複雜度更低，效果相當（對 piano/solo 任務）

    通用性：
    - 音樂：Time × Frequency
    - 影像：X × Y × Scale
    - 可延伸到任何多維度 focusing 問題
    """

    def __init__(
        self,
        d_model: int = 512,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points_freq: int = 2,      # 頻率方向：局部細節
        n_points_time: int = 2,      # 時間方向：局部細節
        freq_offset_scale: float = 0.2,  # ±20%（prior 已定位）
        time_offset_scale: float = 0.1,  # ±10%
    ):
        super().__init__()

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points_freq = n_points_freq
        self.n_points_time = n_points_time
        self.n_points = n_points_freq * n_points_time  # 4
        self.freq_offset_scale = freq_offset_scale
        self.time_offset_scale = time_offset_scale

        # 分開預測 time 和 freq 的 offset
        self.time_offset_proj = nn.Linear(
            d_model,
            n_heads * n_levels * n_points_time
        )
        self.freq_offset_proj = nn.Linear(
            d_model,
            n_heads * n_levels * n_points_freq
        )

        # Attention weights: 對所有採樣點
        self.attention_weights = nn.Linear(
            d_model,
            n_heads * n_levels * self.n_points
        )

        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化：方形分布

        視覺化（n_points_time=2, n_points_freq=2）：

        頻率 ↑
             │           │
             │   · ─ ·   │  ← freq = +0.2
             │   │ × │   │  ← reference (freq_prior 決定)
             │   · ─ ·   │  ← freq = -0.2
             │           │
             └─────┬─────┘
                   │
              time: -0.1, +0.1

        重點：× 的位置由 freq_prior(tgt) 決定
        不是固定在 0.5，而是根據內容動態調整
        """
        # === Time offset 初始化 ===
        nn.init.constant_(self.time_offset_proj.weight, 0.)

        time_init = torch.linspace(
            -self.time_offset_scale,
            self.time_offset_scale,
            self.n_points_time
        )  # [-0.1, +0.1]

        time_bias = time_init.view(1, 1, self.n_points_time)
        time_bias = time_bias.expand(self.n_heads, self.n_levels, -1)

        with torch.no_grad():
            self.time_offset_proj.bias = nn.Parameter(time_bias.flatten())

        # === Freq offset 初始化 ===
        nn.init.constant_(self.freq_offset_proj.weight, 0.)

        freq_init = torch.linspace(
            -self.freq_offset_scale,
            self.freq_offset_scale,
            self.n_points_freq
        )  # [-0.2, +0.2]

        freq_bias = freq_init.view(1, 1, self.n_points_freq)
        freq_bias = freq_bias.expand(self.n_heads, self.n_levels, -1)

        with torch.no_grad():
            self.freq_offset_proj.bias = nn.Parameter(freq_bias.flatten())

        # === Attention weights 初始化：均勻 ===
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

        # === Projection layers: Xavier ===
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(
        self,
        query: torch.Tensor,              # [B, N_q, D]
        reference_points: torch.Tensor,   # [B, N_q, L, 2] 歸一化座標
        value: torch.Tensor,              # [B, N_v, D]
        spatial_shapes: torch.Tensor,     # [L, 2] 每個 level 的 (H, W)
        level_start_index: torch.Tensor,  # [L] 每個 level 在 value 中的起始 index
        valid_ratios: torch.Tensor = None # [B, L, 2] padding 的有效比例
    ) -> torch.Tensor:

        B, N_q, _ = query.shape
        B, N_v, _ = value.shape

        # === 1. 預測 offset ===

        # Time offset: [B, N_q, H, L, Kt]
        time_offset = self.time_offset_proj(query)  # [B, N_q, H*L*Kt]
        time_offset = time_offset.view(
            B, N_q, self.n_heads, self.n_levels, self.n_points_time
        )
        time_offset = time_offset.tanh() * self.time_offset_scale

        # Freq offset: [B, N_q, H, L, Kf]
        freq_offset = self.freq_offset_proj(query)  # [B, N_q, H*L*Kf]
        freq_offset = freq_offset.view(
            B, N_q, self.n_heads, self.n_levels, self.n_points_freq
        )
        freq_offset = freq_offset.tanh() * self.freq_offset_scale

        # === 2. 組合成 2D sampling grid ===

        time_grid = time_offset.unsqueeze(-1)  # [B, N_q, H, L, Kt, 1]
        freq_grid = freq_offset.unsqueeze(-2)  # [B, N_q, H, L, 1, Kf]

        # Stack 成 [B, N_q, H, L, Kt, Kf, 2]
        # 座標順序：(time, freq) 對應 (x, y)
        sampling_offsets = torch.stack([
            time_grid.expand(-1, -1, -1, -1, -1, self.n_points_freq),
            freq_grid.expand(-1, -1, -1, -1, self.n_points_time, -1),
        ], dim=-1)

        # Flatten 成 [B, N_q, H, L, K, 2]
        sampling_offsets = sampling_offsets.flatten(-3, -2)

        # === 3. 計算 sampling locations ===

        # Offset normalization: 除以每個 level 的空間尺寸
        offset_normalizer = torch.stack([
            spatial_shapes[..., 1],  # W (time)
            spatial_shapes[..., 0],  # H (freq)
        ], dim=-1)  # [L, 2]

        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )  # [B, N_q, H, L, K, 2]

        # 限制在有效範圍內
        if valid_ratios is not None:
            sampling_locations = sampling_locations * valid_ratios[:, None, None, :, None, :]

        # === 4. Attention weights ===

        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            B, N_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, N_q, self.n_heads, self.n_levels, self.n_points
        )

        # === 5. Deformable Attention 計算 ===

        value = self.value_proj(value)
        value = value.view(B, N_v, self.n_heads, self.d_model // self.n_heads)

        output = self._deformable_attention_core(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        return output

    def _deformable_attention_core(
        self,
        value: torch.Tensor,              # [B, N_v, H, D_head]
        spatial_shapes: torch.Tensor,     # [L, 2]
        level_start_index: torch.Tensor,  # [L]
        sampling_locations: torch.Tensor, # [B, N_q, H, L, K, 2]
        attention_weights: torch.Tensor,  # [B, N_q, H, L, K]
    ) -> torch.Tensor:
        """
        Pure PyTorch 實作（可替換為 CUDA kernel）
        """
        B, N_q, H, L, K, _ = sampling_locations.shape
        D_head = value.shape[-1]

        # 轉換 sampling_locations 到 grid_sample 格式 [-1, 1]
        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []

        for lid in range(L):
            H_l, W_l = spatial_shapes[lid].tolist()
            start = level_start_index[lid]
            end = start + H_l * W_l

            # 取出該 level 的 value
            value_l = value[:, start:end, :, :]
            value_l = value_l.permute(0, 2, 3, 1).reshape(B * H, D_head, H_l, W_l)

            # 取出該 level 的 sampling grid
            grid_l = sampling_grids[:, :, :, lid, :, :]
            grid_l = grid_l.permute(0, 2, 1, 3, 4).reshape(B * H, N_q, K, 2)

            # Bilinear sampling
            sampled = F.grid_sample(
                value_l,
                grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )

            sampled = sampled.view(B, H, D_head, N_q, K)
            sampling_value_list.append(sampled)

        # Stack levels
        sampling_values = torch.stack(sampling_value_list, dim=-2)

        # 加權求和
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).unsqueeze(2)
        output = (sampling_values * attention_weights).sum(dim=[-1, -2])
        output = output.permute(0, 3, 1, 2).flatten(-2)

        return output
```

#### 4.3 DeformableDecoderLayer（含 Content-Dependent Reference Points）

**檔案**: `src/clef/decoder.py`

```python
class DeformableDecoderLayer(nn.Module):
    """
    Decoder Layer with Content-Aware Deformable Cross-Attention

    1. Causal Self-Attention (標準)
    2. ClefAttention (方形採樣)
    3. FFN

    關鍵創新：Content-Dependent Reference Points
    - time_prior: 從 positional embedding 預測「看哪個時間點」
    - freq_prior: 從 decoder hidden state 預測「看高頻還是低頻」
    - 這對應人腦的 stream tracking（Bregman, 1990）

    鋼琴應用：
    - 預測右手旋律時 → freq_prior 輸出高值（看高頻區域）
    - 預測左手和聲時 → freq_prior 輸出低值（看低頻區域）
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points_freq: int = 2,       # 方形採樣
        n_points_time: int = 2,       # 方形採樣
        freq_offset_scale: float = 0.2,  # ±20%（prior 已定位）
        time_offset_scale: float = 0.1,  # ±10%
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_levels = n_levels

        # 1. Causal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. ClefAttention
        self.cross_attn = ClefAttention(
            d_model=d_model,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points_freq=n_points_freq,
            n_points_time=n_points_time,
            freq_offset_scale=freq_offset_scale,
            time_offset_scale=time_offset_scale,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 3. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        # Content-Dependent Reference Points（關鍵創新！）
        self.time_prior = nn.Linear(d_model, 1)  # 從 position embedding 預測時間
        self.freq_prior = nn.Linear(d_model, 1)  # 從 hidden state 預測頻率區域
        self.reference_refine = nn.Linear(d_model, 2)  # 基於內容微調（±10%）

        self._init_reference_predictors()

    def _init_reference_predictors(self):
        """Initialize reference point predictors.

        IMPORTANT: Cannot use zero weights!
        - Zero weights -> constant output -> zero gradients -> no learning
        - Use small Xavier init for weights (enables gradient flow)
        - Use zero bias (centers output at sigmoid(0)=0.5 or tanh(0)=0)
        """
        # Xavier init enables gradient flow through tgt_pos
        nn.init.xavier_uniform_(self.time_prior.weight, gain=0.001)
        nn.init.constant_(self.time_prior.bias, 0.)  # sigmoid(0) = 0.5

        # Xavier init enables gradient flow through tgt
        nn.init.xavier_uniform_(self.freq_prior.weight, gain=0.001)
        nn.init.constant_(self.freq_prior.bias, 0.)  # sigmoid(0) = 0.5

        # Refinement: small init to keep output close to zero initially
        nn.init.xavier_uniform_(self.reference_refine.weight, gain=0.001)
        nn.init.constant_(self.reference_refine.bias, 0.)  # tanh(0) = 0

    def forward(
        self,
        tgt: torch.Tensor,           # [B, S, D]
        memory: torch.Tensor,        # [B, N_total, D]
        tgt_mask: torch.Tensor,      # [S, S] causal mask
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,  # [B, L, 2]
        tgt_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        B, S, D = tgt.shape

        # Add position embedding
        if tgt_pos is not None:
            q = k = tgt + tgt_pos
        else:
            q = k = tgt

        # 1. Causal Self-Attention
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Content-Dependent Reference Points（關鍵創新！）
        # 時間 prior：從 positional embedding 預測「看哪個時間點」
        time_base = self.time_prior(tgt_pos).sigmoid()  # [B, S, 1]

        # 頻率 prior：從 hidden state 預測「看高頻還是低頻」
        # 這是與原設計的關鍵差異！
        freq_base = self.freq_prior(tgt).sigmoid()      # [B, S, 1]

        base_ref = torch.cat([time_base, freq_base], dim=-1)  # [B, S, 2]

        # 基於內容微調（±10%）
        refine = self.reference_refine(tgt).tanh() * 0.1  # [B, S, 2]

        # 最終 reference point
        reference_points = (base_ref + refine).clamp(0, 1)  # [B, S, 2]

        # Expand 到所有 levels（共用同一個座標）
        reference_points = reference_points[:, :, None, :].expand(
            -1, -1, self.n_levels, -1
        )  # [B, S, L, 2]

        # 3. ClefAttention
        tgt2 = self.cross_attn(
            tgt, reference_points, memory,
            spatial_shapes, level_start_index, valid_ratios
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 4. FFN
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        return tgt
```

#### 4.4 DeformableBridge

**檔案**: `src/clef/bridge.py`

```python
class DeformableBridge(nn.Module):
    """
    Multi-scale Deformable Bridge

    將 Swin 的 4 個 stage 輸出融合，使用 Deformable Self-Attention
    讓每個位置可以「看」不同 scale 的相關資訊
    """

    def __init__(
        self,
        swin_dims: List[int] = [96, 192, 384, 768],
        d_model: int = 512,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points_freq: int = 2,   # 方形採樣
        n_points_time: int = 2,   # 方形採樣
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels

        # 各 level 的投影層
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.LayerNorm(d_model),
            )
            for dim in swin_dims
        ])

        # Level embedding (區分不同 scale)
        self.level_embed = nn.Parameter(torch.zeros(n_levels, d_model))
        nn.init.normal_(self.level_embed)

        # Deformable Self-Attention layers
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_levels=n_levels,
                n_points_freq=n_points_freq,
                n_points_time=n_points_time,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        features: List[torch.Tensor],  # [F1, F2, F3, F4] from Swin
        mel_valid_ratios: torch.Tensor = None,  # [B] 時間軸有效比例
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, N_total, D] fused features
            spatial_shapes: [L, 2]
            level_start_index: [L]
            valid_ratios: [B, L, 2]
        """
        # Project each level to d_model
        src_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(features):
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))

            # Flatten: [B, C, H, W] → [B, H*W, D]
            feat = feat.flatten(2).transpose(1, 2)
            feat = self.input_projs[lvl](feat)

            # Add level embedding
            feat = feat + self.level_embed[lvl].view(1, 1, -1)
            src_flatten.append(feat)

        src_flatten = torch.cat(src_flatten, dim=1)  # [B, N_total, D]

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])

        # 計算 valid ratios（處理 padding）
        if mel_valid_ratios is not None:
            # 時間軸用 mel_valid_ratios，頻率軸總是完整的
            valid_ratios = torch.stack([
                torch.stack([mel_valid_ratios, torch.ones_like(mel_valid_ratios)], dim=-1)
                for _ in range(self.n_levels)
            ], dim=1)  # [B, L, 2]
        else:
            valid_ratios = torch.ones(B, self.n_levels, 2, device=src_flatten.device)

        # Reference points for self-attention
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )

        # Deformable self-attention layers
        output = src_flatten
        for layer in self.layers:
            output = layer(
                output, reference_points,
                spatial_shapes, level_start_index, valid_ratios
            )

        return output, spatial_shapes, level_start_index, valid_ratios

    def get_reference_points(self, spatial_shapes, valid_ratios, device):
        """為每個 token 生成初始 reference point"""
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
```

#### 4.5 ClefPianoBase

**檔案**: `src/clef/piano/model.py`

```python
class ClefPianoBase(nn.Module):
    """
    ISMIR 2026 版本：Swin V2 + ClefAttention

    架構特點：
    - Swin V2 (frozen) 提取 F1/F2/F3/F4 四個尺度特徵
    - Deformable Bridge: 稀疏自注意力融合多尺度特徵
    - ClefAttention Decoder: Content-aware Learned-prior Event Focusing
    - 解決 grace note 問題：可以直接存取 F1 (16ms 解析度)

    ClefAttention 特色：
    - freq_prior: 從內容預測「看高頻還是低頻」（stream tracking）
    - time_prior: 從位置預測「看哪個時間點」
    - 方形採樣 2×2: prior 已定位，offset 只需局部細節
    """

    def __init__(self, config: ClefPianoConfig):
        super().__init__()
        self.config = config

        # === Encoder: Swin V2 (frozen) ===
        self.swin = Swinv2Model.from_pretrained(
            config.swin_model,
            output_hidden_states=True,
        )
        if config.freeze_encoder:
            self.swin.eval()
            for p in self.swin.parameters():
                p.requires_grad = False

        # === Deformable Bridge ===
        self.bridge = DeformableBridge(
            swin_dims=config.swin_dims,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_levels=config.n_levels,
            n_points_freq=config.n_points_freq,
            n_points_time=config.n_points_time,
            n_layers=config.bridge_layers,
            dropout=config.dropout,
        )

        # === Decoder ===
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, config.max_seq_len, config.d_model)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_layers = nn.ModuleList([
            DeformableDecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_levels=config.n_levels,
                n_points_freq=config.n_points_freq,
                n_points_time=config.n_points_time,
                freq_offset_scale=config.freq_offset_scale,
                time_offset_scale=config.time_offset_scale,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.d_model)

        # === Output ===
        self.output_head = nn.Linear(config.d_model, config.vocab_size)

        self._causal_mask = None

    def encode(self, mel, mel_valid_ratios=None):
        """Encode audio to multi-scale features"""
        x = mel.repeat(1, 3, 1, 1)  # [B, 1, 128, T] → [B, 3, 128, T]

        with torch.no_grad() if self.config.freeze_encoder else nullcontext():
            swin_out = self.swin(x, output_hidden_states=True)

        # Extract all 4 stages
        features = [
            swin_out.hidden_states[i].transpose(1, 2).view(
                x.size(0), -1, *self._get_spatial_shape(i, mel.shape[-1])
            )
            for i in range(1, 5)
        ]

        memory, spatial_shapes, level_start_index, valid_ratios = self.bridge(
            features, mel_valid_ratios
        )
        return memory, spatial_shapes, level_start_index, valid_ratios

    def decode(self, input_ids, memory, spatial_shapes, level_start_index, valid_ratios):
        """Decode with ClefAttention (Content-aware Learned-prior Event Focusing)"""
        B, S = input_ids.shape

        tgt = self.token_embed(input_ids)
        tgt_pos = self.decoder_pos_embed[:, :S, :]
        tgt_mask = self._get_causal_mask(S, input_ids.device)

        for layer in self.decoder_layers:
            tgt = layer(
                tgt, memory, tgt_mask,
                spatial_shapes, level_start_index, valid_ratios,
                tgt_pos=tgt_pos,
            )

        tgt = self.decoder_norm(tgt)
        logits = self.output_head(tgt)
        return logits

    def forward(self, mel, input_ids, labels=None, mel_valid_ratios=None):
        memory, spatial_shapes, level_start_index, valid_ratios = self.encode(
            mel, mel_valid_ratios
        )
        logits = self.decode(
            input_ids, memory, spatial_shapes, level_start_index, valid_ratios
        )

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss
```

---

### Step 5: Training Script

**檔案**: `src/clef/piano/train.py`

**開發順序**：

```
Phase 1: Pure PyTorch 驗證
├── 用 F.grid_sample 實作 deformable attention
├── 不需要編譯 CUDA kernel
├── 目標：確認 pipeline 正確、loss 下降
└── 可加 @torch.compile 加速

Phase 2: CUDA Kernel 加速（需要時）
├── 編譯 Deformable DETR 的 CUDA ops
├── 安裝到個人 conda 環境
└── 預期加速 3-5 倍

Phase 3: 正式訓練
├── DDP + AMP (2x RTX 3090)
└── 預估：~1 天
```

**CUDA Kernel 安裝**：

```bash
# 不會影響其他人，安裝到個人 conda 環境

# 1. 準備環境
conda activate clef
module load cuda/11.8

# 2. 申請 GPU 節點
srun --gres=gpu:1 --time=1:00:00 --pty bash

# 3. Clone 和編譯
cd ~/projects/clef
git clone https://github.com/fundamentalvision/Deformable-DETR.git third_party/deformable_detr
cd third_party/deformable_detr/models/ops
python setup.py build install

# 4. 測試
python test.py
```

---

### Step 6: Sanity Checks

| 測試 | 方法 | 預期結果 |
|------|------|----------|
| **Silence Test** | 輸入全零 mel | `<sos> <eos>` 或休止符 |
| **Attention Viz** | 視覺化 reference points + sampling locations | 時間軸對角分布 |

**Attention 視覺化預期**：
- Reference points 大致沿時間軸對角分布
- 輸出 grace note 時，F1 的採樣點較密集
- 輸出和弦時，採樣點在頻率軸上分散

---

## Config 更新

**檔案**: `configs/clef_piano_base.yaml`

```yaml
model:
  name: "clef-piano-base"

  # Encoder: Swin V2
  swin_model: "microsoft/swinv2-tiny-patch4-window8-256"
  swin_dims: [96, 192, 384, 768]
  freeze_encoder: true

  # Deformable Attention（音樂專用）
  d_model: 512
  n_heads: 8
  n_levels: 4
  ff_dim: 2048
  dropout: 0.1

  # 方形採樣 + Content-Dependent Prior（關鍵設計）
  n_points_freq: 2        # 方形採樣（prior 已定位）
  n_points_time: 2        # 方形採樣
  freq_offset_scale: 0.2  # ±20%（局部細節）
  time_offset_scale: 0.1  # ±10%

  # Content-Dependent Reference Points（關鍵創新）
  use_time_prior: true    # time_prior(tgt_pos) → 時間定位
  use_freq_prior: true    # freq_prior(tgt) → 頻率區域選擇
  refine_range: 0.1       # ±10% 微調

  # Bridge
  bridge_layers: 2

  # Decoder
  decoder_layers: 6
  max_seq_len: 4096
  vocab_size: 512

data:
  audio:
    sample_rate: 16000
    n_mels: 128
    n_fft: 2048
    hop_length: 256
    f_min: 20.0
    f_max: 8000.0

  chunking:
    enabled: true
    chunk_frames: 15000     # 4 min
    overlap_frames: 7500    # 2 min overlap
    min_chunk_ratio: 0.5

training:
  distributed:
    enabled: true
    backend: "nccl"
    num_gpus: 2

  precision: "bf16"
  batch_size: 3             # per GPU
  gradient_accumulation_steps: 1
  effective_batch_size: 6

  learning_rate: 1.0e-4
  weight_decay: 0.01
  max_epochs: 50
  warmup_steps: 1000
  gradient_clip: 1.0

  save_every_n_epochs: 10
  save_best: true
  save_last: true
  early_stopping_patience: 15

  wandb:
    enabled: true
    project: "clef-piano-base"
    tags: ["piano", "a2s", "swin", "harmonic-deformable", "ddp"]
```

---

## 記憶體估算

| Duration | Frames | Total Tokens | Training (B=2) | 備註 |
|----------|--------|--------------|----------------|------|
| 2min | 7,500 | ~80K | ~10 GB | 開發測試 |
| 3min | 11,250 | ~120K | ~14 GB | 安全選擇 |
| **4min** | **15,000** | **~160K** | **~18 GB** | 推薦設定 |
| 5min | 18,750 | ~200K | ~22 GB | 接近上限 |

**RTX 3090 (24GB) 建議**：4 min crop + batch=2 + grad_accum=4 → 有效 batch=8

---

## 參數量估算

| Component | 參數量 |
|-----------|--------|
| Swin V2 Tiny (FROZEN) | (29M) |
| Input Projections (4 levels) | 0.74M |
| Level Embedding | 0.002M |
| Deformable Bridge (2 layers) | ~5.6M |
| Token + Position Embedding | 2.36M |
| Decoder (6 layers) | ~23M |
| Output Head | 0.26M |
| **TRAINABLE TOTAL** | **~32M** |
| **MODEL TOTAL** | **~61M** |

---

## 實作優先順序

| 順序 | 任務 | 檔案 | 狀態 |
|------|------|------|------|
| ~~0~~ | ~~Tokenizer~~ | `src/score/kern_tokenizer.py` | ✅ 已完成 |
| ~~0~~ | ~~SynDataset~~ | `src/datasets/syn_dataset.py` | ✅ 已完成 |
| 1 | Swin hidden_states 格式測試 | (script) | ⬜ 待執行 |
| 2 | Audio Transform | `src/audio/transforms.py` | ⬜ 待實作 |
| 3 | ClefAttention | `src/clef/attention.py` | ⬜ 待實作 |
| 4 | DeformableBridge | `src/clef/bridge.py` | ⬜ 待實作 |
| 5 | DeformableDecoderLayer | `src/clef/decoder.py` | ⬜ 待實作 |
| 6 | ClefPianoBase | `src/clef/piano/model.py` | ⬜ 待實作 |
| 7 | ChunkedDataset + Collate | `src/datasets/` | ⬜ 待實作 |
| 8 | Train Script | `src/clef/piano/train.py` | ⬜ 待實作 |
| 9 | Sanity Check | - | ⬜ 待執行 |
| 10 | Inference | `src/clef/piano/inference.py` | ⬜ 待實作 |

**建議下一步**：先執行 Step 1 (Swin 格式測試)，確認 hidden_states 格式後再開始實作模型

---

## 驗證方式

```bash
# 1. Swin 格式測試
python -c "
from transformers import Swinv2Model
import torch
model = Swinv2Model.from_pretrained('microsoft/swinv2-tiny-patch4-window8-256', output_hidden_states=True)
x = torch.randn(1, 3, 128, 256)
out = model(x)
for i, h in enumerate(out.hidden_states):
    print(f'hidden_states[{i}].shape = {h.shape}')
"

# 2. Tokenizer 測試
python -c "from src.score.kern_tokenizer import KernTokenizer; t = KernTokenizer(); print(t.vocab_size)"

# 3. Overfit One Batch
python -m src.clef.piano.train --config configs/clef_piano_base.yaml --sanity-check

# 4. DDP 測試
torchrun --nproc_per_node=2 -m src.clef.piano.train --config configs/clef_piano_base.yaml --max-epochs 1
```

---

## Ablation 建議

訓練穩定後，可以做這些 ablation：

### ISMIR 2026（簡單提及）

| 實驗 | 設定 | 預期 |
|------|------|------|
| 我們的設計 | 2×2 + freq_prior + time_prior | Baseline |

### ICLR 2027（詳細比較）

| 實驗 | 設定 | 預期結果 | 論文意義 |
|------|------|----------|----------|
| Fixed freq_base | freq_base=0.5 固定 | 較差 | 驗證 freq_prior 的價值 |
| Learned freq_prior | freq_prior(tgt) | **較好** | 主要創新 |
| 垂直長條 6×2 | n_points=12, fixed freq | 相近或略好 | 過度設計，參數多 |
| 方形 2×2 + prior | n_points=4, learned prior | **相近** | KISS 原則 |
| 無 time_prior | time_base 固定 | 較差 | 驗證時間定位的價值 |

**預期 Finding**：
> "簡單的 learned prior + 2×2 方形採樣 ≈ 複雜的 6×2 長條設計，但參數更少、更 elegant"

---

## 檢查清單

### 已完成 ✅
| 項目 | 檔案 |
|------|------|
| Tokenizer (Factorized) | `src/score/kern_tokenizer.py` |
| SynDataset | `src/datasets/syn_dataset.py` |

### 待實作 ⬜
| 項目 | 說明 |
|------|------|
| Swin hidden_states 格式 | 先跑測試確認 index |
| Reference point 共用 | `[B, S, 2]` expand to levels |
| **time_prior** | 從 position embedding 預測時間位置 |
| **freq_prior** | 從 decoder hidden state 預測頻率區域（關鍵創新！）|
| ClefAttention | **方形採樣 2×2** |
| 分離 time/freq offset | `time_offset_proj` + `freq_offset_proj` |
| 初始化（方形分布） | freq 均勻 ±0.2, time 均勻 ±0.1 |
| Offset normalization | 除以 spatial_shapes |
| Valid ratios | 處理 padding |
| Pure PyTorch 實作 | 用 F.grid_sample |
| Sanity checks | Overfit、Silence、Attention viz |

### 設計決策記錄
| 決策 | 原因 |
|------|------|
| freq_prior 從 tgt 預測 | 人腦的 stream tracking 是 content-dependent（Bregman, 1990）|
| 方形採樣 2×2 | prior 已做粗定位，offset 只需看局部細節 |
| freq_offset_scale=0.2 | 比原本 0.5 小，因為 freq_prior 已定位 |
| 與 Stripe/hFT 關係 | 用 learned prior 取代 full attention，複雜度更低 |
