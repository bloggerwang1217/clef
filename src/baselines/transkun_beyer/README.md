# Transkun + Beyer Pipeline

執行 ASAP 資料集的完整 Audio-to-Score 評估。

## 版本資訊

| 工具 | 版本 | 備註 |
|------|------|------|
| Transkun | 2.0.1 | pip 安裝，使用 `v2` 模型 |
| Beyer | v0.0.1 + 11 commits | commit `115432b` (2024-10-11) |
| Beyer Checkpoint | `MIDI2ScoreTF.ckpt` | 389MB, 2024-08-21 release |

Transkun 支援模型：`v2`（預設）、`v2_aug`、`v2_no_ext`

## 快速開始

### 方案 A: 平行執行 (推薦)

開兩個 tmux pane：

```bash
# Pane 1: GPU 1 跑 Transkun
cd /home/bloggerwang/clef
GPU_ID=1 bash scripts/transkun_beyer_pipeline/01_run_transkun.sh

# Pane 2: GPU 4 跑 Beyer (watch mode，會持續檢查新的 MIDI)
cd /home/bloggerwang/clef
GPU_ID=4 WATCH_MODE=true bash scripts/transkun_beyer_pipeline/02_run_beyer.sh
```

等兩邊都跑完後：

```bash
# 跑 MV2H 評估
bash scripts/transkun_beyer_pipeline/03_run_mv2h.sh
```

### 方案 B: 循序執行

```bash
cd /home/bloggerwang/clef

# Step 1: Transkun (GPU 1)
GPU_ID=1 bash scripts/transkun_beyer_pipeline/01_run_transkun.sh

# Step 2: Beyer (GPU 4)
GPU_ID=4 bash scripts/transkun_beyer_pipeline/02_run_beyer.sh

# Step 3: MV2H Evaluation
bash scripts/transkun_beyer_pipeline/03_run_mv2h.sh
```

## 環境變數

### 01_run_transkun.sh
| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ASAP_DIR` | `/home/bloggerwang/asap-dataset` | ASAP 資料集路徑 |
| `OUTPUT_DIR` | `data/experiments/transkun_beyer` | 輸出目錄 |
| `GPU_ID` | `1` | GPU 編號 |
| `SKIP_EXISTING` | `true` | 跳過已存在的檔案 |

### 02_run_beyer.sh
| 變數 | 預設值 | 說明 |
|------|--------|------|
| `INPUT_DIR` | `data/experiments/transkun_beyer/midi_from_transkun` | MIDI 輸入目錄 |
| `OUTPUT_DIR` | `data/experiments/transkun_beyer` | 輸出目錄 |
| `GPU_ID` | `4` | GPU 編號 |
| `SKIP_EXISTING` | `true` | 跳過已存在的檔案 |
| `WATCH_MODE` | `false` | 持續監控新檔案 |

### 03_run_mv2h.sh
| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODE` | `chunks` | 評估模式: `full` 或 `chunks` |
| `WORKERS` | `8` | 平行 worker 數量 |

## 輸出結構

```
data/experiments/transkun_beyer/
├── midi_from_transkun/     # Transkun 輸出的 MIDI
├── musicxml_from_beyer/    # Beyer 輸出的 MusicXML
├── chunk_midi/             # 提取的 5-bar chunks
├── results/
│   ├── chunks.csv          # 每個 chunk 的評估結果
│   └── chunks.summary.json # 彙總統計
├── transkun.log
└── beyer.log
```

## 進度監控

```bash
# 查看 Transkun 進度
ls data/experiments/transkun_beyer/midi_from_transkun/ | wc -l

# 查看 Beyer 進度
ls data/experiments/transkun_beyer/musicxml_from_beyer/ | wc -l

# 查看錯誤
tail -f data/experiments/transkun_beyer/transkun.log
tail -f data/experiments/transkun_beyer/beyer.log
```
