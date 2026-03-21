"""
Inference script for ClefPianoTiny model.

Two inference modes:
  time  — Full-song mel → overlapping 10 s chunks → merge → full-song kern.
           Output: data/experiments/clef_piano_base/test_kern_pred/{id}.krn
  bar5  — 5-bar windows aligned to measure boundaries (from
           augmentation_metadata.json) → one kern file per chunk.
           Output: data/experiments/clef_piano_base/test_kern_pred_5_bar/{id}.{n}.krn

Usage:
    # Time-based (full-song):
    poetry run python src/clef/piano/clef_piano_tiny_inference.py \\
        --checkpoint checkpoints/clef_piano_tiny/best.pt \\
        --mode time

    # 5-bar chunk:
    poetry run python src/clef/piano/clef_piano_tiny_inference.py \\
        --checkpoint checkpoints/clef_piano_tiny/best.pt \\
        --mode bar5 \\
        --metadata data/experiments/clef_piano_base/augmentation_metadata.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import converter21
import music21
import torch
from tqdm import tqdm

from src.clef.piano.clef_piano_tiny import ClefPianoTiny
from src.clef.piano.tokenizer import KernTokenizer
from src.score.reconstruct_kern import reconstruct_kern_from_token_ids

converter21.register()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MEL_FPS = 100
DEFAULT_CHUNK_FRAMES = 1000   # 10 s @ 100 fps
DEFAULT_OVERLAP_FRAMES = 200  # 2 s overlap
DEFAULT_MAX_LEN = 512


# =============================================================================
# Model loading
# =============================================================================


def load_model(checkpoint_path: str, device: str = "cuda") -> ClefPianoTiny:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    logger.info(f"Loaded config from checkpoint (vocab_size={config.vocab_size})")
    model = ClefPianoTiny(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    return model


# =============================================================================
# Core generation helpers (shared by both modes)
# =============================================================================


@torch.no_grad()
def _beam_search(
    model: ClefPianoTiny,
    mel_chunk: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    num_beams: int,
    length_penalty: float = 1.0,
) -> List[int]:
    """Beam search for ClefPianoTiny (O(n²) decoder, correct).

    Encoder outputs are constant across beams — expanded each step from
    the original single-batch tensors, so no beam-reorder needed on encoder.
    """
    NEG_INF = float("-inf")
    device = mel_chunk.device

    memory, ss, lsi, vr = model.encode(mel_chunk)  # [1, N_kv, D]

    seqs = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)

    for _ in range(max_len - 1):
        B_cur = seqs.shape[0]
        mem_b = memory.expand(B_cur, -1, -1).contiguous()
        vr_b = vr.expand(B_cur, -1, -1).contiguous()

        tgt = model.token_embed(seqs).contiguous()
        if tgt.shape[1] == 1:
            tgt = tgt.repeat(1, 2, 1)  # Mamba2 S=1 stride workaround
        dec_out = model.decoder(tgt, mem_b, ss, lsi, vr_b, input_ids=seqs)
        if isinstance(dec_out, tuple):
            dec_out = dec_out[0]

        logits = model.output_projection(dec_out[:, -1:, :])  # [B, 1, vocab]
        log_p = torch.log_softmax(logits[:, 0, :], dim=-1)    # [B, vocab]

        done = seqs[:, -1].eq(eos_id)
        log_p[done, :] = NEG_INF
        log_p[done, eos_id] = 0.0

        cand = scores.unsqueeze(1) + log_p
        flat = cand.reshape(-1)
        top_scores, top_idx = flat.topk(num_beams)
        beam_idx = top_idx // logits.shape[-1]
        token_idx = top_idx % logits.shape[-1]

        seqs = torch.cat([seqs[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        if seqs[:, -1].eq(eos_id).all():
            break

    pen_scores = scores / (seqs.shape[1] ** length_penalty)
    best = pen_scores.argmax()
    return seqs[best].tolist()


@torch.no_grad()
def generate_kern_chunk(
    model: ClefPianoTiny,
    mel_chunk: torch.Tensor,
    tokenizer: KernTokenizer,
    max_len: int = DEFAULT_MAX_LEN,
    num_beams: int = 1,
    device: str = "cuda",
) -> List[int]:
    """Run greedy (num_beams=1) or beam search generation on a single mel chunk [1, 1, 128, T].

    Returns:
        Token IDs excluding BOS/EOS/continue.
    """
    bos_id = tokenizer.vocab["<sos>"]
    eos_id = tokenizer.vocab["<eos>"]
    continue_id = tokenizer.vocab.get("<continue>", -1)

    mel_chunk = mel_chunk.to(device)

    if num_beams > 1:
        ids = _beam_search(model, mel_chunk, bos_id, eos_id, max_len, num_beams)
    else:
        generated_ids = model.generate(
            mel_chunk, max_len=max_len, bos_token_id=bos_id, eos_token_id=eos_id
        )
        ids = generated_ids[0].tolist()

    if ids and ids[0] == bos_id:
        ids = ids[1:]
    # Truncate at the first <eos> or <continue> token.
    stop_ids = {eos_id, continue_id}
    for i, tok in enumerate(ids):
        if tok in stop_ids:
            ids = ids[:i]
            break
    return ids


def truncate_token_ids_to_n_bars(token_ids: List[int], bar_id: int, n_bars: int) -> List[int]:
    """Keep only the first n_bars bars from a token sequence.

    Counts <bar> tokens; once n_bars have been seen, drops everything after.
    """
    count = 0
    for i, tok in enumerate(token_ids):
        if tok == bar_id:
            count += 1
            if count > n_bars:
                return token_ids[:i]
    return token_ids


def split_tokens_by_bar(token_ids: List[int], bar_id: int) -> List[List[int]]:
    """Split token sequence into measures at <bar> tokens."""
    measures: List[List[int]] = []
    current: List[int] = []
    for tok in token_ids:
        current.append(tok)
        if tok == bar_id:
            measures.append(current)
            current = []
    if current:
        measures.append(current)
    return measures


def estimate_measure_end_times(
    measures: List[List[int]],
    chunk_start_sec: float,
    chunk_duration_sec: float,
) -> List[float]:
    total_tokens = sum(len(m) for m in measures)
    if total_tokens == 0:
        return []
    end_times: List[float] = []
    cumulative = 0
    for m in measures:
        cumulative += len(m)
        end_times.append(
            chunk_start_sec + (cumulative / total_tokens) * chunk_duration_sec
        )
    return end_times


def find_cut_measure_idx(end_times: List[float], cut_time: float) -> int:
    for i, t in enumerate(end_times):
        if t >= cut_time:
            return i
    return len(end_times)


def merge_chunk_measures(
    all_chunk_measures: List[List[List[int]]],
    chunk_boundaries: List[Tuple[float, float]],
) -> List[int]:
    """Barline-aligned center-cut merge of overlapping chunks."""
    n_chunks = len(all_chunk_measures)
    if n_chunks == 1:
        return [tok for m in all_chunk_measures[0] for tok in m]

    result_tokens: List[int] = []
    for i in range(n_chunks):
        measures = all_chunk_measures[i]
        start_sec, end_sec = chunk_boundaries[i]
        if not measures:
            continue

        end_times = estimate_measure_end_times(measures, start_sec, end_sec - start_sec)

        if i == 0:
            cut_time = (chunk_boundaries[i + 1][0] + end_sec) / 2.0
            cut_idx = find_cut_measure_idx(end_times, cut_time)
            selected = measures[:cut_idx]
        elif i == n_chunks - 1:
            cut_time = (start_sec + chunk_boundaries[i - 1][1]) / 2.0
            cut_idx = find_cut_measure_idx(end_times, cut_time)
            selected = measures[cut_idx:]
        else:
            left_cut = (start_sec + chunk_boundaries[i - 1][1]) / 2.0
            right_cut = (chunk_boundaries[i + 1][0] + end_sec) / 2.0
            left_idx = find_cut_measure_idx(end_times, left_cut)
            right_idx = find_cut_measure_idx(end_times, right_cut)
            selected = measures[left_idx:right_idx]

        for m in selected:
            result_tokens.extend(m)
    return result_tokens


def kern_to_midi(kern_content: str, output_midi_path: str) -> bool:
    """Convert kern string to MIDI via MusicXML.

    kern (tempfile) → kern_to_musicxml() → MusicXML (tempfile) → MIDI
    Uses src.score.generate_score.kern_to_musicxml for correct converter21 parsing.
    """
    import tempfile, os
    import music21 as _music21
    from src.score.generate_score import kern_to_musicxml

    tmp_krn = tmp_xml = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w") as f:
            tmp_krn = f.name
            f.write(kern_content)
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as f:
            tmp_xml = f.name
        kern_to_musicxml(tmp_krn, tmp_xml)
        score = _music21.converter.parse(tmp_xml)
        score.write("midi", fp=output_midi_path)
        return True
    except Exception as e:
        logger.warning(f"kern → MIDI failed: {e}")
        return False
    finally:
        for p in [tmp_krn, tmp_xml]:
            if p and os.path.exists(p):
                os.remove(p)


# =============================================================================
# Mode: time — full-song with overlapping time-based chunks
# =============================================================================


@torch.no_grad()
def generate_kern_time(
    model: ClefPianoTiny,
    mel_path: str,
    tokenizer: KernTokenizer,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = DEFAULT_OVERLAP_FRAMES,
    max_len: int = DEFAULT_MAX_LEN,
    num_beams: int = 1,
    device: str = "cuda",
) -> str:
    """Generate full-song **kern using overlapping time-based chunks.

    Args:
        chunk_frames:   Mel frames per chunk (default 1000 = 10 s @ 100 fps).
        overlap_frames: Overlap between adjacent chunks (default 200 = 2 s).
    """
    mel = torch.load(mel_path, map_location="cpu", weights_only=True)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # [128, T] → [1, 128, T]

    total_frames = mel.shape[-1]
    stride = chunk_frames - overlap_frames

    chunks: List[Tuple[int, int, torch.Tensor]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        mel_chunk = mel[:, :, start:end].unsqueeze(0)  # [1, 1, 128, T_chunk]
        chunks.append((start, end, mel_chunk))
        if end >= total_frames:
            break
        start += stride

    logger.info(
        f"  {total_frames} frames ({total_frames / MEL_FPS:.1f} s) "
        f"→ {len(chunks)} chunks"
    )

    bar_id = tokenizer.vocab["<bar>"]
    all_chunk_measures: List[List[List[int]]] = []
    chunk_boundaries: List[Tuple[float, float]] = []

    for i, (start_f, end_f, mel_chunk) in enumerate(chunks):
        start_sec = start_f / MEL_FPS
        end_sec = end_f / MEL_FPS
        token_ids = generate_kern_chunk(model, mel_chunk, tokenizer, max_len, num_beams, device)
        measures = split_tokens_by_bar(token_ids, bar_id)
        logger.debug(
            f"  Chunk {i + 1}/{len(chunks)} [{start_sec:.1f} s-{end_sec:.1f} s]: "
            f"{len(token_ids)} tokens, {len(measures)} measures"
        )
        all_chunk_measures.append(measures)
        chunk_boundaries.append((start_sec, end_sec))

    if len(chunks) == 1:
        merged_tokens = [tok for m in all_chunk_measures[0] for tok in m]
    else:
        merged_tokens = merge_chunk_measures(all_chunk_measures, chunk_boundaries)

    logger.info(f"  Merged: {len(merged_tokens)} tokens")
    return reconstruct_kern_from_token_ids(merged_tokens, tokenizer)


def run_time_mode(args) -> None:
    """Full-song inference with time-based chunking."""
    kern_dir = Path(args.output_dir)
    kern_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = (
        Path(args.output_midi_dir)
        if args.output_midi_dir
        else Path(str(kern_dir) + "_midi")
    )
    midi_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)
    tokenizer = KernTokenizer()

    with open(args.manifest) as f:
        manifest = json.load(f)
    manifest_dir = Path(args.manifest_dir)
    logger.info(f"{len(manifest)} samples")

    if args.max_samples:
        manifest = manifest[: args.max_samples]

    kern_ok = midi_ok = 0
    for idx, item in enumerate(tqdm(manifest, desc="Inference (time)"), 1):
        sample_id = item["id"]
        mel_path = manifest_dir / item["mel_path"]

        kern_out = kern_dir / f"{sample_id}.krn"
        midi_out = midi_dir / f"{sample_id}.mid"

        if args.skip_existing and kern_out.exists():
            kern_ok += 1
            if midi_out.exists():
                midi_ok += 1
            continue

        logger.info(f"[{idx}/{len(manifest)}] {sample_id}")

        pred_kern = generate_kern_time(
            model=model,
            mel_path=str(mel_path),
            tokenizer=tokenizer,
            chunk_frames=args.chunk_frames,
            overlap_frames=args.overlap_frames,
            max_len=args.max_len,
            num_beams=args.num_beams,
            device=args.device,
        )

        with open(kern_out, "w") as f:
            f.write(pred_kern)
        kern_ok += 1

        if kern_to_midi(pred_kern, str(midi_out)):
            midi_ok += 1

    logger.info(f"Done — kern: {kern_ok}/{len(manifest)} → {kern_dir}")
    logger.info(f"       midi: {midi_ok}/{len(manifest)} → {midi_dir}")


# =============================================================================
# Mode: bar5 — one kern file per 5-bar window, aligned to measure boundaries
# =============================================================================


def _pad_or_crop(mel_slice: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad (zero) or crop mel_slice [C, T] to target_frames."""
    T = mel_slice.shape[-1]
    if T < target_frames:
        pad = torch.zeros(*mel_slice.shape[:-1], target_frames - T)
        return torch.cat([mel_slice, pad], dim=-1)
    return mel_slice[..., :target_frames]


@torch.no_grad()
def run_bar5_mode(args) -> None:
    """5-bar chunk inference aligned to measure boundaries from metadata."""
    if not args.metadata:
        raise ValueError("--metadata is required for bar5 mode")

    kern_dir = Path(args.output_dir)
    kern_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = (
        Path(args.output_midi_dir)
        if args.output_midi_dir
        else Path(str(kern_dir) + "_midi")
    )
    midi_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)
    tokenizer = KernTokenizer()

    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.metadata) as f:
        metadata = json.load(f)

    manifest_dir = Path(args.manifest_dir)
    n_bars = args.n_bars
    chunk_frames = args.chunk_frames

    if args.max_samples:
        manifest = manifest[: args.max_samples]

    kern_ok = midi_ok = total = 0
    max_chunks = args.max_chunks

    for item in tqdm(manifest, desc="Inference (bar5)"):
        if max_chunks is not None and total >= max_chunks:
            break

        perf_id = item["id"]
        if perf_id not in metadata:
            logger.warning(f"No metadata for {perf_id}, skipping")
            continue

        meta = metadata[perf_id]
        audio_measures: List[dict] = meta.get("audio_measures", [])
        mel_path_abs = manifest_dir / item["mel_path"]

        if not mel_path_abs.exists():
            logger.warning(f"Mel missing: {mel_path_abs}, skipping")
            continue

        mel = torch.load(str(mel_path_abs), map_location="cpu", weights_only=True)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [C, T] → [1, C, T]
        total_frames = mel.shape[-1]

        n_measures = len(audio_measures)
        if n_measures < n_bars:
            continue

        logger.info(f"{perf_id}: {n_measures} measures → {n_measures - n_bars + 1} windows")

        for i, window in enumerate(
            audio_measures[j : j + n_bars]
            for j in range(0, n_measures - n_bars + 1)
        ):
            if max_chunks is not None and total >= max_chunks:
                break
            kern_out = kern_dir / f"{perf_id}.{i}.krn"
            midi_out = midi_dir / f"{perf_id}.{i}.mid"

            if args.skip_existing and kern_out.exists():
                kern_ok += 1
                if midi_out.exists():
                    midi_ok += 1
                total += 1
                continue

            start_frame = int(window[0]["start_sec"] * MEL_FPS)
            end_frame = min(int(window[-1]["end_sec"] * MEL_FPS), total_frames)

            mel_slice = mel[:, :, start_frame:end_frame]
            mel_slice = _pad_or_crop(mel_slice, chunk_frames)
            mel_input = mel_slice.unsqueeze(0)  # [1, 1, 128, chunk_frames]

            token_ids = generate_kern_chunk(
                model, mel_input, tokenizer, args.max_len, args.num_beams, args.device
            )
            bar_id = tokenizer.vocab["<bar>"]
            token_ids = truncate_token_ids_to_n_bars(token_ids, bar_id, n_bars)
            pred_kern = reconstruct_kern_from_token_ids(token_ids, tokenizer)

            with open(kern_out, "w") as f:
                f.write(pred_kern)
            kern_ok += 1

            if kern_to_midi(pred_kern, str(midi_out)):
                midi_ok += 1

            total += 1

    logger.info(f"Done — kern: {kern_ok}/{total} → {kern_dir}")
    logger.info(f"       midi: {midi_ok}/{total} → {midi_dir}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for ClefPianoTiny")

    # Shared args
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--mode",
        choices=["time", "bar5"],
        default="time",
        help="time: full-song with sliding time window; bar5: 5-bar measure-aligned chunks",
    )
    parser.add_argument(
        "--manifest",
        default="data/experiments/clef_piano_base/test_manifest.json",
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/experiments/clef_piano_base",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output kern directory. "
            "Defaults to data/experiments/clef_piano_base/test_kern_pred (time) "
            "or data/experiments/clef_piano_base/test_kern_pred_5_bar (bar5)."
        ),
    )
    parser.add_argument("--output-midi-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help="Mel frames per chunk (default 1000 = 10 s @ 100 fps)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Max tokens per chunk",
    )
    parser.add_argument("--skip-existing", action="store_true")

    # time-mode only
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=DEFAULT_OVERLAP_FRAMES,
        help="(time mode) Overlap between adjacent chunks (default 200 = 2 s)",
    )

    # bar5-mode only
    parser.add_argument(
        "--metadata",
        default=None,
        help="(bar5 mode) Path to augmentation_metadata.json",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5,
        help="(bar5 mode) Bars per chunk (default 5)",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width (1 = greedy)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of performances (for smoke tests)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="(bar5 mode) Limit total number of chunks (for smoke tests)",
    )

    args = parser.parse_args()

    # Resolve default output directory by mode.
    if args.output_dir is None:
        base = "data/experiments/clef_piano_base"
        args.output_dir = (
            f"{base}/test_kern_pred"
            if args.mode == "time"
            else f"{base}/test_kern_pred_5_bar"
        )

    if args.mode == "time":
        run_time_mode(args)
    else:
        run_bar5_mode(args)


if __name__ == "__main__":
    main()
