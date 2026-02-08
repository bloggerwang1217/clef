"""
Inference script for clef-piano-base model.

Generates **kern predictions and converts to MIDI for evaluation.

Pipeline:
1. Audio (mel) → **kern (autoregressive generation)
2. **kern → MIDI (via converter21)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import converter21
import music21
import torch
from tqdm import tqdm

from src.clef.piano.config import ClefPianoConfig
from src.clef.piano.model import ClefPianoBase
from src.clef.piano.tokenizer import KernTokenizer
from src.score.reconstruct_kern import reconstruct_kern_from_token_ids

# Register converter21 for kern→MIDI conversion
converter21.register()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config_path: str, device: str = 'cuda') -> ClefPianoBase:
    """Load model from checkpoint."""
    # Load checkpoint (weights_only=False for backward compatibility with PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Prefer config from checkpoint (has correct vocab_size, n_freq_groups, etc.)
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info(f'Using config from checkpoint (vocab_size={config.vocab_size})')
    else:
        config = ClefPianoConfig.from_yaml(config_path)
        logger.info(f'Using config from {config_path}')

    model = ClefPianoBase(config)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f'Loaded model from {checkpoint_path}')
    logger.info(f'Model has {model.get_num_params():,} parameters')

    return model


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load test manifest."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    logger.info(f'Loaded {len(manifest)} samples from {manifest_path}')
    return manifest


@torch.no_grad()
def generate_kern_chunk(
    model: ClefPianoBase,
    mel_chunk: torch.Tensor,
    tokenizer: KernTokenizer,
    max_length: int = 16384,
    device: str = 'cuda',
) -> List[int]:
    """Generate token IDs from a single mel chunk.

    Uses encoder caching (encode once) and decoder state cache for efficient
    autoregressive generation. Each step only processes the new token.

    Supports hybrid Mamba+SA decoder: Mamba layers use shared InferenceParams,
    SA layers use KV-cache.

    Args:
        model: Trained ClefPianoBase model
        mel_chunk: Mel spectrogram chunk [1, 1, n_mels, n_frames]
        tokenizer: KernTokenizer instance
        max_length: Maximum sequence length for this chunk
        device: Device to run on

    Returns:
        List of generated token IDs (excluding <sos>).
    """
    sos_id = tokenizer.vocab['<sos>']
    eos_id = tokenizer.vocab['<eos>']
    continue_id = tokenizer.vocab['<continue>']

    # Encode mel once (Swin + Bridge)
    memory, spatial_shapes, level_start_index, valid_ratios = model.encode(mel_chunk)

    # Pre-compute cross-attention value projections (once per chunk)
    value_cache = model.prepare_value_cache(memory, spatial_shapes, level_start_index)

    # Initialize inference states (Mamba InferenceParams + SA KV-cache)
    past_states = model._init_inference_states(1, max_length, device)

    # First step: decode <sos>
    input_ids = torch.tensor([[sos_id]], device=device)
    logits, past_states = model.decode(
        input_ids, memory, spatial_shapes, level_start_index, valid_ratios,
        past_states=past_states, use_cache=True, value_cache=value_cache,
    )

    next_id = logits[:, -1, :].argmax(dim=-1).item()
    generated: List[int] = []

    if next_id in (eos_id, continue_id):
        return generated

    generated.append(next_id)

    # Subsequent steps: decode one token at a time with cache
    for step in range(1, max_length):
        input_ids = torch.tensor([[next_id]], device=device)

        logits, past_states = model.decode(
            input_ids, memory, spatial_shapes, level_start_index, valid_ratios,
            past_states=past_states, use_cache=True, value_cache=value_cache,
        )

        next_id = logits[:, -1, :].argmax(dim=-1).item()

        if next_id == eos_id:
            logger.debug(f'Chunk: generated {step + 1} tokens, stopped at <eos>')
            break

        if next_id == continue_id:
            logger.debug(f'Chunk: generated {step + 1} tokens, stopped at <continue>')
            break

        generated.append(next_id)

        if (step + 1) % 500 == 0:
            logger.info(f'    Token progress: {step + 1}/{max_length}...')

    if len(generated) >= max_length - 1:
        logger.warning(f'Chunk hit max_length={max_length} without <eos>/<continue>')

    return generated


def split_tokens_by_bar(token_ids: List[int], bar_id: int) -> List[List[int]]:
    """Split token sequence into measures at <bar> tokens.

    Each measure includes the trailing <bar> token (except possibly the last
    segment if the sequence doesn't end with <bar>).

    Args:
        token_ids: Flat list of token IDs
        bar_id: Token ID for <bar>

    Returns:
        List of measures, each a list of token IDs.
    """
    measures = []
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
    """Estimate the end time (in seconds) of each measure.

    Uses linear interpolation: assumes token density is roughly proportional
    to audio time within a chunk.

    Returns:
        List of estimated end times (one per measure).
    """
    total_tokens = sum(len(m) for m in measures)
    if total_tokens == 0:
        return []

    end_times = []
    cumulative = 0
    for m in measures:
        cumulative += len(m)
        end_time = chunk_start_sec + (cumulative / total_tokens) * chunk_duration_sec
        end_times.append(end_time)
    return end_times


def find_cut_measure_idx(end_times: List[float], cut_time: float) -> int:
    """Find the measure index where we should cut.

    Returns the index of the first measure whose end time >= cut_time.
    Measures before this index belong to the earlier chunk;
    measures from this index onward belong to the later chunk.
    """
    for i, t in enumerate(end_times):
        if t >= cut_time:
            return i
    return len(end_times)


def merge_chunk_measures(
    all_chunk_measures: List[List[List[int]]],
    chunk_boundaries: List[Tuple[float, float]],
) -> List[int]:
    """Merge measures from overlapping chunks using barline-aligned center cut.

    For each pair of adjacent chunks, the cut point is the temporal midpoint
    of their overlap region. We find the nearest barline to that midpoint
    and split there.

    Args:
        all_chunk_measures: Measures for each chunk (from split_tokens_by_bar).
        chunk_boundaries: (start_sec, end_sec) for each chunk.

    Returns:
        Merged flat list of token IDs.
    """
    n_chunks = len(all_chunk_measures)

    if n_chunks == 1:
        return [tok for m in all_chunk_measures[0] for tok in m]

    result_tokens: List[int] = []

    for i in range(n_chunks):
        measures = all_chunk_measures[i]
        start_sec, end_sec = chunk_boundaries[i]
        chunk_dur = end_sec - start_sec

        if not measures:
            continue

        end_times = estimate_measure_end_times(measures, start_sec, chunk_dur)

        if i == 0:
            # First chunk: keep up to the overlap midpoint with next chunk
            next_start = chunk_boundaries[i + 1][0]
            cut_time = (next_start + end_sec) / 2.0
            cut_idx = find_cut_measure_idx(end_times, cut_time)
            selected = measures[:cut_idx]
            logger.info(
                f'    Chunk {i}: keep measures 0..{cut_idx - 1} '
                f'(cut at {cut_time:.1f}s, {len(selected)} measures)'
            )

        elif i == n_chunks - 1:
            # Last chunk: keep from the overlap midpoint with previous chunk
            prev_end = chunk_boundaries[i - 1][1]
            cut_time = (start_sec + prev_end) / 2.0
            cut_idx = find_cut_measure_idx(end_times, cut_time)
            selected = measures[cut_idx:]
            logger.info(
                f'    Chunk {i}: keep measures {cut_idx}.. '
                f'(cut at {cut_time:.1f}s, {len(selected)} measures)'
            )

        else:
            # Middle chunk: cut on both sides
            prev_end = chunk_boundaries[i - 1][1]
            left_cut_time = (start_sec + prev_end) / 2.0
            next_start = chunk_boundaries[i + 1][0]
            right_cut_time = (next_start + end_sec) / 2.0

            left_idx = find_cut_measure_idx(end_times, left_cut_time)
            right_idx = find_cut_measure_idx(end_times, right_cut_time)

            selected = measures[left_idx:right_idx]
            logger.info(
                f'    Chunk {i}: keep measures {left_idx}..{right_idx - 1} '
                f'(cut at {left_cut_time:.1f}s/{right_cut_time:.1f}s, '
                f'{len(selected)} measures)'
            )

        for m in selected:
            result_tokens.extend(m)

    return result_tokens


# Mel spectrogram frame rate (frames per second)
MEL_FPS = 100


@torch.no_grad()
def generate_kern(
    model: ClefPianoBase,
    mel_path: str,
    tokenizer: KernTokenizer,
    max_length: int = 16384,
    chunk_frames: int = 24000,
    overlap_frames: int = 6000,
    device: str = 'cuda',
) -> str:
    """Generate **kern from mel spectrogram with chunked inference.

    Splits the mel spectrogram into overlapping chunks, generates tokens
    for each chunk independently, then merges using barline-aligned
    center-cut strategy.

    Chunking parameters (defaults):
        chunk_frames = 24000 (4 min @ 100fps)
        overlap_frames = 6000 (1 min overlap)
        stride = chunk_frames - overlap_frames = 18000 (3 min)

    Args:
        model: Trained ClefPianoBase model
        mel_path: Path to mel spectrogram (.pt file)
        tokenizer: KernTokenizer instance
        max_length: Maximum sequence length per chunk
        chunk_frames: Number of mel frames per chunk
        overlap_frames: Overlap between adjacent chunks
        device: Device to run on

    Returns:
        Generated **kern string
    """
    # Load mel
    mel = torch.load(mel_path, map_location=device)  # [1, n_mels, n_frames] or [n_mels, n_frames]
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # [n_mels, n_frames] -> [1, n_mels, n_frames]

    total_frames = mel.shape[2]
    stride = chunk_frames - overlap_frames
    logger.info(
        f'  Mel: {total_frames} frames ({total_frames / MEL_FPS:.1f}s), '
        f'chunk={chunk_frames} ({chunk_frames / MEL_FPS:.0f}s), '
        f'overlap={overlap_frames} ({overlap_frames / MEL_FPS:.0f}s), '
        f'stride={stride} ({stride / MEL_FPS:.0f}s)'
    )

    # Build chunk list with frame boundaries
    chunks: List[Tuple[int, int, torch.Tensor]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        mel_chunk = mel[:, :, start:end].unsqueeze(0).to(device)  # [1, 1, n_mels, chunk_len]
        chunks.append((start, end, mel_chunk))

        if end >= total_frames:
            break
        start += stride

    logger.info(f'  Split into {len(chunks)} chunk(s)')

    bar_id = tokenizer.vocab['<bar>']

    # Generate tokens for each chunk, split into measures
    all_chunk_measures: List[List[List[int]]] = []
    chunk_boundaries: List[Tuple[float, float]] = []

    for i, (start_frame, end_frame, mel_chunk) in enumerate(chunks):
        start_sec = start_frame / MEL_FPS
        end_sec = end_frame / MEL_FPS
        logger.info(
            f'  Chunk {i + 1}/{len(chunks)}: '
            f'[{start_sec:.1f}s - {end_sec:.1f}s] ({end_frame - start_frame} frames)'
        )

        chunk_tokens = generate_kern_chunk(
            model=model,
            mel_chunk=mel_chunk,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )

        # Strip trailing <eos>/<continue> before splitting by barline
        eos_id = tokenizer.vocab['<eos>']
        continue_id = tokenizer.vocab['<continue>']
        while chunk_tokens and chunk_tokens[-1] in (eos_id, continue_id):
            chunk_tokens.pop()

        measures = split_tokens_by_bar(chunk_tokens, bar_id)
        logger.info(f'    Generated {len(chunk_tokens)} tokens, {len(measures)} measures')

        all_chunk_measures.append(measures)
        chunk_boundaries.append((start_sec, end_sec))

    # Merge chunks
    if len(chunks) == 1:
        merged_tokens = [tok for m in all_chunk_measures[0] for tok in m]
        logger.info('  Single chunk, no merging needed')
    else:
        logger.info('  Merging chunks (barline-aligned center cut):')
        merged_tokens = merge_chunk_measures(all_chunk_measures, chunk_boundaries)

    logger.info(f'  Total merged tokens: {len(merged_tokens)}')

    # Reconstruct kern from token IDs (handles multi-spine, barlines, split/merge)
    pred_kern = reconstruct_kern_from_token_ids(merged_tokens, tokenizer)

    return pred_kern


def kern_to_midi(kern_content: str, output_midi_path: str) -> bool:
    """Convert **kern string to MIDI file.

    Args:
        kern_content: **kern string content
        output_midi_path: Path to save MIDI file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Parse kern with music21 (via converter21)
        score = music21.converter.parse(kern_content, format='humdrum')

        # Write to MIDI
        score.write('midi', fp=output_midi_path)

        return True

    except Exception as e:
        logger.error(f'Failed to convert kern to MIDI: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='Inference for clef-piano-base')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/clef_piano_base.yaml',
                        help='Path to model config')
    parser.add_argument('--manifest', type=str,
                        default='data/experiments/clef_piano_base/test_manifest.json',
                        help='Path to test manifest')
    parser.add_argument('--manifest-dir', type=str,
                        default='data/experiments/clef_piano_base',
                        help='Base directory for manifest paths')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for kern predictions')
    parser.add_argument('--output-midi-dir', type=str, default=None,
                        help='Output directory for MIDI files (optional, defaults to {output-dir}_midi)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--max-length', type=int, default=16384,
                        help='Maximum generation length per chunk')
    parser.add_argument('--chunk-frames', type=int, default=24000,
                        help='Mel frames per chunk (default: 24000 = 4min @ 100fps)')
    parser.add_argument('--overlap-frames', type=int, default=6000,
                        help='Overlap frames between chunks (default: 6000 = 1min, stride = 3min)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)

    # MIDI output directory
    if args.output_midi_dir:
        output_midi_dir = Path(args.output_midi_dir)
    else:
        output_midi_dir = Path(str(output_dir) + '_midi')
    output_midi_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.config, args.device)
    tokenizer = KernTokenizer()

    # Load manifest
    manifest = load_manifest(args.manifest)

    # Inference
    logger.info(f'Starting inference on {len(manifest)} samples')

    success_count = 0
    midi_success_count = 0

    for idx, item in enumerate(tqdm(manifest, desc='Generating'), 1):
        sample_id = item['id']
        mel_path = manifest_dir / item['mel_path']

        logger.info(f'[{idx}/{len(manifest)}] Processing: {sample_id}')

        # Generate kern
        pred_kern = generate_kern(
            model=model,
            mel_path=str(mel_path),
            tokenizer=tokenizer,
            max_length=args.max_length,
            chunk_frames=args.chunk_frames,
            overlap_frames=args.overlap_frames,
            device=args.device,
        )

        # Save kern prediction
        kern_output_path = output_dir / f'{sample_id}.krn'
        with open(kern_output_path, 'w', encoding='utf-8') as f:
            f.write(pred_kern)
        success_count += 1

        # Convert to MIDI
        midi_output_path = output_midi_dir / f'{sample_id}.mid'
        if kern_to_midi(pred_kern, str(midi_output_path)):
            midi_success_count += 1

    logger.info(f'Inference complete!')
    logger.info(f'  Kern predictions: {success_count}/{len(manifest)} → {output_dir}')
    logger.info(f'  MIDI conversions: {midi_success_count}/{len(manifest)} → {output_midi_dir}')


if __name__ == '__main__':
    main()
