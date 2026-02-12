#!/usr/bin/env python3
"""Analyze Cross-Attention level preferences by token type (pitch vs duration)."""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import numpy as np

from src.clef.piano.model import ClefPianoBase
from src.clef.piano.config import ClefPianoConfig
from src.clef.piano.tokenizer import KernTokenizer
from src.clef.data import ManifestDataset, ChunkedDataset


def is_pitch_token(token: str) -> bool:
    """Check if token is a pitch (note) token."""
    if not token or token.startswith('<'):
        return False
    # Pitch tokens: duration + pitch, e.g., "4c", "8f#", "16g-"
    # Check if contains a letter (pitch class)
    return any(c in token.lower() for c in 'abcdefg')


def is_duration_token(token: str) -> bool:
    """Check if token is a pure duration token (rest or just number)."""
    if not token or token.startswith('<'):
        return False
    # Rests: "4r", "8r", etc.
    if 'r' in token.lower():
        return True
    return False


def analyze_checkpoint(ckpt_path: str, manifest_dir: str, n_samples: int = 10):
    """Analyze CA level preferences for a checkpoint."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Create model
    model = ClefPianoBase(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get tokenizer
    tokenizer = KernTokenizer()

    # Load validation data
    manifest_path = Path(manifest_dir) / "valid_manifest.json"
    valid_dataset = ManifestDataset(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        # mel_path in manifest already includes "mel/" prefix
        mel_dir=None,
        kern_dir=None,
        augmentation_metadata_path=Path(manifest_dir) / "augmentation_metadata.json",
    )

    chunked = ChunkedDataset(
        valid_dataset,
        tokenizer=tokenizer,
        chunk_frames=24000,
        overlap_frames=12000,
    )

    # Storage for level weights by token type
    level_weights_by_type = {
        'pitch': defaultdict(list),  # decoder_layer -> list of [L] weights
        'duration': defaultdict(list),
        'struct': defaultdict(list),  # <bar>, <nl>, etc.
        'all': defaultdict(list),
    }

    # Hook to capture level weights
    level_weights_captured = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Access the level_weights computation
            # We need to re-compute it here since it's not returned
            query = input[0]  # First input is query
            B, N_q, _ = query.shape
            lw = module.level_weights(query)
            lw = lw.view(B, N_q, module.n_heads, module.n_levels)
            lw = F.softmax(lw, dim=-1)  # [B, N_q, H, L]
            # Average over heads
            lw = lw.mean(dim=2)  # [B, N_q, L]
            level_weights_captured[layer_idx] = lw.detach().cpu()
        return hook

    # Register hooks on FluxCA modules in decoder
    hooks = []
    for i, layer in enumerate(model.decoder.layers):
        if hasattr(layer, 'cross_attn'):
            h = layer.cross_attn.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Analyze samples
    print(f"\nAnalyzing {n_samples} samples...")

    for sample_idx in range(min(n_samples, len(chunked))):
        mel, tokens, meta = chunked[sample_idx]
        mel = mel.unsqueeze(0).to(device)  # [1, C, T]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Teacher forcing: input is tokens[:-1], target is tokens[1:]
        input_tokens = tokens_tensor[:, :-1]

        with torch.no_grad():
            level_weights_captured.clear()
            _ = model(mel, input_tokens)

        # Classify each token and store level weights
        for pos, tok_id in enumerate(tokens[:-1]):
            tok_str = tokenizer.decode([tok_id])

            for layer_idx, lw in level_weights_captured.items():
                if pos < lw.shape[1]:
                    weights = lw[0, pos].numpy()  # [L]
                    level_weights_by_type['all'][layer_idx].append(weights)

                    if is_pitch_token(tok_str):
                        level_weights_by_type['pitch'][layer_idx].append(weights)
                    elif is_duration_token(tok_str):
                        level_weights_by_type['duration'][layer_idx].append(weights)
                    elif tok_str.startswith('<'):
                        level_weights_by_type['struct'][layer_idx].append(weights)

        if (sample_idx + 1) % 5 == 0:
            print(f"  Processed {sample_idx + 1}/{n_samples} samples")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    level_names = ['Octopus', 'Flow', 'S0', 'S1', 'S2', 'S3']

    print("\n" + "=" * 80)
    print("CA Level Preferences by Token Type")
    print("=" * 80)

    for tok_type in ['all', 'pitch', 'duration', 'struct']:
        data = level_weights_by_type[tok_type]
        if not data:
            continue

        print(f"\n### {tok_type.upper()} tokens ###")
        print(f"{'Layer':<8}", end='')
        for name in level_names:
            print(f"{name:>8}", end='')
        print(f"{'Count':>10}")
        print("-" * 70)

        for layer_idx in sorted(data.keys()):
            weights_list = data[layer_idx]
            if weights_list:
                avg = np.mean(weights_list, axis=0)
                count = len(weights_list)
                print(f"L{layer_idx:<7}", end='')
                for w in avg:
                    print(f"{w*100:>7.1f}%", end='')
                print(f"{count:>10}")

    # Pitch vs Duration comparison
    print("\n" + "=" * 80)
    print("Pitch vs Duration Preference Difference (Pitch - Duration)")
    print("=" * 80)

    if level_weights_by_type['pitch'] and level_weights_by_type['duration']:
        print(f"{'Layer':<8}", end='')
        for name in level_names:
            print(f"{name:>8}", end='')
        print()
        print("-" * 60)

        for layer_idx in sorted(level_weights_by_type['pitch'].keys()):
            pitch_weights = level_weights_by_type['pitch'].get(layer_idx, [])
            dur_weights = level_weights_by_type['duration'].get(layer_idx, [])

            if pitch_weights and dur_weights:
                pitch_avg = np.mean(pitch_weights, axis=0)
                dur_avg = np.mean(dur_weights, axis=0)
                diff = pitch_avg - dur_avg

                print(f"L{layer_idx:<7}", end='')
                for d in diff:
                    sign = '+' if d >= 0 else ''
                    print(f"{sign}{d*100:>6.1f}%", end='')
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest-dir', type=str,
                        default='data/experiments/clef_piano_base')
    parser.add_argument('--n-samples', type=int, default=10)
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint, args.manifest_dir, args.n_samples)
