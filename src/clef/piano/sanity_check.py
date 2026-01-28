#!/usr/bin/env python
"""
Sanity Check: Overfit One Batch
================================

Verify the training pipeline works by overfitting on a single batch.
If the model can't overfit one batch, something is fundamentally broken.

Usage:
    # GPU 0
    python scripts/sanity_check.py --gpu 0

    # GPU 4
    python scripts/sanity_check.py --gpu 4

    # With real data
    python scripts/sanity_check.py --gpu 0 --use-real-data

Expected:
    - Loss should drop from ~6.0 to < 0.1 in 100 steps
    - If loss doesn't decrease, check gradient flow
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path (src/clef/piano -> 3 levels up to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clef.piano import ClefPianoBase, ClefPianoConfig
from src.clef.piano.tokenizer import KernTokenizer

DATA_ROOT = PROJECT_ROOT / 'data' / 'experiments' / 'clef_piano_base'


def create_real_batch(
    batch_size: int,
    tokenizer: KernTokenizer,
    device: torch.device,
    max_frames: int = 6000,  # 1 min @ 100 fps
    max_seq_len: int = 512,
):
    """Load real data from manifest for overfitting test."""
    manifest_path = DATA_ROOT / 'train_manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find short samples for faster testing
    short_samples = [m for m in manifest if m['n_frames'] <= max_frames]
    if len(short_samples) < batch_size:
        short_samples = manifest[:batch_size]  # Fallback to first samples

    # Load batch_size samples
    mels = []
    input_ids_list = []
    labels_list = []

    for i in range(batch_size):
        sample = short_samples[i % len(short_samples)]

        # Load mel
        mel_path = DATA_ROOT / sample['mel_path']
        mel = torch.load(mel_path, weights_only=True)  # [1, 128, T]

        # Truncate if too long
        if mel.shape[-1] > max_frames:
            mel = mel[..., :max_frames]

        mels.append(mel)

        # Load and tokenize kern
        kern_path = DATA_ROOT / sample['kern_gt_path']
        with open(kern_path) as f:
            kern_text = f.read()

        tokens = tokenizer.encode(kern_text)
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]

        input_ids_list.append(tokens[:-1])  # Without <eos>
        labels_list.append(tokens[1:])      # Without <sos>

    # Pad mels to same length
    mel_lengths = [m.shape[-1] for m in mels]
    max_mel_len = max(mel_lengths)
    max_mel_len = ((max_mel_len + 31) // 32) * 32  # Pad to multiple of 32

    padded_mels = []
    for mel in mels:
        pad_len = max_mel_len - mel.shape[-1]
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len))
        padded_mels.append(mel)

    # Pad token sequences
    seq_lengths = [len(ids) for ids in input_ids_list]
    max_seq = max(seq_lengths)

    PAD_ID = 0  # <pad> token id
    padded_input_ids = []
    padded_labels = []
    for inp, lab in zip(input_ids_list, labels_list):
        pad_len = max_seq - len(inp)
        padded_input_ids.append(inp + [PAD_ID] * pad_len)
        padded_labels.append(lab + [PAD_ID] * pad_len)

    mel_valid_ratios = torch.tensor([l / max_mel_len for l in mel_lengths], device=device)

    return {
        'mel': torch.stack(padded_mels).to(device),
        'input_ids': torch.tensor(padded_input_ids, device=device),
        'labels': torch.tensor(padded_labels, device=device),
        'mel_valid_ratios': mel_valid_ratios,
    }


def create_synthetic_batch(batch_size: int, duration_sec: int, vocab_size: int, device: torch.device):
    """Create synthetic batch for testing."""
    T = int(duration_sec * 100)  # 100 fps
    T = ((T + 31) // 32) * 32    # Pad to multiple of 32

    mel = torch.randn(batch_size, 1, 128, T, device=device)

    # Create plausible token sequences
    seq_len = 256
    input_ids = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    input_ids[:, 0] = 1  # <sos>

    labels = torch.randint(3, vocab_size, (batch_size, seq_len), device=device)
    labels[:, -1] = 2    # <eos>

    mel_valid_ratios = torch.ones(batch_size, device=device)

    return {
        'mel': mel,
        'input_ids': input_ids,
        'labels': labels,
        'mel_valid_ratios': mel_valid_ratios,
    }


def main():
    parser = argparse.ArgumentParser(description='Sanity check: overfit one batch')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--duration', type=int, default=60, help='Audio duration in seconds')
    parser.add_argument('--steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--use-real-data', action='store_true', help='Use real data from manifest')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='Clef-piano-base', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='bloggerwang1217-national-taiwan-university', help='Wandb entity/team')
    args = parser.parse_args()

    # Setup wandb
    use_wandb = args.wandb and HAS_WANDB
    if args.wandb and not HAS_WANDB:
        print('Warning: wandb not installed, skipping logging')
    if use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f'sanity-check-gpu{args.gpu}',
            config={
                'batch_size': args.batch_size,
                'duration_sec': args.duration,
                'steps': args.steps,
                'lr': args.lr,
                'grad_clip': args.grad_clip,
            },
            tags=['sanity-check', 'overfit'],
        )
        print(f'Wandb: {wandb.run.url}')

    # Setup device
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    print(f'Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}')
    print(f'Available memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB')
    print()

    # Create model
    config = ClefPianoConfig()

    # Update vocab size from tokenizer
    tokenizer = KernTokenizer()
    config.vocab_size = tokenizer.vocab_size
    print(f'Vocab size: {config.vocab_size}')

    model = ClefPianoBase(config).to(device)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')
    print()

    # Create batch
    print(f'Creating {"real" if args.use_real_data else "synthetic"} batch...')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Duration: {args.duration} sec')

    if args.use_real_data:
        batch = create_real_batch(
            args.batch_size, tokenizer, device,
            max_frames=args.duration * 100,  # 100 fps
            max_seq_len=512,
        )
    else:
        batch = create_synthetic_batch(
            args.batch_size, args.duration, config.vocab_size, device
        )

    print(f'  Mel shape: {batch["mel"].shape}')
    print(f'  Token shape: {batch["input_ids"].shape}')
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f'Starting overfit test ({args.steps} steps)...')
    print('=' * 50)

    initial_loss = None
    start_time = time.time()

    for step in range(args.steps):
        step_start = time.time()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(
                mel=batch['mel'],
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                mel_valid_ratios=batch['mel_valid_ratios'],
            )

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        step_time = time.time() - step_start

        if initial_loss is None:
            initial_loss = loss.item()

        # Logging
        if step % 10 == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f'Step {step:3d}: loss={loss.item():.4f}, grad_norm={grad_norm:.2f}, mem={mem:.2f}GB, {step_time:.2f}s/step')

        if use_wandb:
            wandb.log({
                'train/loss': loss.item(),
                'train/grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                'train/step_time': step_time,
                'system/gpu_memory_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
            }, step=step, commit=True)  # commit=True for real-time sync

    total_time = time.time() - start_time
    print(f'Total time: {total_time:.1f}s ({total_time/args.steps:.2f}s/step)')

    print('=' * 50)

    # Check results
    final_loss = loss.item()
    print()
    print('Results:')
    print(f'  Initial loss: {initial_loss:.4f}')
    print(f'  Final loss:   {final_loss:.4f}')
    print(f'  Reduction:    {initial_loss - final_loss:.4f} ({(1 - final_loss/initial_loss)*100:.1f}%)')
    print()

    if final_loss < 0.1:
        status = 'PASSED'
        print('PASSED: Model can overfit one batch!')
    elif final_loss < initial_loss * 0.5:
        status = 'PARTIAL'
        print('PARTIAL: Loss decreased but not fully overfit. May need more steps.')
    else:
        status = 'FAILED'
        print('FAILED: Loss did not decrease significantly. Check gradient flow!')

        # Debug: check if gradients are flowing
        print()
        print('Debugging gradient flow...')
        has_grad = 0
        no_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad += 1
                else:
                    no_grad += 1
                    if no_grad <= 5:
                        print(f'  No gradient: {name}')
        print(f'  Params with gradient: {has_grad}')
        print(f'  Params without gradient: {no_grad}')

    # Log final results to wandb
    if use_wandb:
        wandb.log({
            'result/initial_loss': initial_loss,
            'result/final_loss': final_loss,
            'result/loss_reduction': initial_loss - final_loss,
            'result/total_time_sec': total_time,
            'result/status': status,
        })
        run_url = wandb.run.url
        wandb.finish()
        print(f'\nWandb run finished: {run_url}')


if __name__ == '__main__':
    main()
