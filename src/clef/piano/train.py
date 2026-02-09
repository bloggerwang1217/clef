"""
Training Script for clef-piano-base
===================================

Supports:
- Single GPU training
- DDP (Distributed Data Parallel) multi-GPU training
- Mixed precision (BF16/FP16)
- Gradient accumulation
- Wandb logging
- Checkpoint saving/loading
- Step-based validation

Usage:
    # Single GPU
    python -m src.clef.piano.train --config configs/clef_piano_base.yaml

    # Multi-GPU DDP (e.g., GPU 1 and 4)
    CUDA_VISIBLE_DEVICES=1,4 torchrun --nproc_per_node=2 -m src.clef.piano.train --config configs/clef_piano_base.yaml

    # Sanity check (overfit one batch)
    python -m src.clef.piano.train --config configs/clef_piano_base.yaml --sanity-check
"""

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

# Debug: Print CUDA visibility info BEFORE any torch import
_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f"[DEBUG] CUDA_VISIBLE_DEVICES = {_cuda_visible}", flush=True)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

# Debug: Print CUDA device info after torch import
print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)
print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[DEBUG] cuda:{i} = {torch.cuda.get_device_name(i)}", flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.clef.piano.config import ClefPianoConfig
from src.clef.piano.model import ClefPianoBase
from src.clef import ChunkedDataset, ManifestDataset, ClefCollator, BucketSampler, DistributedBucketSampler
from src.clef.piano.tokenizer import KernTokenizer
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0, level: int = logging.INFO):
    """Setup logging (only rank 0 logs to console)."""
    if rank == 0:
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """Trainer for clef-piano-base model."""

    def __init__(
        self,
        config: ClefPianoConfig,
        model: ClefPianoBase,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        use_wandb: bool = False,
        gradient_accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
        validate_every_n_steps: int = 500,
        early_stopping_patience: int = 0,  # 0 = disabled
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip = gradient_clip
        self.validate_every_n_steps = validate_every_n_steps
        self.early_stopping_patience = early_stopping_patience

        # Model
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank])

        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Optimizer setup:
        # - Main optimizer (OneCycleLR): offset params (0.1x) + everything else (1x)
        base_lr = config.learning_rate if hasattr(config, 'learning_rate') else 1e-4
        offset_lr = base_lr * 0.1  # 0.1x for offset-related params
        weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.01

        offset_param_names = [
            'time_prior', 'reference_refine',                  # Decoder reference points
            'time_offset_proj', 'freq_offset_proj',           # Attention offsets
        ]
        offset_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(offset_name in name for offset_name in offset_param_names):
                offset_params.append(param)
                if self.rank == 0:
                    logger.debug(f'Offset param (lr={offset_lr:.2e}): {name}')
            else:
                other_params.append(param)

        if self.rank == 0:
            logger.info(f'Optimizer: {len(offset_params)} offset params (lr={offset_lr:.2e}), '
                       f'{len(other_params)} other params (lr={base_lr:.2e})')

        self.optimizer = torch.optim.AdamW([
            {'params': offset_params, 'lr': offset_lr},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=weight_decay)

        # Learning rate scheduler (main optimizer only)
        total_steps = len(train_loader) * (config.max_epochs if hasattr(config, 'max_epochs') else 50)
        total_steps = total_steps // gradient_accumulation_steps
        warmup_steps = config.warmup_steps if hasattr(config, 'warmup_steps') else 1000

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[offset_lr, base_lr],  # 2 groups: offset, other
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.best_valid_loss = float('inf')
        self.epochs_without_improvement = 0

        # Wandb
        self.use_wandb = use_wandb and rank == 0

        # Build pitch token mask for loss breakdown (pitch vs structure vs duration)
        self._build_token_type_masks()

    def _build_token_type_masks(self):
        """Build boolean masks over vocab to classify token types.

        Creates four 1-D boolean tensors of shape [vocab_size]:
        - pitch_mask: pitch tokens (e.g. C, D#, ee, GG, r)
        - duration_mask: duration tokens (e.g. 4, 8., 16, q)
        - struct_mask: structural tokens (<coc>, <bar>, <nl>, ...)
        - schema_mask: schema tokens (<ts:N>, <key:X>) — isolated for fair comparison

        Used for per-type loss breakdown logged to wandb (monitoring only,
        does not affect training).
        """
        tokenizer = KernTokenizer()
        V = self.config.vocab_size

        struct_names = {
            '<pad>', '<sos>', '<eos>', '<coc>', '<bar>',
            '<b>', '<continue>', '<nl>', '<split>', '<merge>', '<*>',
            '(', ')',    # slur start/end
            '{', '}',    # phrase start/end
            'L', 'J',    # beam start/end
            ';',         # fermata
        }
        duration_names = {
            '0', '00',  # breve, longa
            '1', '2', '4', '8', '16', '32', '64', '128',
            '0.', '00.', '1.', '2.', '4.', '8.', '16.', '32.', '64.',
            '3', '6', '12', '24', '48', '96',
            '20', '40', '112', '176',  # tuplet durations
            '.', 'q', 'Q', 'P',
            '[', ']', '_',  # tie start/end/continue (onset vs sustain)
            '\t', '\n',  # whitespace (should not appear in labels)
        }
        # Schema tokens: time signature numerators + key signatures
        # Isolated so they don't inflate/deflate struct or pitch loss
        schema_names = {
            # Time signature numerators
            '<2>', '<3>', '<4>', '<5>', '<6>', '<7>',
            '<8>', '<9>', '<10>', '<12>', '<17>',
            # Key signatures
            '<key:0>',
            '<key:1#>', '<key:2#>', '<key:3#>', '<key:4#>',
            '<key:5#>', '<key:6#>', '<key:7#>',
            '<key:1b>', '<key:2b>', '<key:3b>', '<key:4b>',
            '<key:5b>', '<key:6b>', '<key:7b>',
        }
        # r (rest) stays in pitch: it's the "no pitch" decision, needs audio

        struct_ids = set()
        duration_ids = set()
        pitch_ids = set()
        schema_ids = set()

        for tok_str, tok_id in tokenizer.vocab.items():
            if tok_str in schema_names:
                schema_ids.add(tok_id)
            elif tok_str in struct_names:
                struct_ids.add(tok_id)
            elif tok_str in duration_names:
                duration_ids.add(tok_id)

        # Everything else is a pitch token (note names, r, accidentals)
        classified = struct_ids | duration_ids | schema_ids
        for tok_id in range(V):
            if tok_id not in classified:
                pitch_ids.add(tok_id)

        self._struct_mask = torch.zeros(V, dtype=torch.bool)
        self._duration_mask = torch.zeros(V, dtype=torch.bool)
        self._pitch_mask = torch.zeros(V, dtype=torch.bool)
        self._schema_mask = torch.zeros(V, dtype=torch.bool)
        for i in struct_ids:
            self._struct_mask[i] = True
        for i in duration_ids:
            self._duration_mask[i] = True
        for i in pitch_ids:
            self._pitch_mask[i] = True
        for i in schema_ids:
            self._schema_mask[i] = True

        if self.rank == 0:
            logger.info(
                f'Token type masks: {self._pitch_mask.sum()} pitch, '
                f'{self._duration_mask.sum()} duration, '
                f'{self._struct_mask.sum()} struct, '
                f'{self._schema_mask.sum()} schema'
            )

    @torch.no_grad()
    def _compute_loss_breakdown(
        self,
        logits: torch.Tensor,  # [B, S, V]
        labels: torch.Tensor,  # [B, S]
    ) -> Dict[str, float]:
        """Compute per-token-type loss (monitoring only, detached)."""
        ce = nn.CrossEntropyLoss(reduction='none')
        per_token = ce(logits.view(-1, logits.size(-1)), labels.view(-1))  # [B*S]
        flat_labels = labels.view(-1)

        # Map each label to its type using pre-built masks
        pitch_mask = self._pitch_mask.to(flat_labels.device)
        duration_mask = self._duration_mask.to(flat_labels.device)
        struct_mask = self._struct_mask.to(flat_labels.device)
        schema_mask = self._schema_mask.to(flat_labels.device)

        non_pad = flat_labels != 0
        is_pitch = pitch_mask[flat_labels] & non_pad
        is_duration = duration_mask[flat_labels] & non_pad
        is_struct = struct_mask[flat_labels] & non_pad
        is_schema = schema_mask[flat_labels] & non_pad

        result = {}
        if is_pitch.any():
            result['loss_pitch'] = per_token[is_pitch].mean().item()
        if is_duration.any():
            result['loss_duration'] = per_token[is_duration].mean().item()
        if is_struct.any():
            result['loss_struct'] = per_token[is_struct].mean().item()
        if is_schema.any():
            result['loss_schema'] = per_token[is_schema].mean().item()
        return result

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch}',
            disable=self.rank != 0,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Skip batches where all samples were filtered out
            if batch is None:
                continue

            # Move to device
            mel = batch['mel'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            mel_valid_ratios = batch['mel_valid_ratios'].to(self.device)

            # Skip DDP all-reduce on non-update micro-batches to save
            # communication overhead during gradient accumulation.
            is_update_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            maybe_no_sync = (
                self.model.no_sync()
                if isinstance(self.model, DDP) and not is_update_step
                else nullcontext()
            )

            with maybe_no_sync:
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    logits, loss = self.model(
                        mel=mel,
                        input_ids=input_ids,
                        labels=labels,
                        mel_valid_ratios=mel_valid_ratios,
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            accumulated_loss += loss.item() * self.gradient_accumulation_steps

            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Per-group clipping: offset/base clipped independently
                # so reference_refine's huge gradient doesn't starve base params
                offset_params = [p for p in self.optimizer.param_groups[0]['params']]
                base_params = [p for p in self.optimizer.param_groups[1]['params']]
                offset_grad_norm = torch.nn.utils.clip_grad_norm_(offset_params, self.gradient_clip)
                grad_norm = torch.nn.utils.clip_grad_norm_(base_params, self.gradient_clip)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update metrics (average over accumulation steps)
                avg_accum_loss = accumulated_loss / self.gradient_accumulation_steps
                total_loss += avg_accum_loss
                num_batches += 1
                self.global_step += 1

                # Log to wandb
                if self.use_wandb:
                    model_unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
                    layers = model_unwrapped.decoder.layers

                    log_dict = {
                        'train/loss': avg_accum_loss,
                        'train/grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        'train/offset_grad_norm': offset_grad_norm.item() if hasattr(offset_grad_norm, 'item') else offset_grad_norm,
                        'train/lr_base': self.scheduler.get_last_lr()[1],
                        'train/lr_offset': self.scheduler.get_last_lr()[0],
                        'train/epoch': self.epoch,
                        'system/gpu_memory_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    }

                    # Predictive coding gate: log gain and pred_loss
                    gain_mean = sum(
                        F.softplus(l.ca_gain.bias.data).item() for l in layers
                    ) / len(layers)
                    log_dict['train/pc_gain_mean'] = gain_mean
                    if model_unwrapped._last_pred_loss is not None:
                        log_dict['train/pred_loss'] = model_unwrapped._last_pred_loss.item()

                    # Per-token-type loss breakdown (last micro-batch only)
                    breakdown = self._compute_loss_breakdown(logits.detach(), labels)
                    for k, v in breakdown.items():
                        log_dict[f'train/{k}'] = v

                    wandb.log(log_dict, step=self.global_step)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_accum_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[1]:.2e}',
                    'grad': f'{grad_norm:.2f}' if hasattr(grad_norm, '__float__') else 'N/A',
                })

                accumulated_loss = 0.0

                # Step-based validation
                if self.validate_every_n_steps > 0 and self.global_step % self.validate_every_n_steps == 0:
                    valid_metrics = self.validate()
                    if valid_metrics and self.rank == 0:
                        logger.info(f'Step {self.global_step}: valid_loss={valid_metrics["valid_loss"]:.4f}')
                        if self.use_wandb:
                            valid_log = {'valid/loss': valid_metrics['valid_loss']}
                            for k in ('loss_pitch', 'loss_duration', 'loss_struct', 'loss_schema'):
                                if k in valid_metrics:
                                    valid_log[f'valid/{k}'] = valid_metrics[k]
                            wandb.log(valid_log, step=self.global_step)
                    self.model.train()

        avg_loss = total_loss / max(num_batches, 1)

        return {'train_loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.valid_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        # Accumulators for per-type loss breakdown
        type_loss_sums = {'loss_pitch': 0.0, 'loss_duration': 0.0, 'loss_struct': 0.0, 'loss_schema': 0.0}
        type_loss_counts = {'loss_pitch': 0, 'loss_duration': 0, 'loss_struct': 0, 'loss_schema': 0}

        for batch_idx, batch in enumerate(self.valid_loader):
            if batch is None:
                if self.rank == 0:
                    logger.warning(f'Validation: skipping batch {batch_idx} (all samples filtered)')
                continue
            mel = batch['mel'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            mel_valid_ratios = batch['mel_valid_ratios'].to(self.device)

            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                logits, loss = self.model(
                    mel=mel,
                    input_ids=input_ids,
                    labels=labels,
                    mel_valid_ratios=mel_valid_ratios,
                )

            total_loss += loss.item()
            num_batches += 1

            # Accumulate per-type losses
            breakdown = self._compute_loss_breakdown(logits, labels)
            for k, v in breakdown.items():
                type_loss_sums[k] += v
                type_loss_counts[k] += 1

        avg_loss = total_loss / max(num_batches, 1)

        result = {'valid_loss': avg_loss}
        for k in type_loss_sums:
            if type_loss_counts[k] > 0:
                result[k] = type_loss_sums[k] / type_loss_counts[k]

        return result

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save checkpoint."""
        if self.rank != 0:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_valid_loss': self.best_valid_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f'Saved checkpoint to {path}')

        if is_best:
            best_path = path.parent / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f'Saved best model to {best_path}')

    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        if not path.exists():
            logger.warning(f'Checkpoint not found: {path}')
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_valid_loss = checkpoint['best_valid_loss']
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f'Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})')

    def train(self, max_epochs: int, checkpoint_dir: Path):
        """Full training loop."""
        model_for_info = self.model.module if isinstance(self.model, DDP) else self.model
        logger.info(f'Starting training for {max_epochs} epochs')
        logger.info(f'Model params: {model_for_info.get_num_params():,} trainable')
        logger.info(f'Gradient accumulation steps: {self.gradient_accumulation_steps}')
        logger.info(f'Effective batch size: {self.train_loader.batch_size * self.world_size * self.gradient_accumulation_steps}')
        if self.early_stopping_patience > 0:
            logger.info(f'Early stopping patience: {self.early_stopping_patience} epochs')

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch()

            # Log OOV skipped chunks (if any)
            if self.rank == 0 and hasattr(self.train_loader.dataset, 'oov_skipped_chunks'):
                skipped = self.train_loader.dataset.oov_skipped_chunks
                if skipped > 0:
                    total = len(self.train_loader.dataset)
                    logger.info(
                        f'Epoch {epoch}: skipped {skipped}/{total} chunks '
                        f'due to OOV tokens ({skipped/total*100:.1f}%)'
                    )
                    self.train_loader.dataset.oov_skipped_chunks = 0

            # Validate at end of epoch
            valid_metrics = self.validate()

            # Log
            if self.rank == 0:
                logger.info(
                    f'Epoch {epoch}: '
                    f'train_loss={train_metrics["train_loss"]:.4f}, '
                    f'valid_loss={valid_metrics.get("valid_loss", "N/A")}'
                )

                if self.use_wandb:
                    epoch_log = {
                        'epoch/train_loss': train_metrics['train_loss'],
                    }
                    if 'valid_loss' in valid_metrics:
                        epoch_log['epoch/valid_loss'] = valid_metrics['valid_loss']
                    for k in ('loss_pitch', 'loss_duration', 'loss_struct', 'loss_schema'):
                        if k in valid_metrics:
                            epoch_log[f'epoch/valid_{k}'] = valid_metrics[k]
                    wandb.log(epoch_log, step=self.global_step)

            # Save checkpoint and check early stopping
            is_best = False
            if 'valid_loss' in valid_metrics:
                if valid_metrics['valid_loss'] < self.best_valid_loss:
                    self.best_valid_loss = valid_metrics['valid_loss']
                    is_best = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(
                    checkpoint_dir / f'epoch_{epoch:03d}.pt',
                    is_best=is_best,
                )

            # Early stopping check
            if self.early_stopping_patience > 0 and self.epochs_without_improvement >= self.early_stopping_patience:
                if self.rank == 0:
                    logger.info(
                        f'Early stopping triggered: no improvement for {self.epochs_without_improvement} epochs. '
                        f'Best valid_loss: {self.best_valid_loss:.4f}'
                    )
                break

        # Save final checkpoint
        self.save_checkpoint(checkpoint_dir / 'last.pt')

        if self.use_wandb:
            wandb.finish()


def sanity_check(model: ClefPianoBase, train_loader: DataLoader, device: torch.device):
    """Overfit one batch to verify pipeline works."""
    logger.info('Running sanity check (overfit one batch)...')

    model.train()
    model = model.to(device)

    # Get one batch
    batch = next(iter(train_loader))
    mel = batch['mel'].to(device)
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    mel_valid_ratios = batch['mel_valid_ratios'].to(device)

    logger.info(f'Batch shapes: mel={mel.shape}, input_ids={input_ids.shape}, labels={labels.shape}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(100):
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(
                mel=mel,
                input_ids=input_ids,
                labels=labels,
                mel_valid_ratios=mel_valid_ratios,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            logger.info(f'Step {step}: loss={loss.item():.4f}')

    logger.info(f'Final loss: {loss.item():.4f}')
    if loss.item() < 0.1:
        logger.info('Sanity check PASSED: model can overfit one batch')
    else:
        logger.warning('Sanity check: loss still high, may need more steps')


def main():
    parser = argparse.ArgumentParser(description='Train clef-piano-base')
    parser.add_argument('--config', type=str, default='configs/clef_piano_base.yaml',
                        help='Path to config file')
    parser.add_argument('--manifest-dir', type=str, default='data/experiments/clef_piano_base',
                        help='Directory containing train/valid/test manifests')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/clef_piano_base',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size per GPU')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--validate-every-n-steps', type=int, default=500,
                        help='Validate every N steps (0 to disable step-based validation)')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Stop training if valid_loss does not improve for N epochs (0 to disable)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check (overfit one batch)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='clef-piano-base',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='bloggerwang1217-national-taiwan-university',
                        help='Wandb entity/team')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    setup_logging(rank, logging.DEBUG if args.debug else logging.INFO)

    logger.info(f'Rank {rank}/{world_size}, local_rank={local_rank}')

    try:
        # Load config
        config = ClefPianoConfig.from_yaml(args.config)

        # Set seed for reproducibility
        training_seed = getattr(config, 'training_seed', 1234)
        set_seed(training_seed)
        logger.info(f'Set training seed: {training_seed}')

        # Update vocab size from tokenizer
        tokenizer = KernTokenizer()
        config.vocab_size = tokenizer.vocab_size
        logger.info(f'Vocab size: {config.vocab_size}')

        # Create model
        model = ClefPianoBase(config)
        logger.info(f'Model created: {model.get_num_params():,} trainable params')

        # Create datasets
        manifest_dir = Path(args.manifest_dir)
        train_manifest = manifest_dir / 'train_manifest.json'
        valid_manifest = manifest_dir / 'valid_manifest.json'

        if not train_manifest.exists():
            logger.error(f'Train manifest not found: {train_manifest}')
            logger.info('Please run the data preparation script first.')
            return

        aug_metadata_path = manifest_dir / 'augmentation_metadata.json'

        train_dataset = ManifestDataset(
            train_manifest,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            augmentation_metadata_path=aug_metadata_path,
        )

        # Optionally chunk long pieces
        if hasattr(config, 'chunk_frames') and config.chunk_frames > 0:
            train_dataset = ChunkedDataset(
                train_dataset,
                tokenizer=tokenizer,
                chunk_frames=config.chunk_frames,
                overlap_frames=config.overlap_frames,
                min_chunk_ratio=config.min_chunk_ratio,
            )

        valid_dataset = None
        if valid_manifest.exists():
            valid_dataset = ManifestDataset(
                valid_manifest,
                tokenizer=tokenizer,
                max_seq_len=config.max_seq_len,
                augmentation_metadata_path=aug_metadata_path,
            )
            # Chunk validation set too: ensures mel-kern alignment and
            # prevents OOM on long pieces (some valid pieces > 10 min).
            # No overlap needed for validation (just computing loss).
            if hasattr(config, 'chunk_frames') and config.chunk_frames > 0:
                valid_dataset = ChunkedDataset(
                    valid_dataset,
                    tokenizer=tokenizer,
                    chunk_frames=config.chunk_frames,
                    overlap_frames=0,
                    min_chunk_ratio=config.min_chunk_ratio,
                )

        # Create collator
        collator = ClefCollator(
            pad_token_id=tokenizer.vocab['<pad>'],
            max_seq_len=config.max_seq_len,
        )

        # Create data loaders
        if world_size > 1:
            # Use DistributedBucketSampler for efficient padding
            train_sampler = DistributedBucketSampler(
                train_dataset,
                batch_size=args.batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )
        else:
            # Single GPU: use regular BucketSampler
            train_sampler = BucketSampler(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )

        valid_loader = None
        if valid_dataset is not None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )

        # Sanity check
        if args.sanity_check:
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
            sanity_check(model, train_loader, device)
            return

        # Setup wandb
        use_wandb = args.wandb and HAS_WANDB and rank == 0
        if args.wandb and not HAS_WANDB:
            logger.warning('wandb not installed, skipping logging')
        if use_wandb:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=f'train-{world_size}gpu',
                config={
                    'batch_size': args.batch_size,
                    'world_size': world_size,
                    'effective_batch_size': args.batch_size * world_size * args.gradient_accumulation_steps,
                    'gradient_accumulation_steps': args.gradient_accumulation_steps,
                    'gradient_clip': args.gradient_clip,
                    'max_epochs': args.max_epochs,
                    'learning_rate': config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
                    'validate_every_n_steps': args.validate_every_n_steps,
                    'model_params': model.get_num_params(),
                },
                tags=['training', f'{world_size}gpu'],
            )
            logger.info(f'Wandb: {wandb.run.url}')

        # Create trainer
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            use_wandb=use_wandb,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clip=args.gradient_clip,
            validate_every_n_steps=args.validate_every_n_steps,
            early_stopping_patience=args.early_stopping_patience,
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(Path(args.resume))

        # Train
        trainer.train(
            max_epochs=args.max_epochs,
            checkpoint_dir=Path(args.checkpoint_dir),
        )

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
