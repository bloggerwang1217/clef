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
from src.clef.piano.clef_piano_tiny import ClefPianoTiny
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
        save_every_n_epochs: int = 10,
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
        self.save_every_n_epochs = save_every_n_epochs

        # Model
        self.model = model.to(self.device)
        if world_size > 1:
            use_gc = getattr(config, 'gradient_checkpointing', False)
            self.model = DDP(
                self.model, device_ids=[local_rank],
                find_unused_parameters=not use_gc,
                static_graph=use_gc,
            )

        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Optimizer setup:
        # - 2 groups: other (base_lr) + swin (0.1x)
        # Window CA architecture: no deformable offsets, no time_prior — all params use base_lr.
        # Swin unfrozen params use 0.1x to preserve pretrained features.
        base_lr = config.learning_rate if hasattr(config, 'learning_rate') else 1e-4
        swin_lr_scale = getattr(config, 'swin_lr_scale', 0.1)
        swin_lr = base_lr * swin_lr_scale
        weight_decay = config.weight_decay if hasattr(config, 'weight_decay') else 0.01
        # Fine-tuning Swin: smaller WD preserves pretrained representations.
        # Defaults to 0 (rely on reduced LR alone); set explicitly to match main WD if training from scratch.
        swin_weight_decay = getattr(config, 'swin_weight_decay', 0.0)

        # No-decay parameter names: embeddings, norms, biases, Mamba SSM state params.
        # Mamba's A_log encodes the SSM poles (sensitive to magnitude shrinking).
        NO_DECAY_SUFFIXES = ('bias',)
        NO_DECAY_KEYWORDS = (
            'norm', 'ln_',               # LayerNorm / RMSNorm weights
            '.embedding', 'token_emb',   # token / bar PE embeddings
            'A_log', '.D',               # Mamba SSM poles and skip
        )

        def _is_no_decay(name: str) -> bool:
            if name.endswith(NO_DECAY_SUFFIXES):
                return True
            return any(kw in name for kw in NO_DECAY_KEYWORDS)

        # Four groups: (swin / other) × (decay / no_decay)
        swin_decay, swin_nodecay = [], []
        other_decay, other_nodecay = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            nodecay = _is_no_decay(name)
            if name.startswith('swin.'):
                (swin_nodecay if nodecay else swin_decay).append(param)
                if self.rank == 0:
                    logger.debug(f'Swin param (lr={swin_lr:.2e}, wd={"0" if nodecay else weight_decay}): {name}')
            else:
                (other_nodecay if nodecay else other_decay).append(param)

        n_decay   = len(other_decay)   + len(swin_decay)
        n_nodecay = len(other_nodecay) + len(swin_nodecay)
        if self.rank == 0:
            logger.info(
                f'Optimizer: {n_decay} decay params (wd={weight_decay}), '
                f'{n_nodecay} no-decay params (wd=0) | '
                f'base_lr={base_lr:.2e}, swin_lr={swin_lr:.2e}, swin_wd={swin_weight_decay}'
            )

        param_groups = [
            {'params': other_decay,   'lr': base_lr,  'weight_decay': weight_decay},
            {'params': other_nodecay, 'lr': base_lr,  'weight_decay': 0.0},
        ]
        if swin_decay or swin_nodecay:
            param_groups += [
                {'params': swin_decay,   'lr': swin_lr, 'weight_decay': swin_weight_decay},
                {'params': swin_nodecay, 'lr': swin_lr, 'weight_decay': 0.0},
            ]

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)  # per-group WD above

        # Learning rate scheduler
        total_steps = len(train_loader) * (config.max_epochs if hasattr(config, 'max_epochs') else 50)
        total_steps = total_steps // gradient_accumulation_steps
        self.total_steps = total_steps
        warmup_steps = config.warmup_steps if hasattr(config, 'warmup_steps') else 1000
        
        # LR schedule: Cosine with warmup + min LR
        # - Steps 0 → warmup_steps: linear warmup from 0 to max_lr
        # - Steps warmup_steps → total_steps: cosine decay from max_lr to min_lr (not 0)
        # - min_lr = max_lr / 10 (default, configurable via lr_min_ratio)
        lr_min_ratio = getattr(config, 'lr_min_ratio', 0.1)

        # Build one lambda per param group; no-decay groups share LR schedule with decay peers.
        other_lambda = lambda step: self._get_lr_scale(step, base_lr, base_lr * lr_min_ratio, warmup_steps, total_steps)
        swin_lambda  = lambda step: self._get_lr_scale(step, swin_lr, swin_lr * lr_min_ratio, warmup_steps, total_steps)
        lr_lambdas = [other_lambda, other_lambda]           # other_decay, other_nodecay
        if swin_decay or swin_nodecay:
            lr_lambdas += [swin_lambda, swin_lambda]        # swin_decay, swin_nodecay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambdas,
        )
        self.base_lr = base_lr
        self.min_lr = base_lr * lr_min_ratio
        has_swin = bool(swin_decay or swin_nodecay)
        self.swin_lr = swin_lr if has_swin else None
        self.swin_min_lr = swin_lr * lr_min_ratio if has_swin else None

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

    def _get_guidance_weight(self) -> float:
        """Cosine-decay guidance_loss_weight, starting decay after warmup.

        Schedule:
          Steps 0 → warmup_steps              : weight = guidance_loss_weight  (constant)
          Steps warmup_steps → decay_end      : cosine decay → guidance_loss_weight_end
          Steps > decay_end                   : weight = guidance_loss_weight_end (constant)

        guidance_decay_steps sets the number of steps AFTER warmup for decay
        (e.g., guidance_decay_steps=2500 means decay from warmup_steps to warmup_steps+2500).
        If 0 or unset, falls back to total_steps - warmup_steps (old behaviour).
        """
        import math
        start = self.config.guidance_loss_weight
        end = getattr(self.config, 'guidance_loss_weight_end', start)
        if start == end or self.global_step <= self.warmup_steps:
            return start
        
        # guidance_decay_steps is now relative to warmup_steps (duration of decay, not absolute step)
        decay_duration = getattr(self.config, 'guidance_decay_steps', 0) or (self.total_steps - self.warmup_steps)
        decay_end = self.warmup_steps + decay_duration
        
        if self.global_step >= decay_end:
            return end
        
        # Cosine decay from start to end over [warmup_steps, decay_end]
        decay_steps = max(decay_duration, 1)
        t = (self.global_step - self.warmup_steps) / decay_steps
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))

    def _get_lr_scale(self, step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
        """Return LR scale factor for LambdaLR scheduler.

        Schedule (WSD — Warmup → Stable → Decay):
          Steps 0          → warmup_steps : linear warmup 0 → 1.0
          Steps warmup_steps → flat_end   : flat plateau at 1.0  (grokking zone)
          Steps flat_end   → total_steps  : cosine decay 1.0 → min_lr/max_lr
          Steps >= total_steps             : hold at min_lr/max_lr

        lr_flat_ratio (config, default 0.0) controls the fraction of the
        post-warmup budget spent on the flat plateau.
          0.0 → original cosine-only behaviour (backward compatible)
          0.6 → 60% of post-warmup steps are flat, 40% cosine decay
        """
        import math

        flat_ratio = getattr(self.config, 'lr_flat_ratio', 0.0)
        post_warmup = max(total_steps - warmup_steps, 1)
        flat_steps = int(flat_ratio * post_warmup)
        flat_end = warmup_steps + flat_steps

        # Linear warmup
        if step < warmup_steps:
            return step / max(warmup_steps, 1)

        # Flat plateau
        if step < flat_end:
            return 1.0

        # Hold at min after total_steps
        if step >= total_steps:
            return min_lr / max_lr

        # Cosine decay: flat_end → total_steps
        decay_steps = max(total_steps - flat_end, 1)
        t = (step - flat_end) / decay_steps  # 0 → 1
        min_scale = min_lr / max_lr
        return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * t))

    def _get_tf_ratio(self) -> float:
        """Linear decay of teacher-forcing ratio for BarGRU scheduled sampling.

        Schedule (per bar-gru-redesign.md §TF Ratio Schedule):
          step 0 → warmup_steps         : 1.0   (full GT, model still unstable)
          warmup_steps → tf_anneal_steps: 1.0 → 0.0  (linear decay)
          tf_anneal_steps and beyond    : 0.0   (inference-equivalent, no exposure bias)

        tf_anneal_steps is read from config (default 5000 if not set).
        """
        tf_anneal_steps = getattr(self.config, 'tf_anneal_steps', 5000)
        if self.global_step <= self.warmup_steps:
            return 1.0
        if self.global_step >= tf_anneal_steps:
            return 0.0
        decay_steps = max(tf_anneal_steps - self.warmup_steps, 1)
        t = (self.global_step - self.warmup_steps) / decay_steps
        return max(0.0, 1.0 - t)

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
            '<continue>', '<nl>', '<split>', '<merge>', '<*>',
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
        epoch_ce_loss_accum = 0.0  # running sum of avg_accum_loss for epoch-level reporting
        micro_batch_count = 0  # count successful micro-batches (not raw batch_idx)

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch}',
            disable=self.rank != 0,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # DDP-safe skip: all ranks must agree to skip, otherwise
            # the rank that skips misses a gradient all-reduce and NCCL
            # times out waiting for the missing collective.
            if isinstance(self.model, DDP):
                has_data = torch.tensor(
                    [batch is not None], device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0:
                    continue
            elif batch is None:
                continue

            # Move to device
            mel = batch['mel'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            mel_valid_ratios = batch['mel_valid_ratios'].to(self.device)
            micro_batch_count += 1

            # Skip DDP all-reduce on non-update micro-batches to save
            # communication overhead during gradient accumulation.
            is_update_step = micro_batch_count % self.gradient_accumulation_steps == 0
            maybe_no_sync = (
                self.model.no_sync()
                if isinstance(self.model, DDP) and not is_update_step
                else nullcontext()
            )

            with maybe_no_sync:
                # Forward pass with mixed precision
                current_guidance_weight = self._get_guidance_weight()
                current_tf_ratio = self._get_tf_ratio()
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    logits, ce_loss, total_loss = self.model(
                        mel=mel,
                        input_ids=input_ids,
                        labels=labels,
                        mel_valid_ratios=mel_valid_ratios,
                        chunk_audio_measures=batch.get('chunk_audio_measures'),
                        chunk_start_frames=batch.get('chunk_start_frames'),
                        chunk_end_frames=batch.get('chunk_end_frames'),
                        guidance_loss_weight=current_guidance_weight,
                        tf_ratio=current_tf_ratio,
                    )
                    # Scale loss for gradient accumulation
                    total_loss = total_loss / self.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            accumulated_loss += ce_loss.item()

            # Update weights every gradient_accumulation_steps
            if is_update_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping — clip all trainable params together
                all_params = [p for g in self.optimizer.param_groups for p in g['params']]
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.gradient_clip)

                if self.scaler is not None:
                    old_scale = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # Only step scheduler if optimizer actually updated
                    # (scaler skips on inf/NaN gradients)
                    if self.scaler.get_scale() >= old_scale:
                        self.scheduler.step()
                else:
                    self.optimizer.step()
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # Update metrics (average over accumulation steps)
                avg_accum_loss = accumulated_loss / self.gradient_accumulation_steps
                epoch_ce_loss_accum += avg_accum_loss
                num_batches += 1
                self.global_step += 1

                # Sync curriculum step on decoder for build_curriculum_input
                _model = self.model.module if hasattr(self.model, 'module') else self.model
                _decoder = getattr(_model, 'decoder', None)
                if _decoder is not None:
                    _decoder._curriculum_step = self.global_step


                # Log to wandb
                if self.use_wandb:
                    last_lrs = self.scheduler.get_last_lr()
                    log_dict = {
                        'train/loss': avg_accum_loss,
                        'train/grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                        'train/lr_base': last_lrs[0],         # group[0]: other (base_lr)
                        'train/epoch': self.epoch,
                        'system/gpu_memory_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                    }

                    # Guidance/CIF loss breakdown (last micro-batch)
                    decoder = getattr(self.model.module if hasattr(self.model, 'module')
                                      else self.model, 'decoder', None)
                    guidance_loss_cached = getattr(decoder, '_cached_guidance_loss', None)
                    if guidance_loss_cached is not None:
                        log_dict['train/guidance_loss'] = guidance_loss_cached.item()
                        log_dict['train/guidance_weight'] = current_guidance_weight
                    cif_qty_loss_cached = getattr(decoder, '_cif_quantity_loss', None)
                    if cif_qty_loss_cached is not None:
                        log_dict['train/cif_quantity_loss'] = cif_qty_loss_cached.item()
                        # Also log sum_alpha and target for diagnosis
                        cif_sum_alpha = getattr(decoder, '_cif_sum_alpha', None)
                        cif_target = getattr(decoder, '_cif_target', None)
                        if cif_sum_alpha is not None:
                            log_dict['train/cif_sum_alpha'] = cif_sum_alpha
                        if cif_target is not None:
                            log_dict['train/cif_target'] = cif_target
                        # Log CIF bias to monitor training progress
                        if decoder.cif is not None:
                            cif_bias = decoder.cif.weight_proj.bias.item()
                            log_dict['train/cif_bias'] = cif_bias
                    log_dict['train/tf_ratio'] = current_tf_ratio

                    if len(last_lrs) > 1:
                        log_dict['train/lr_swin'] = last_lrs[1]  # group[1]: swin (swin_lr)

                    # Per-token-type loss breakdown (last micro-batch only)
                    breakdown = self._compute_loss_breakdown(logits.detach(), labels)
                    for k, v in breakdown.items():
                        log_dict[f'train/{k}'] = v

                    wandb.log(log_dict, step=self.global_step)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_accum_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
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
                            # Add quantity loss if available
                            if 'quantity_loss' in valid_metrics:
                                valid_log['valid/quantity_loss'] = valid_metrics['quantity_loss']
                            # Add CIF bias
                            decoder = getattr(self.model.module if hasattr(self.model, 'module')
                                              else self.model, 'decoder', None)
                            if decoder is not None and decoder.cif is not None:
                                cif_bias = decoder.cif.weight_proj.bias.item()
                                valid_log['valid/cif_bias'] = cif_bias
                            for k in ('loss_pitch', 'loss_duration', 'loss_struct', 'loss_schema'):
                                if k in valid_metrics:
                                    valid_log[f'valid/{k}'] = valid_metrics[k]
                            wandb.log(valid_log, step=self.global_step)
                    self.model.train()

        avg_loss = epoch_ce_loss_accum / max(num_batches, 1)

        return {'train_loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.valid_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_qty_loss = 0.0  # CIF quantity loss accumulator
        num_batches = 0
        # Accumulators for per-type loss breakdown
        type_loss_sums = {'loss_pitch': 0.0, 'loss_duration': 0.0, 'loss_struct': 0.0, 'loss_schema': 0.0}
        type_loss_counts = {'loss_pitch': 0, 'loss_duration': 0, 'loss_struct': 0, 'loss_schema': 0}

        for batch_idx, batch in enumerate(self.valid_loader):
            # DDP-safe skip (same as train_epoch)
            if isinstance(self.model, DDP):
                has_data = torch.tensor(
                    [batch is not None], device=self.device, dtype=torch.int32
                )
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0:
                    if self.rank == 0:
                        logger.warning(f'Validation: skipping batch {batch_idx} (filtered on some rank)')
                    continue
            elif batch is None:
                if self.rank == 0:
                    logger.warning(f'Validation: skipping batch {batch_idx} (all samples filtered)')
                continue
            mel = batch['mel'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            mel_valid_ratios = batch['mel_valid_ratios'].to(self.device)

            with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                logits, ce_loss, _ = self.model(
                    mel=mel,
                    input_ids=input_ids,
                    labels=labels,
                    mel_valid_ratios=mel_valid_ratios,
                )

            total_loss += ce_loss.item()
            num_batches += 1

            # Accumulate CIF quantity loss (if available)
            model_unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
            cif_qty_loss = getattr(model_unwrapped, '_cif_qty_loss', None)
            if cif_qty_loss is not None:
                total_qty_loss += cif_qty_loss.item()

            # Accumulate per-type losses
            breakdown = self._compute_loss_breakdown(logits, labels)
            for k, v in breakdown.items():
                type_loss_sums[k] += v
                type_loss_counts[k] += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_qty_loss = total_qty_loss / max(num_batches, 1)

        result = {'valid_loss': avg_loss, 'quantity_loss': avg_qty_loss}
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
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing:
            logger.info(f'New params (not in checkpoint): {missing}')
        if unexpected:
            logger.warning(f'Unexpected params in checkpoint: {unexpected}')

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch'] + 1  # resume from NEXT epoch
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

            if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
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
    chunk_audio_measures = batch.get('chunk_audio_measures')
    chunk_start_frames = batch.get('chunk_start_frames')
    chunk_end_frames = batch.get('chunk_end_frames')

    logger.info(f'Batch shapes: mel={mel.shape}, input_ids={input_ids.shape}, labels={labels.shape}')
    has_guidance = chunk_audio_measures is not None and any(m is not None for m in chunk_audio_measures)
    logger.info(f'Guided attention: {"enabled" if has_guidance else "disabled (no alignment info)"}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(100):
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, ce_loss, total_loss = model(
                mel=mel,
                input_ids=input_ids,
                labels=labels,
                mel_valid_ratios=mel_valid_ratios,
                chunk_audio_measures=chunk_audio_measures,
                chunk_start_frames=chunk_start_frames,
                chunk_end_frames=chunk_end_frames,
            )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            g_loss = getattr(model.decoder, '_cached_guidance_loss', None)
            g_str = f'  guidance={g_loss.item():.4f}' if g_loss is not None else ''
            logger.info(f'Step {step}: ce={ce_loss.item():.4f}{g_str}')

    import math
    logger.info(f'Final loss: {ce_loss.item():.4f}  (perplexity={math.exp(ce_loss.item()):.2f})')

    # Loss breakdown: pitch / duration / structure
    from src.clef.piano.tokenizer import KernTokenizer
    tokenizer = KernTokenizer()
    tok = tokenizer

    def _mask(fn):
        ids = [i for i in range(model.config.vocab_size) if fn(tok.id_to_token.get(i, ''))]
        m = torch.zeros(model.config.vocab_size, dtype=torch.bool)
        m[ids] = True
        return m.to(device)

    dur_chars = set('0123456789')
    pitch_chars = set('abcdefgrABCDEFGR')
    dur_mask   = _mask(lambda t: bool(t) and t[0] in dur_chars and t not in ('<sos>','<eos>','<pad>','<bar>','<nl>','<coc>','<continue>'))
    pitch_mask = _mask(lambda t: bool(t) and t[0] in pitch_chars)
    struct_mask= _mask(lambda t: t in ('<bar>','<nl>','<coc>','<split>','<merge>','<*>','<sos>','<eos>'))

    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits_eval, _, _ = model(mel=mel, input_ids=input_ids,
                                  labels=labels, mel_valid_ratios=mel_valid_ratios)
    ce = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
    per_tok = ce(logits_eval.view(-1, logits_eval.size(-1)), labels.view(-1))  # [B*S]
    lab_flat = labels.view(-1)

    def _group_loss(mask):
        sel = mask[lab_flat] & (lab_flat != 0)
        return per_tok[sel].mean().item() if sel.any() else float('nan')

    logger.info(f'  pitch  loss: {_group_loss(pitch_mask):.4f}  dur loss: {_group_loss(dur_mask):.4f}  struct loss: {_group_loss(struct_mask):.4f}')

    if ce_loss.item() < 0.1:
        logger.info('Sanity check PASSED: model can overfit one batch')
    else:
        logger.warning('Sanity check: loss still high, may need more steps')


def main():
    # Handle Ctrl+C gracefully - finish wandb and kill child processes
    import signal
    import os

    def cleanup_and_exit(signum, frame):
        print("\n[INFO] Caught signal, cleaning up...", flush=True)
        # Finish wandb run (so it doesn't stay "running" on server)
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass
        # Kill all processes in our process group
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    parser = argparse.ArgumentParser(description='Train clef-piano-base')
    parser.add_argument('--config', type=str, default='configs/clef_piano_base.yaml',
                        help='Path to config YAML (single source of truth for all hyperparameters)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Run sanity check (overfit one batch)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='clef-piano-base',
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity (team/user); defaults to wandb default entity if not set')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (defaults to train-{N}gpu)')
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

        # Create model based on model name in config
        model_name = getattr(config, 'name', 'clef-piano-base')
        if model_name == 'clef-piano-tiny':
            model = ClefPianoTiny(config)
        else:
            model = ClefPianoBase(config)
        logger.info(f'Model created ({model_name}): {model.get_num_params():,} trainable params')

        # Create datasets
        manifest_dir = Path(config.manifest_dir)
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
                max_seq_len=config.max_seq_len,
                fallback_chunk_frames=config.fallback_chunk_frames,
                fallback_overlap_frames=config.fallback_overlap_frames,
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
                    max_seq_len=config.max_seq_len,
                    fallback_chunk_frames=config.fallback_chunk_frames,
                    fallback_overlap_frames=0,
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
                batch_size=config.batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )
        else:
            # Single GPU: use regular BucketSampler
            train_sampler = BucketSampler(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )

        valid_loader = None
        if valid_dataset is not None:
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.batch_size,
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
                name=args.wandb_run_name or f'train-{world_size}gpu',
                config={
                    'batch_size': config.batch_size,
                    'world_size': world_size,
                    'effective_batch_size': config.batch_size * world_size * config.gradient_accumulation_steps,
                    'gradient_accumulation_steps': config.gradient_accumulation_steps,
                    'gradient_clip': config.gradient_clip,
                    'max_epochs': config.max_epochs,
                    'learning_rate': config.learning_rate,
                    'validate_every_n_steps': config.validate_every_n_steps,
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
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_clip=config.gradient_clip,
            validate_every_n_steps=config.validate_every_n_steps,
            early_stopping_patience=config.early_stopping_patience,
            save_every_n_epochs=config.save_every_n_epochs,
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(Path(args.resume))

        # Train
        trainer.train(
            max_epochs=config.max_epochs,
            checkpoint_dir=Path(config.checkpoint_dir),
        )

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
