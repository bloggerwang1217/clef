"""
Collate Functions for Clef Training
===================================

Handles batching of variable-length mel spectrograms and token sequences.

Features:
- BucketSampler: Groups similar-length samples to minimize padding
- ClefCollator: Pads sequences and computes valid_ratios for padding handling
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class DistributedBucketSampler(Sampler[int]):
    """Distributed sampler with bucket grouping for multi-GPU training.

    Combines BucketSampler with DistributedSampler:
    - Groups samples by length into buckets (minimize padding)
    - Distributes samples across ranks (DDP support)
    - Ensures each rank gets similar-length samples in each batch

    Usage with DDP:
        sampler = DistributedBucketSampler(dataset, batch_size=3, num_replicas=2, rank=0)
        loader = DataLoader(dataset, batch_size=3, sampler=sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        bucket_boundaries: Optional[List[int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """Initialize DistributedBucketSampler.

        Args:
            dataset: Dataset with get_mel_length(idx) or get_chunk_length(idx) method
            batch_size: Batch size per GPU
            num_replicas: Number of processes (world_size), auto-detected if None
            rank: Process rank, auto-detected if None
            bucket_boundaries: Frame count boundaries (default: [6000, 12000, 18000, 24000])
            shuffle: Whether to shuffle within and across buckets
            drop_last: Whether to drop incomplete batches
            seed: Random seed for reproducibility
        """
        import torch.distributed as dist

        if num_replicas is None:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.bucket_boundaries = bucket_boundaries or [6000, 12000, 18000, 24000]
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Pre-compute bucket assignments
        self.buckets = self._assign_buckets()

        # Compute samples per rank
        self._compute_num_samples()

    def _assign_buckets(self) -> Dict[int, List[int]]:
        """Assign each sample to a bucket based on its length."""
        buckets = defaultdict(list)

        for idx in range(len(self.dataset)):
            length = self._get_length(idx)
            bucket_id = self._get_bucket_id(length)
            buckets[bucket_id].append(idx)

        if self.rank == 0:
            logger.info(
                f"DistributedBucketSampler: {len(self.dataset)} samples -> "
                f"{len(buckets)} buckets: {dict((k, len(v)) for k, v in sorted(buckets.items()))}"
            )

        return dict(buckets)

    def _get_length(self, idx: int) -> int:
        """Get sample length in frames."""
        if hasattr(self.dataset, 'get_chunk_length'):
            return self.dataset.get_chunk_length(idx)
        elif hasattr(self.dataset, 'get_mel_length'):
            return self.dataset.get_mel_length(idx)
        else:
            return 12000

    def _get_bucket_id(self, length: int) -> int:
        """Get bucket ID for a given length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length < boundary:
                return i
        return len(self.bucket_boundaries)

    def _compute_num_samples(self):
        """Compute number of samples for this rank."""
        # Total batches across all buckets
        total_batches = 0
        for indices in self.buckets.values():
            num_batches = len(indices) // (self.batch_size * self.num_replicas)
            if not self.drop_last and len(indices) % (self.batch_size * self.num_replicas) > 0:
                num_batches += 1
            total_batches += num_batches

        # Samples for this rank
        self.num_samples = total_batches * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices for this rank."""
        rng = random.Random(self.seed + self.epoch)

        all_batches = []

        for bucket_id in sorted(self.buckets.keys()):
            indices = self.buckets[bucket_id].copy()

            # Shuffle within bucket
            if self.shuffle:
                rng.shuffle(indices)

            # Pad to make divisible by (batch_size * num_replicas)
            chunk_size = self.batch_size * self.num_replicas
            remainder = len(indices) % chunk_size
            if remainder > 0 and not self.drop_last:
                # Pad with repeated samples
                indices += indices[:chunk_size - remainder]

            # Split into batches of size (batch_size * num_replicas)
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i:i + chunk_size]
                if len(chunk) == chunk_size:
                    # Each rank gets batch_size samples from this chunk
                    # Rank 0: [0:batch_size], Rank 1: [batch_size:2*batch_size], ...
                    rank_batch = chunk[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
                    all_batches.append(rank_batch)

        # Shuffle batches (all ranks use same seed so they stay synchronized)
        if self.shuffle:
            rng.shuffle(all_batches)

        # Yield indices
        for batch in all_batches:
            yield from batch

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        return self.num_samples


class BucketSampler(Sampler[int]):
    """Sampler that groups samples by length to minimize padding waste.

    Bucket design (at 100 fps):
      Bucket 0: < 1 min   (< 6000 frames)
      Bucket 1: 1-2 min   (6000-12000 frames)
      Bucket 2: 2-3 min   (12000-18000 frames)
      Bucket 3: 3-4 min   (18000-24000 frames)
      Bucket 4: > 4 min   (> 24000 frames)

    Benefits:
    - Samples in same batch have similar lengths
    - Less padding -> more efficient GPU utilization
    - Still shuffles within and across buckets for randomness
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        bucket_boundaries: Optional[List[int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        """Initialize BucketSampler.

        Args:
            dataset: Dataset with get_mel_length(idx) or get_chunk_length(idx) method
            batch_size: Batch size
            bucket_boundaries: Frame count boundaries (default: [6000, 12000, 18000, 24000])
            shuffle: Whether to shuffle within and across buckets
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries or [6000, 12000, 18000, 24000]
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Pre-compute bucket assignments
        self.buckets = self._assign_buckets()

    def _assign_buckets(self) -> Dict[int, List[int]]:
        """Assign each sample to a bucket based on its length."""
        buckets = defaultdict(list)

        for idx in range(len(self.dataset)):
            length = self._get_length(idx)
            bucket_id = self._get_bucket_id(length)
            buckets[bucket_id].append(idx)

        logger.info(
            f"BucketSampler: {len(self.dataset)} samples -> "
            f"{len(buckets)} buckets: {dict((k, len(v)) for k, v in buckets.items())}"
        )

        return dict(buckets)

    def _get_length(self, idx: int) -> int:
        """Get sample length in frames."""
        if hasattr(self.dataset, 'get_chunk_length'):
            return self.dataset.get_chunk_length(idx)
        elif hasattr(self.dataset, 'get_mel_length'):
            return self.dataset.get_mel_length(idx)
        else:
            # Fallback: assume all same length
            return 12000

    def _get_bucket_id(self, length: int) -> int:
        """Get bucket ID for a given length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length < boundary:
                return i
        return len(self.bucket_boundaries)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield sample indices in batched order."""
        rng = random.Random(self.seed + self.epoch)

        # Shuffle within each bucket
        bucket_indices = {}
        for bucket_id, indices in self.buckets.items():
            shuffled = indices.copy()
            if self.shuffle:
                rng.shuffle(shuffled)
            bucket_indices[bucket_id] = shuffled

        # Create batches within each bucket
        batches = []
        for bucket_id, indices in bucket_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            rng.shuffle(batches)

        # Yield indices
        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        """Return total number of samples (approximate for drop_last)."""
        if self.drop_last:
            total = 0
            for indices in self.buckets.values():
                total += (len(indices) // self.batch_size) * self.batch_size
            return total
        return len(self.dataset)


class ClefCollator:
    """Collate function for Clef training.

    Handles:
    - Padding mel spectrograms to same length (multiple of 32 for Swin)
    - Padding token sequences to same length
    - Computing mel_valid_ratios for padding handling in attention
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        max_seq_len: int = 4096,
        pad_to_multiple: int = 32,  # Swin window size constraint
    ):
        """Initialize ClefCollator.

        Args:
            pad_token_id: Token ID used for padding (default: 0 = <pad>)
            max_seq_len: Maximum token sequence length
            pad_to_multiple: Pad mel width to multiple of this (for Swin)
        """
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, List[int], Dict[str, Any]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of (mel, token_ids, metadata) tuples
                - mel: [1, n_mels, T] tensor
                - token_ids: List of token IDs
                - metadata: Dict with sample info

        Returns:
            Dict with:
                - mel: [B, 1, n_mels, max_T] padded mel spectrograms
                - mel_lengths: [B] original mel lengths
                - mel_valid_ratios: [B] ratio of valid frames (for attention)
                - input_ids: [B, max_S] input token IDs (shifted right)
                - labels: [B, max_S] target labels
                - label_lengths: [B] original sequence lengths
        """
        # Filter out None samples (skipped by ChunkedDataset due to
        # missing alignment info)
        batch = [s for s in batch if s is not None]
        if not batch:
            return None

        mels = []
        mel_lengths = []
        input_ids_list = []
        labels_list = []

        for mel, tokens, meta in batch:
            # Ensure mel is [1, n_mels, T]
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mels.append(mel)
            mel_lengths.append(mel.shape[-1])

            # Truncate tokens if needed
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]

            # Input: everything except last token
            # Labels: everything except first token
            input_ids_list.append(tokens[:-1])
            labels_list.append(tokens[1:])

        # === Pad mels ===
        max_mel_len = max(mel_lengths)
        # Pad to multiple of 32 (Swin constraint)
        max_mel_len = ((max_mel_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)

        padded_mels = []
        for mel in mels:
            pad_len = max_mel_len - mel.shape[-1]
            if pad_len > 0:
                mel = F.pad(mel, (0, pad_len), value=0)
            padded_mels.append(mel)

        # Compute valid ratios (for padding handling in attention)
        mel_valid_ratios = torch.tensor([
            mel_len / max_mel_len for mel_len in mel_lengths
        ], dtype=torch.float32)

        # === Pad token sequences ===
        seq_lengths = [len(ids) for ids in labels_list]
        max_seq_len = max(seq_lengths)

        padded_input_ids = []
        padded_labels = []
        for inp, lab in zip(input_ids_list, labels_list):
            pad_len = max_seq_len - len(inp)
            padded_input_ids.append(inp + [self.pad_token_id] * pad_len)
            padded_labels.append(lab + [self.pad_token_id] * pad_len)

        return {
            'mel': torch.stack(padded_mels),
            'mel_lengths': torch.tensor(mel_lengths, dtype=torch.long),
            'mel_valid_ratios': mel_valid_ratios,
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'label_lengths': torch.tensor(seq_lengths, dtype=torch.long),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    use_bucket_sampler: bool = True,
    collator: Optional[ClefCollator] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with optional bucket sampling.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if use_bucket_sampler)
        num_workers: Number of data loading workers
        use_bucket_sampler: Whether to use BucketSampler
        collator: Optional custom collator
        **kwargs: Additional arguments for DataLoader

    Returns:
        DataLoader instance
    """
    if collator is None:
        collator = ClefCollator()

    if use_bucket_sampler:
        sampler = BucketSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            **kwargs,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            **kwargs,
        )
