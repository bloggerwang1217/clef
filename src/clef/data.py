"""
Chunked Dataset for Long Audio
==============================

Wraps a base dataset to split long pieces into overlapping chunks.
Each chunk is treated as an independent sample during training.

Design rationale:
- 8 min piece -> 3 chunks (4 min each, 2 min overlap)
- 10 min piece -> 4 chunks
- <= 4 min -> 1 chunk (original piece)

Benefits:
- Every part of long pieces gets trained
- Overlap regions learn "how to connect"
- No data wasted
- Sample count +35% approximately
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ChunkedDataset(Dataset):
    """Dataset wrapper that splits long audio into overlapping chunks.

    This is essential for training on long piano pieces (often 5-15 minutes)
    while keeping memory usage manageable.

    Args:
        base_dataset: Underlying dataset that returns (mel, tokens, metadata)
        chunk_frames: Maximum chunk length in mel frames (default: 4 min @ 100 fps)
        overlap_frames: Overlap between chunks in frames (default: 2 min)
        min_chunk_ratio: Minimum ratio of chunk_frames for the last chunk (default: 0.5)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: Any = None,  # KernTokenizer for re-tokenizing chunk kern
        chunk_frames: int = 24000,      # 4 min @ 100 fps
        overlap_frames: int = 12000,    # 2 min overlap
        min_chunk_ratio: float = 0.5,   # Last chunk at least 2 min
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.chunk_frames = chunk_frames
        self.overlap_frames = overlap_frames
        self.stride = chunk_frames - overlap_frames  # 12000 frames = 2 min
        self.min_chunk_frames = int(chunk_frames * min_chunk_ratio)

        # Pre-compute all chunks
        self.chunks = self._create_chunks()

        logger.info(
            f"ChunkedDataset: {len(self.base_dataset)} pieces -> {len(self.chunks)} chunks "
            f"(chunk={chunk_frames} frames, overlap={overlap_frames} frames)"
        )

    def _create_chunks(self) -> List[Tuple[int, int, int]]:
        """Pre-compute all chunk boundaries.

        Returns:
            List of (base_idx, start_frame, end_frame) tuples
        """
        chunks = []

        for idx in range(len(self.base_dataset)):
            length = self._get_audio_length(idx)

            if length <= self.chunk_frames:
                # Short piece: treat as single chunk
                chunks.append((idx, 0, length))
            else:
                # Long piece: split into overlapping chunks
                start = 0
                while start + self.min_chunk_frames <= length:
                    end = min(start + self.chunk_frames, length)
                    chunks.append((idx, start, end))

                    if end >= length:
                        break
                    start += self.stride

        return chunks

    def _get_audio_length(self, idx: int) -> int:
        """Get audio length in frames for a base dataset item.

        This requires the base dataset to support get_mel_length() or similar.
        Falls back to loading the item if not available.
        """
        # Try to get length without loading full data
        if hasattr(self.base_dataset, 'get_mel_length'):
            return self.base_dataset.get_mel_length(idx)
        elif hasattr(self.base_dataset, 'get_audio_length'):
            return self.base_dataset.get_audio_length(idx)
        else:
            # Fallback: load and check shape (less efficient)
            mel, _, _ = self.base_dataset[idx]
            return mel.shape[-1]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        """Get a chunk.

        Returns:
            Tuple of (mel_chunk, token_ids, metadata)
            - mel_chunk: [1, n_mels, T_chunk] tensor
            - token_ids: List of token IDs for this chunk
            - metadata: Dict with original piece info and chunk boundaries
        """
        base_idx, start_frame, end_frame = self.chunks[idx]
        mel, tokens, meta = self.base_dataset[base_idx]

        # Slice mel spectrogram
        mel_chunk = mel[..., start_frame:end_frame]

        # Token slicing: use audio_measures + kern_measures alignment
        # If alignment info is missing, we MUST NOT silently return unaligned
        # tokens (e.g. beginning-of-piece tokens for a middle chunk), as this
        # would feed garbage pairs to the model.
        audio_measures = meta.get('audio_measures', [])
        kern_measures = meta.get('kern_measures', [])
        is_last_chunk = True  # default: single chunk or no alignment info

        if not audio_measures or not kern_measures or self.tokenizer is None:
            if start_frame == 0:
                # First (and only) chunk without alignment: tokens from
                # ManifestDataset are acceptable since both start from 0.
                pass
            else:
                name = meta.get('name', f'base_idx={base_idx}')
                logger.error(
                    f"Chunk {idx} ({name}): missing alignment info for "
                    f"non-first chunk [{start_frame}:{end_frame}], skipping"
                )
                return None
        else:
            # Convert frames to seconds (100 fps for mel spectrogram)
            start_sec = start_frame / 100.0
            end_sec = end_frame / 100.0

            # Find overlapping audio measures
            first_m = None
            last_m = None
            for i, am in enumerate(audio_measures):
                if am['end_sec'] > start_sec and am['start_sec'] < end_sec:
                    if first_m is None:
                        first_m = i
                    last_m = i

            if (first_m is not None and last_m is not None
                    and first_m < len(kern_measures) and last_m < len(kern_measures)):
                line_start = kern_measures[first_m]['line_start']
                line_end = kern_measures[last_m]['line_end']
                is_last_chunk = (last_m == len(audio_measures) - 1)

                # Read relevant kern lines and re-tokenize
                kern_path = meta.get('kern_path', '')
                if kern_path:
                    with open(kern_path, 'r', encoding='utf-8') as f:
                        all_lines = f.readlines()
                    chunk_kern = ''.join(all_lines[line_start - 1:line_end])
                    tokens = self.tokenizer.encode(chunk_kern)

                    # tokenizer.encode() always appends <eos>.
                    # For non-final chunks, replace <eos> with <continue>
                    # so the model learns "this chunk ends but the piece
                    # continues" vs "the piece is finished".
                    if not is_last_chunk and tokens:
                        eos_id = self.tokenizer.vocab["<eos>"]
                        cont_id = self.tokenizer.vocab["<continue>"]
                        if tokens[-1] == eos_id:
                            tokens[-1] = cont_id
            else:
                name = meta.get('name', f'base_idx={base_idx}')
                logger.error(
                    f"Chunk {idx} ({name}): measure alignment failed "
                    f"(first_m={first_m}, last_m={last_m}, "
                    f"kern_measures={len(kern_measures)}), skipping"
                )
                return None

        # Update metadata
        chunk_meta = {
            **meta,
            'chunk_start_frame': start_frame,
            'chunk_end_frame': end_frame,
            'chunk_idx': idx,
            'base_idx': base_idx,
            'is_chunked': True,
            'is_last_chunk': is_last_chunk,
        }

        return mel_chunk, tokens, chunk_meta

    def get_chunk_length(self, idx: int) -> int:
        """Get chunk length in frames (for BucketSampler)."""
        _, start, end = self.chunks[idx]
        return end - start


class ManifestDataset(Dataset):
    """Dataset that loads from pre-computed manifest files.

    Expected manifest format (JSON):
    [
        {
            "mel_path": "path/to/mel.pt",
            "kern_path": "path/to/kern.krn",
            "name": "piece_name",
            "duration_frames": 12345,
            ...
        },
        ...
    ]
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        tokenizer: Any,  # KernTokenizer
        max_seq_len: int = 4096,
        mel_dir: Optional[Path] = None,
        kern_dir: Optional[Path] = None,
        augmentation_metadata_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize ManifestDataset.

        Args:
            manifest_path: Path to JSON manifest file
            tokenizer: KernTokenizer instance
            max_seq_len: Maximum token sequence length
            mel_dir: Optional base directory for mel files
            kern_dir: Optional base directory for kern files
            augmentation_metadata_path: Path to augmentation_metadata.json
                (provides audio_measures and kern_measures for chunk alignment)
        """
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mel_dir = Path(mel_dir) if mel_dir else None
        self.kern_dir = Path(kern_dir) if kern_dir else None

        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        # Load augmentation metadata for alignment info
        self.aug_meta: Dict[str, Any] = {}
        if augmentation_metadata_path:
            aug_path = Path(augmentation_metadata_path)
            if aug_path.exists():
                with open(aug_path, 'r', encoding='utf-8') as f:
                    self.aug_meta = json.load(f)
                logger.info(f"ManifestDataset: loaded augmentation metadata ({len(self.aug_meta)} entries)")
            else:
                logger.warning(f"ManifestDataset: augmentation metadata not found at {aug_path}")

        logger.info(f"ManifestDataset: loaded {len(self.manifest)} items from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
        """Load mel and kern, return tokenized data."""
        item = self.manifest[idx]

        # Resolve paths
        mel_path = self._resolve_path(item['mel_path'], self.mel_dir)
        # Support both 'kern_path' and 'kern_gt_path' for backward compatibility
        kern_key = 'kern_path' if 'kern_path' in item else 'kern_gt_path'
        kern_path = self._resolve_path(item[kern_key], self.kern_dir)

        # Load mel spectrogram
        mel = torch.load(mel_path)  # [1, n_mels, T]

        # Load and tokenize kern
        with open(kern_path, 'r', encoding='utf-8') as f:
            kern_content = f.read()
        tokens = self.tokenizer.encode(kern_content)

        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        metadata = {
            'name': item.get('name', mel_path.stem),
            'mel_path': str(mel_path),
            'kern_path': str(kern_path),
            'duration_frames': item.get('duration_frames', mel.shape[-1]),
        }

        # Add alignment info from augmentation metadata
        item_id = item.get('id', mel_path.stem)
        if item_id in self.aug_meta:
            aug_info = self.aug_meta[item_id]
            if aug_info.get('audio_measures'):
                metadata['audio_measures'] = aug_info['audio_measures']
            if aug_info.get('kern_measures'):
                metadata['kern_measures'] = aug_info['kern_measures']

        return mel, tokens, metadata

    def _resolve_path(self, path: str, base_dir: Optional[Path]) -> Path:
        """Resolve path, optionally relative to base directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        if base_dir:
            return base_dir / p
        return self.manifest_path.parent / p

    def get_mel_length(self, idx: int) -> int:
        """Get mel length in frames without loading."""
        item = self.manifest[idx]
        # Support both 'duration_frames' and 'n_frames'
        if 'n_frames' in item:
            return item['n_frames']
        if 'duration_frames' in item:
            return item['duration_frames']
        # Fallback: load and check
        mel_path = self._resolve_path(item['mel_path'], self.mel_dir)
        mel = torch.load(mel_path, weights_only=True)
        return mel.shape[-1]
