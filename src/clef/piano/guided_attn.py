"""Guided Attention Loss Utilities

Computes per-token measure bounds for supervising L1 SAFullCALayer
to attend to the correct audio time region via a CoM hinge loss.

Design:
- Each note token at position s belongs to measure m, determined by
  counting <bar> tokens before position s in input_ids.
- bounds[b, s] = (norm_start, norm_end) for note tokens AND <bar> tokens:
  the normalized time range [0,1] of the measure in the chunk.
- Other structural tokens (IDs 0-3, 5-9) receive sentinel (-1, -1) and are skipped.

Loss (CoM hinge, applied in SAFullCALayer):
  CoM[s] = Σ_j A[s,j] * time_norm[j]           # weighted mean of attention positions
  loss[s] = relu(norm_start - CoM) + relu(CoM - norm_end)
           = 0  when CoM is within the correct measure  (zero gradient inside interval)
           = linear penalty otherwise

This is strictly less demanding than CE-form (Liu et al. 2016):
- CE forces a specific distribution over N_kv positions (3760 positions at
  0.64s+1.28s resolution — far more precision than bar-level L1 can provide)
- CoM hinge only constrains the first moment (scalar center), zero gradient
  when the attention center is already in the right measure.

Memory: O(B·S) vs O(B·S·N_kv) for CE-form — massive reduction.
"""

import torch
from typing import List, Optional

# <bar> token ID (from tokenizer vocabulary)
BAR_TOKEN_ID = 4

# IDs 0-9: pad, sos, eos, coc, bar, continue, nl, split, merge, <*>
# These tokens do not correspond to individual notes → skip guidance.
STRUCTURAL_TOKEN_MAX_ID = 9


def build_guidance_bounds(
    input_ids: torch.Tensor,                              # [B, S]
    chunk_audio_measures_list: List[Optional[List[dict]]],  # per-sample measure timing
    chunk_start_frames: List[int],                        # per-sample chunk start (mel frames)
    chunk_end_frames: List[int],                          # per-sample chunk end (mel frames)
    mel_fps: float = 100.0,
) -> Optional[torch.Tensor]:                              # [B, S, 2] or None
    """Build per-token guidance bounds for CoM hinge loss in L1 Full CA.

    For each note token AND <bar> token at position s, returns the normalized
    time range (norm_start, norm_end) of its containing audio measure.
    Other structural tokens (IDs 0-3, 5-9) receive sentinel (-1, -1) and are
    excluded from the loss by the ``valid = lo >= 0`` mask in SAFullCALayer.

    Measure assignment: token at position s is in measure m where m equals
    the number of <bar> tokens seen before position s in input_ids.
    <bar> tokens are assigned to the measure they close.

    Args:
        input_ids: Decoder input IDs [B, S].
        chunk_audio_measures_list: Per-sample list of measure timing dicts
            [{'start_sec': float, 'end_sec': float}, ...]. None if absent.
        chunk_start_frames: Per-sample chunk start frame in mel spectrogram.
        chunk_end_frames: Per-sample chunk end frame in mel spectrogram.
        mel_fps: Mel frames per second (default 100).

    Returns:
        bounds [B, S, 2] float32 on same device as input_ids, or None if
        no guidance info is available for any sample in the batch.
        bounds[:, :, 0] = norm_start,  bounds[:, :, 1] = norm_end.
        Structural tokens have value -1 in both channels.
    """
    B, S = input_ids.shape
    device = input_ids.device

    # Initialize with sentinel; structural tokens stay at -1
    bounds = torch.full((B, S, 2), -1.0, device=device, dtype=torch.float32)
    any_valid = False

    for b in range(B):
        measures = chunk_audio_measures_list[b]
        if not measures:
            continue

        chunk_start_sec = chunk_start_frames[b] / mel_fps
        chunk_end_sec   = chunk_end_frames[b]   / mel_fps
        chunk_dur_sec   = chunk_end_sec - chunk_start_sec
        if chunk_dur_sec <= 0.0:
            continue

        ids = input_ids[b]  # [S]
        is_bar = (ids == BAR_TOKEN_ID).long()
        # cumbar[s] = # <bar> tokens in input_ids[b, 0 : s]
        cumbar = torch.cat([
            is_bar.new_zeros(1),
            torch.cumsum(is_bar[:-1], dim=0),
        ])  # [S]

        n_measures = len(measures)

        # Note-token mask: IDs > STRUCTURAL_TOKEN_MAX_ID
        is_note = ids > STRUCTURAL_TOKEN_MAX_ID  # [S]

        for m_idx in range(n_measures):
            # Note tokens in measure m_idx
            tok_mask = (cumbar == m_idx) & is_note  # [S]
            # <bar> token ending measure m_idx (cumbar[s] == m_idx for <bar> at s)
            bar_mask = (cumbar == m_idx) & is_bar.bool()  # [S]

            if not (tok_mask.any() or bar_mask.any()):
                continue

            m = measures[m_idx]
            norm_start = (m['start_sec'] - chunk_start_sec) / chunk_dur_sec
            norm_end   = (m['end_sec']   - chunk_start_sec) / chunk_dur_sec
            norm_start = max(0.0, min(1.0, norm_start))
            norm_end   = max(0.0, min(1.0, norm_end))
            if norm_end <= norm_start:
                continue

            # Supervise both note tokens and <bar> token in this measure
            bounds[b][tok_mask, 0] = norm_start
            bounds[b][tok_mask, 1] = norm_end
            bounds[b][bar_mask, 0] = norm_start
            bounds[b][bar_mask, 1] = norm_end
            any_valid = True

        # Tail tokens (notes and bars) beyond the last measure → last measure's range
        tok_tail = (cumbar >= n_measures) & is_note
        bar_tail = (cumbar >= n_measures) & is_bar.bool()
        if tok_tail.any() or bar_tail.any():
            m = measures[-1]
            norm_start = max(0.0, min(1.0,
                (m['start_sec'] - chunk_start_sec) / chunk_dur_sec))
            norm_end = max(0.0, min(1.0,
                (m['end_sec'] - chunk_start_sec) / chunk_dur_sec))
            if norm_end > norm_start:
                bounds[b][tok_tail, 0] = norm_start
                bounds[b][tok_tail, 1] = norm_end
                bounds[b][bar_tail, 0] = norm_start
                bounds[b][bar_tail, 1] = norm_end

    if not any_valid:
        return None

    return bounds
