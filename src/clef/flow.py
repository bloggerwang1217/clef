"""
HarmonicFlow: Physics-based Coordinate Transform
=================================================

Per-frame invertible transform from mel frequency space to pitch space.

Neuroscience analogy:
- Mel spectrogram = acoustic signal (microphone output)
- Swin = cochlea (local frequency decomposition, 2D attention)
- HarmonicFlow = brainstem (coordinate transform: frequency -> pitch space)

The transform is a 128x128 square matrix (invertible, lossless):
- Rows 0-87: harmonic templates for 88 piano keys (A0 to C8)
- Rows 88-127: orthogonal complement (captures non-pitch spectral info)

Physics initialization:
- Each pitch template encodes energy at f0, 2*f0, 3*f0, ... (harmonic series)
- Gaussian smoothing accounts for mel filter overlap
- Amplitude weighting 1/h models natural harmonic rolloff
- Orthogonal complement via QR decomposition ensures invertibility

The matrix is a learnable nn.Parameter: physics gives structure,
end-to-end training refines for inharmonicity, polyphonic interference, etc.
"""

import librosa
import numpy as np
import torch
import torch.nn as nn


def _build_harmonic_template(
    n_mels: int = 128,
    n_pitches: int = 88,
    n_harmonics: int = 6,
    f_min: float = 27.5,
    f_max: float = 7040.0,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    sigma: float = 0.5,
) -> torch.Tensor:
    """Build harmonic template matrix using physics of piano overtones.

    For each of the 88 piano keys, creates a template vector in mel-bin space
    that encodes where its fundamental and harmonics fall.

    Args:
        n_mels: Number of mel bins (must match spectrogram)
        n_pitches: Number of piano keys (88: A0 to C8)
        n_harmonics: Number of harmonics to include (1=fundamental only)
        f_min: Minimum frequency (Hz), A0 = 27.5
        f_max: Maximum frequency (Hz)
        sample_rate: Audio sample rate
        n_fft: FFT size (for mel filterbank computation)
        sigma: Gaussian spread in mel bins (accounts for filter overlap)

    Returns:
        template: [n_pitches, n_mels] normalized harmonic templates
    """
    # Get mel filterbank center frequencies
    # librosa.mel_frequencies returns n_mels+2 frequencies (including edges)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=f_min, fmax=f_max)
    # Center frequencies are indices 1 to n_mels (skip edges)
    center_freqs = mel_freqs[1:-1]  # [n_mels]

    # Convert center frequencies to mel scale for distance computation
    center_mels = librosa.hz_to_mel(center_freqs)  # [n_mels]

    # Piano key frequencies: A0 (27.5 Hz) to C8 (4186 Hz)
    # MIDI note 21 (A0) to 108 (C8)
    midi_notes = np.arange(21, 21 + n_pitches)
    key_freqs = 440.0 * (2.0 ** ((midi_notes - 69) / 12.0))  # [n_pitches]

    template = np.zeros((n_pitches, n_mels), dtype=np.float32)

    for p in range(n_pitches):
        f0 = key_freqs[p]
        for h in range(1, n_harmonics + 1):
            fh = f0 * h  # h-th harmonic frequency

            # Skip if harmonic is above Nyquist or f_max
            if fh > min(sample_rate / 2, f_max):
                break

            # Convert harmonic frequency to mel scale
            fh_mel = librosa.hz_to_mel(fh)

            # Gaussian activation centered at harmonic's mel position
            # Distance in mel scale (more perceptually uniform)
            distances = (center_mels - fh_mel) / sigma
            activation = np.exp(-0.5 * distances ** 2)

            # Weight by 1/h (natural harmonic amplitude rolloff)
            template[p] += activation / h

    # Normalize each template to unit norm
    norms = np.linalg.norm(template, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    template = template / norms

    template = torch.from_numpy(template)

    # Stabilize rank: low piano keys have near-identical harmonic patterns in mel
    # space, causing rank deficiency. Lift near-zero singular values to ensure
    # full rank (invertibility) while preserving harmonic structure.
    U, S, Vh = torch.linalg.svd(template, full_matrices=False)
    min_sv = 0.01  # Floor for singular values
    S = torch.clamp(S, min=min_sv)
    template = U @ torch.diag(S) @ Vh

    return template


def _build_orthogonal_complement(
    template: torch.Tensor,
    n_mels: int = 128,
) -> torch.Tensor:
    """Build orthogonal complement to make the full matrix invertible.

    Uses QR decomposition: given template [K, n_mels] with K < n_mels,
    find the orthogonal complement spanning the remaining n_mels - K dimensions.

    Args:
        template: [n_pitches, n_mels] harmonic templates (must have n_pitches < n_mels)
        n_mels: Total mel bins

    Returns:
        complement: [n_mels - n_pitches, n_mels] orthogonal complement rows
    """
    n_pitches = template.shape[0]
    assert n_pitches < n_mels, f"Need n_pitches({n_pitches}) < n_mels({n_mels})"

    # QR decomposition of template.T: [n_mels, n_pitches] = Q @ R
    # Q is [n_mels, n_mels] orthogonal matrix
    # The last (n_mels - n_pitches) columns of Q span the null space
    Q, _ = torch.linalg.qr(template.T, mode='complete')  # Q: [n_mels, n_mels]

    # Complement = last (n_mels - n_pitches) columns of Q, transposed to row form
    complement = Q[:, n_pitches:].T  # [n_mels - n_pitches, n_mels]

    return complement


class HarmonicFlow(nn.Module):
    """Per-frame invertible transform: mel frequency space -> pitch space.

    Applies a 128x128 square matrix to each time frame of the mel spectrogram.
    The matrix is initialized with physics (harmonic templates) and refined
    through end-to-end training.

    Input:  mel [B, 1, 128, T] in frequency space
    Output: pitch activation [B, T, 128] in pitch space
            - First 88 dims: piano key activations
            - Last 40 dims: residual spectral info (orthogonal complement)

    The transform is invertible (det != 0) by construction:
    rows 0-87 come from linearly independent harmonic templates,
    rows 88-127 are their orthogonal complement.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_pitches: int = 88,
        n_harmonics: int = 6,
        f_min: float = 27.5,
        f_max: float = 7040.0,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        init: str = 'harmonic',
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_pitches = n_pitches

        if init == 'harmonic':
            # Physics-based: harmonic templates + orthogonal complement
            # Uses known overtone structure (f0, 2f0, 3f0, ...) as starting point.
            # This is universal acoustics, not instrument-specific.
            template = _build_harmonic_template(
                n_mels=n_mels,
                n_pitches=n_pitches,
                n_harmonics=n_harmonics,
                f_min=f_min,
                f_max=f_max,
                sample_rate=sample_rate,
                n_fft=n_fft,
            )  # [88, 128]
            complement = _build_orthogonal_complement(template, n_mels)  # [40, 128]
            full_matrix = torch.cat([template, complement], dim=0)  # [128, 128]
        elif init == 'orthogonal':
            # Standard flow initialization (Glow): random orthogonal matrix.
            # No inductive bias — the model discovers the mapping from scratch.
            full_matrix, _ = torch.linalg.qr(torch.randn(n_mels, n_mels))
        else:
            raise ValueError(f"Unknown init: {init!r}. Use 'harmonic' or 'orthogonal'.")

        # Learnable parameter
        self.transform = nn.Parameter(full_matrix)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply coordinate transform to mel spectrogram.

        Args:
            mel: [B, 1, n_mels, T] log-mel spectrogram

        Returns:
            pitch_features: [B, T, n_mels] in pitch space
                - [:, :, :88] = piano key activations
                - [:, :, 88:] = residual spectral info
        """
        # mel: [B, 1, 128, T] -> [B, T, 128]
        x = mel.squeeze(1).permute(0, 2, 1)  # [B, T, 128]

        # Apply transform: [B, T, 128] @ [128, 128]^T -> [B, T, 128]
        return x @ self.transform.T

    def check_invertibility(self) -> dict:
        """Check invertibility diagnostics.

        Returns dict with sign, log|det|, condition number, and rank.
        The determinant itself can underflow in float32 due to many small SVs,
        so we use slogdet and SVD for robust checking.
        """
        with torch.no_grad():
            W = self.transform.double()
            sign, logabsdet = torch.linalg.slogdet(W)
            S = torch.linalg.svdvals(W)
            return {
                'sign': sign.item(),
                'logabsdet': logabsdet.item(),
                'condition_number': (S.max() / S.min()).item(),
                'rank': (S > 1e-6).sum().item(),
                'min_sv': S.min().item(),
                'max_sv': S.max().item(),
            }
