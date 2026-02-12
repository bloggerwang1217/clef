"""
HarmonizingFlow: Physics-based Coordinate Transform
=================================================

Per-frame invertible transform from mel frequency space to pitch space.

Neuroscience analogy:
- Mel spectrogram = acoustic signal (microphone output)
- Swin = cochlea (local frequency decomposition, 2D attention)
- HarmonizingFlow = brainstem (coordinate transform: frequency -> pitch space)

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
import torch.nn.functional as F


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


class _CausalConvBlock(nn.Module):
    """Single causal conv1d block with residual connection.

    Causal: left-pad only, no future information leakage.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, C, T]. Returns: [B, C, T]."""
        residual = x
        h = F.pad(x, (self.causal_pad, 0))  # left-pad only
        h = self.conv(h)  # [B, C, T]
        h = h.transpose(1, 2)  # [B, T, C] for LayerNorm
        h = self.norm(h)
        h = F.gelu(h)
        return h.transpose(1, 2) + residual  # [B, C, T]


def _init_difference_operator(conv: nn.Conv1d) -> None:
    """Initialize Conv1d as causal temporal difference operator.

    Physics prior: onset = energy change in pitch space.
    Each output channel computes the temporal derivative of the
    corresponding input channel: out[c,t] = in[c,t] - in[c,t-2].

    For causal conv with kernel_size=3 and left-padding=2:
        kernel positions [0, 1, 2] correspond to [t-2, t-1, t].
        Setting weights to [-1, 0, +1] computes central difference.

    This gives onset spikes from epoch 0, analogous to HarmonizingFlow's
    harmonic template initialization giving pitch activations from epoch 0.
    """
    with torch.no_grad():
        nn.init.zeros_(conv.weight)   # [C_out, C_in, K]
        nn.init.zeros_(conv.bias)
        C = conv.weight.shape[0]
        for c in range(C):
            conv.weight[c, c, 0] = -1.0  # t-2 (oldest, due to causal pad)
            conv.weight[c, c, 2] = +1.0  # t   (current)


class Octopus2D(nn.Module):
    """Cross-frequency onset detector (CN octopus cells).

    Neuroscience analogy: cochlear nucleus octopus cells detect synchronous
    onset across frequency bands via thick dendritic arbors spanning many
    auditory nerve fibers.

    Architecture: 2D conv with large freq kernel x small time kernel.
    - freq_kernel=31: spans ~4 harmonics of a mid-register note
    - time_kernel=3: captures onset transient (10-30ms)
    - 32 output channels: different onset pattern detectors

    Two outputs:
    - Output A: onset-enhanced mel (residual addition) for downstream Flow
    - Output B: onset timing features (freq-pooled) for FluxAttention Level 0

    Input:  [B, 1, 128, T] mel spectrogram
    Output: (enhanced_mel [B, 1, 128, T], onset_level [B, T//pool, 32])
    """

    def __init__(
        self,
        freq_kernel: int = 31,
        time_kernel: int = 3,
        channels: int = 32,
        time_pool_stride: int = 2,
        freq_pool_stride: int = 4,  # 128 mels / 4 = 32 freq bins
    ):
        super().__init__()
        self.time_pool_stride = time_pool_stride
        self.freq_pool_stride = freq_pool_stride

        # Cross-frequency onset detector
        # Odd kernel sizes for symmetric padding (same output shape)
        self.conv = nn.Conv2d(
            1, channels,
            kernel_size=(freq_kernel, time_kernel),
            padding=(freq_kernel // 2, time_kernel // 2),
        )

        # Project back to 1 channel for residual enhancement
        self.proj_back = nn.Conv2d(channels, 1, kernel_size=1)

        # Residual scale: start small to preserve mel for Flow
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, mel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect cross-frequency onsets and enhance mel.

        Args:
            mel: [B, 1, 128, T] log-mel spectrogram

        Returns:
            enhanced_mel: [B, 1, 128, T] mel + onset residual (for Flow)
            onset_level: [B, H*W, C] onset features with 2D spatial info (for Level 0)
                         H = 128 // freq_pool_stride (default 32)
                         W = T // time_pool_stride
        """
        # Cross-frequency onset detection
        onset = F.gelu(self.conv(mel))  # [B, C, 128, T]

        # Output A: residual enhancement for downstream Flow
        residual = self.proj_back(onset)  # [B, 1, 128, T]
        enhanced = mel + self.scale * residual

        # Output B: onset features with 2D spatial structure for FluxAttention Level 0
        # Preserve frequency axis (onset SHAPE across freq bands = timbre/pitch cue)
        # Pool to [B, C, H, W] where H=32 (freq), W=T//stride (time)
        onset_pooled = F.avg_pool2d(
            onset,
            kernel_size=(self.freq_pool_stride, self.time_pool_stride),
            stride=(self.freq_pool_stride, self.time_pool_stride),
        )  # [B, C, H, W]
        B, C, H, W = onset_pooled.shape
        # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
        onset_level = onset_pooled.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return enhanced, onset_level


class TemporalCNN(nn.Module):
    """Causal 1D CNN for temporal dynamics in pitch space.

    Neuroscience analogy: inferior colliculus temporal modulation filtering.
    Detects onset/sustain/release patterns in pitch activation timelines.

    Architecture: 2-layer causal dilated conv with residual connections.
    - Layer 1: difference operator init → onset/release from epoch 0
    - Layer 2: random init → learns higher-order temporal patterns

    Physics initialization (layer 1):
        kernel = [-1, 0, +1] per channel (causal central difference)
        → positive spike at onset (energy appeared)
        → negative spike at release (energy disappeared)
        → near zero during sustain (energy stable)
        Equivalent to spectral flux onset detection in pitch space.

    Input:  [B, C, T] pitch-space features (detached from HarmonizingFlow)
    Output: [B, C, T] with temporal context baked in
    """

    def __init__(self, channels: int = 128, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CausalConvBlock(channels, kernel_size, dilation=1),  # RF: 3 frames
            _CausalConvBlock(channels, kernel_size, dilation=2),  # RF: 7 frames
        ])
        # Layer 1: difference operator (onset detection from epoch 0)
        _init_difference_operator(self.blocks[0].conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, C, T]. Returns: [B, C, T]."""
        for block in self.blocks:
            x = block(x)
        return x


class HarmonizingFlow(nn.Module):
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
        use_temporal_cnn: bool = False,
        temporal_pool_stride: int = 8,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_pitches = n_pitches
        self.use_temporal_cnn = use_temporal_cnn
        self.temporal_pool_stride = temporal_pool_stride

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

        # Temporal CNN: onset/sustain/release detection in pitch space
        # Reads Flow output but gradient does NOT flow back to self.transform
        if use_temporal_cnn:
            self.temporal_cnn = TemporalCNN(n_mels, kernel_size=3)

    def forward(self, mel: torch.Tensor):
        """Apply coordinate transform to mel spectrogram.

        Args:
            mel: [B, 1, n_mels, T] log-mel spectrogram

        Returns:
            If use_temporal_cnn=False:
                pitch_features: [B, T, n_mels] in pitch space
            If use_temporal_cnn=True:
                (pitch_features, temporal_features):
                    pitch_features:    [B, T, n_mels] pitch space (full resolution)
                    temporal_features: [B, T//pool_stride, n_mels] with temporal context
        """
        # mel: [B, 1, 128, T] -> [B, T, 128]
        x = mel.squeeze(1).permute(0, 2, 1)  # [B, T, 128]

        # Apply transform: [B, T, 128] @ [128, 128]^T -> [B, T, 128]
        pitch = x @ self.transform.T

        if not self.use_temporal_cnn:
            return pitch

        # Temporal CNN: detached from transform (observer, not operator)
        # CNN reads pitch-space activations but gradient does NOT flow back
        # to self.transform — same principle as reference_refine detach.
        t = pitch.detach().transpose(1, 2)  # [B, 128, T]
        t = self.temporal_cnn(t)  # [B, 128, T]
        t = F.avg_pool1d(t, kernel_size=self.temporal_pool_stride,
                         stride=self.temporal_pool_stride)
        temporal = t.transpose(1, 2)  # [B, T//stride, 128]

        return pitch, temporal

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
