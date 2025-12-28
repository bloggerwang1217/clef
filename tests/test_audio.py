"""Tests for audio processing utilities."""

import numpy as np
import pytest

from src.utils.device import get_device, get_device_info


class TestDeviceDetection:
    """Test GPU/MPS/CPU device detection."""

    def test_get_device_returns_string(self):
        """Device should be one of cuda, mps, or cpu."""
        device = get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_get_device_info_returns_dict(self):
        """Device info should be a dictionary with required keys."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "selected" in info
        assert "cuda_available" in info
        assert "mps_available" in info


class TestAudioConverter:
    """Test audio format conversion."""

    def test_check_ffmpeg(self):
        """FFmpeg availability check should return boolean."""
        from src.audio.converter import check_ffmpeg

        result = check_ffmpeg()
        assert isinstance(result, bool)
