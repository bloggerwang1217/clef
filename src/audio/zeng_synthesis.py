# Copyright 2024 Wei Zeng (National University of Singapore)
# Licensed under the Apache License, Version 2.0
# Original source: https://github.com/wei-zeng98/piano-a2s
#
# MIDI processing and audio synthesis utilities for Zeng et al. baseline reproduction.
# This code is used for apple-to-apple comparison only.

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from mido import MidiFile
from midi2audio import FluidSynth
from pedalboard import Compressor


class MIDIProcess:
    """MIDI post-processing for Zeng et al. baseline.

    Handles:
    - Cutting initial blank
    - Cutting last pedal
    - Random tempo scaling for data augmentation
    """

    def __init__(self, midi_path: str, split: str = "train"):
        self.midi = MidiFile(midi_path)
        assert split in ["train", "valid", "test"]
        self.split = split

    def cut_last_pedal(self):
        for track in self.midi.tracks:
            if (
                track[-2].type == "control_change"
                and track[-2].channel == 0
                and track[-2].control == 64
                and track[-2].value == 0
            ):
                track[-2].time = 0

    def cut_initial_blank(self):
        total_time_before_first_note = 0
        found_first_note = False

        for track in self.midi.tracks:
            time_accumulated = 0
            for msg in track:
                if not found_first_note:
                    time_accumulated += msg.time
                    if (msg.type == "note_on" and msg.velocity > 0) or (
                        msg.type == "control_change" and msg.value > 0
                    ):
                        found_first_note = True
                        total_time_before_first_note = time_accumulated - msg.time
                        msg.time = 0
                else:
                    msg.time -= total_time_before_first_note
                    break

    def random_scaling(self, range=(0.85, 1.15)):
        """Apply random tempo scaling for data augmentation.

        Note: Removed Zeng's 4-12 second length constraint, which was designed
        for 5-bar chunks. For full-song processing, we apply tempo_range directly.
        """
        original_length = self.midi.length
        lower_bound = range[0]
        upper_bound = range[1]
        if self.split == "test" or self.split == "valid":
            if lower_bound > 1:
                scaling = lower_bound
            elif upper_bound < 1:
                scaling = upper_bound
            else:
                scaling = 1
        elif self.split == "train":
            scaling = np.random.uniform(lower_bound, upper_bound)
        for track in self.midi.tracks:
            for msg in track:
                if msg.type in ["note_on", "note_off", "control_change", "program_change"]:
                    msg.time = int(msg.time * scaling)
        return scaling, original_length

    def save(self, path: str):
        try:
            self.midi.save(path)
        except Exception:
            print(f"Error in saving midi file {path}")

    def process(self, path: str, temp_path: str = "temp/temp.mid", tempo_range: tuple = (0.85, 1.15)):
        """Full processing pipeline.

        Args:
            path: Output path for processed MIDI
            temp_path: Temporary file path for intermediate processing
            tempo_range: Tuple of (min_scale, max_scale) for tempo augmentation

        Returns:
            Tuple of (scaling, original_length, success):
            - scaling: The tempo scaling factor applied (or 1.0 if failed)
            - original_length: Original MIDI length in seconds
            - success: True if tempo scaling succeeded, False if failed (negative delta time)
        """
        self.cut_last_pedal()
        self.cut_initial_blank()
        # Save to get correct length
        try:
            self.midi.save(temp_path)
            self.midi = MidiFile(temp_path)
            scaling, original_length = self.random_scaling(range=tempo_range)
            if scaling is not None:
                self.save(path)
            return scaling, original_length, True
        except ValueError as e:
            if "negative" in str(e).lower():
                # MIDI has negative delta time - can't apply tempo scaling
                # Return failure flag so caller can fallback to original MIDI with tempo=1.0
                print(f"[TEMPO SCALING FAILED] {path} - negative delta time, will use original MIDI", flush=True)
                return 1.0, 0.0, False
            raise


def render_one_midi(
    fs: FluidSynth,
    dynamic_compression: Compressor,
    midi_path: str,
    wav_path: str,
):
    """Render MIDI to WAV with loudness normalization.

    This function replicates Zeng et al.'s audio synthesis pipeline exactly
    to ensure apple-to-apple comparison.

    Args:
        fs: FluidSynth object with soundfont loaded
        dynamic_compression: Pedalboard Compressor for dynamic range control
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
    """
    try:
        fs.midi_to_audio(midi_path, wav_path)
        data, rate = sf.read(wav_path)
        meter = pyln.Meter(rate)  # Create BS.1770 meter
        if np.ndim(data) > 1:
            data = np.mean(data, axis=1)  # Convert to mono

        data_copy = pyln.normalize.peak(data, -1.0)
        attempt = 0
        while meter.integrated_loudness(data_copy) < -20:
            loudness_normalized_audio = pyln.normalize.peak(data, -1.0)
            threshold = meter.integrated_loudness(loudness_normalized_audio) + 15
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 1
                if dynamic_compression.threshold_db < threshold:
                    break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.7
                if dynamic_compression.attack_ms < 3:
                    break
            else:
                dynamic_compression.ratio += 2
                if dynamic_compression.ratio > 34:
                    break
            loudness_normalized_audio = np.array(loudness_normalized_audio)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            data_copy = pyln.normalize.peak(data_copy, -1.0)
            attempt += 1

        dynamic_compression.threshold_db = -5
        dynamic_compression.attack_ms = 10
        dynamic_compression.ratio = 1
        attempt = 0

        data = data_copy
        data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)

        while data_copy.max() > 0.9 or data_copy.min() < -0.9:
            data_copy = pyln.normalize.loudness(data, meter.integrated_loudness(data), -15)
            if attempt % 3 == 2:
                dynamic_compression.threshold_db -= 0.5
                if dynamic_compression.threshold_db < -10:
                    break
            elif attempt % 3 == 1:
                dynamic_compression.attack_ms *= 0.75
                if dynamic_compression.attack_ms < 1:
                    break
            else:
                dynamic_compression.ratio += 1.5
                if dynamic_compression.ratio > 15:
                    break
            loudness_normalized_audio = np.array(data_copy)
            data_copy = dynamic_compression(loudness_normalized_audio, rate)
            attempt += 1

        dynamic_compression.threshold_db = -1
        dynamic_compression.attack_ms = 50
        dynamic_compression.ratio = 18

        data = pyln.normalize.peak(data_copy, -1.0)
        sf.write(wav_path, data, rate)

    except ValueError:
        print(f"Error rendering: {wav_path}")
        with open("errors.txt", "a") as f:
            f.write(wav_path + "\n")


def create_default_compressor() -> Compressor:
    """Create default compressor with Zeng et al.'s settings."""
    return Compressor(threshold_db=-1, ratio=18, attack_ms=50)
