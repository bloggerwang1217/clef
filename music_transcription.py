#!/usr/bin/env python3
"""
Music Transcription System
==========================
A Python script that processes MusicXML scores and MP3 audio files to extract
singing melody and harmony parts.

This system performs the following stages:
1. Input & Preprocessing: Parse MusicXML and convert audio format
2. Source Separation: Separate vocals from accompaniment using Demucs
3. Automatic Music Transcription: Convert vocals to MIDI using Basic Pitch
4. Post-Processing: Quantize, filter accompaniment conflicts, and divide melody/harmony
5. MusicXML Generation: Output final score with singing melody and harmony parts
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Audio processing libraries
import librosa
import soundfile as sf
import numpy as np

# Music transcription and score processing
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import music21

# Demucs for source separation
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MusicTranscriptionSystem:
    """
    Main class for the music transcription system.
    Handles the complete pipeline from input to output.
    """

    def __init__(self, temp_dir: str = "./temp"):
        """
        Initialize the transcription system.

        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def __del__(self):
        """Cleanup temporary directory on object destruction."""
        self.cleanup()

    def cleanup(self):
        """Remove all temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")

    # ========================================================================
    # STAGE 1: INPUT & PREPROCESSING
    # ========================================================================

    def parse_musicxml_reference(self, xml_path: str) -> List[dict]:
        """
        Parse MusicXML file and extract non-vocal accompaniment notes.

        Args:
            xml_path: Path to input MusicXML file

        Returns:
            List of accompaniment notes with pitch, onset, and duration
        """
        logger.info("Stage 1A: Parsing MusicXML reference score...")

        try:
            score = music21.converter.parse(xml_path)
        except Exception as e:
            logger.error(f"Failed to parse MusicXML: {e}")
            raise

        accompaniment_notes = []

        # Extract all parts (assuming all are accompaniment, e.g., piano)
        for part in score.parts:
            part_name = part.partName or "Unknown"
            logger.info(f"  Extracting notes from part: {part_name}")

            # Flatten the part to get all notes with their absolute positions
            for element in part.flatten().notesAndRests:
                if isinstance(element, music21.note.Note):
                    note_data = {
                        'pitch': element.pitch.midi,  # MIDI note number
                        'onset': float(element.offset),  # Start time in quarter notes
                        'duration': float(element.quarterLength),  # Duration in quarter notes
                        'part': part_name
                    }
                    accompaniment_notes.append(note_data)
                elif isinstance(element, music21.chord.Chord):
                    # Handle chords by extracting each note
                    for pitch in element.pitches:
                        note_data = {
                            'pitch': pitch.midi,
                            'onset': float(element.offset),
                            'duration': float(element.quarterLength),
                            'part': part_name
                        }
                        accompaniment_notes.append(note_data)

        logger.info(f"  Extracted {len(accompaniment_notes)} accompaniment notes")
        return accompaniment_notes

    def convert_audio_to_wav(self, mp3_path: str) -> str:
        """
        Convert MP3 to mono WAV file using ffmpeg.

        Args:
            mp3_path: Path to input MP3 file

        Returns:
            Path to converted WAV file
        """
        logger.info("Stage 1B: Converting audio to WAV format...")

        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'],
                         capture_output=True,
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg is not installed or not in PATH. "
                             "Please install ffmpeg to continue.")

        wav_path = str(self.temp_dir / "temp_audio.wav")

        # Convert to mono, 44100 Hz WAV
        cmd = [
            'ffmpeg',
            '-i', mp3_path,
            '-ac', '1',  # Mono
            '-ar', '44100',  # 44.1 kHz sampling rate
            '-y',  # Overwrite output file
            wav_path
        ]

        try:
            subprocess.run(cmd,
                         capture_output=True,
                         check=True,
                         text=True)
            logger.info(f"  Converted audio saved to: {wav_path}")
            return wav_path
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr}")
            raise

    # ========================================================================
    # STAGE 2: SOURCE SEPARATION
    # ========================================================================

    def separate_vocals(self, wav_path: str) -> str:
        """
        Separate vocals from audio using Demucs htdemucs model.

        Args:
            wav_path: Path to input WAV file

        Returns:
            Path to separated vocals WAV file
        """
        logger.info("Stage 2: Separating vocals using Demucs...")

        # Load the htdemucs model
        logger.info("  Loading htdemucs model...")
        model = get_model('htdemucs')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # Load audio
        logger.info(f"  Loading audio from {wav_path}...")
        audio, sr = librosa.load(wav_path, sr=44100, mono=False)

        # Ensure audio is in correct shape (channels, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])  # Convert mono to stereo

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        # Apply source separation
        logger.info("  Running source separation (this may take a while)...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device=device)

        # Extract vocals (index depends on model output order)
        # htdemucs outputs: drums, bass, other, vocals
        vocals = sources[0, 3].cpu().numpy()  # Index 3 is vocals

        # Save vocals
        vocals_path = str(self.temp_dir / "vocals.wav")

        # Convert stereo to mono by averaging channels
        if vocals.shape[0] == 2:
            vocals_mono = np.mean(vocals, axis=0)
        else:
            vocals_mono = vocals[0]

        sf.write(vocals_path, vocals_mono, 44100)
        logger.info(f"  Vocals saved to: {vocals_path}")

        return vocals_path

    # ========================================================================
    # STAGE 3: AUTOMATIC MUSIC TRANSCRIPTION
    # ========================================================================

    def transcribe_vocals(self, vocals_path: str) -> str:
        """
        Transcribe vocals to MIDI using Basic Pitch.

        Args:
            vocals_path: Path to vocals WAV file

        Returns:
            Path to transcribed MIDI file
        """
        logger.info("Stage 3: Transcribing vocals to MIDI...")

        output_dir = self.temp_dir / "transcription"
        output_dir.mkdir(exist_ok=True)

        # Use Basic Pitch to transcribe
        logger.info("  Running Basic Pitch transcription...")
        predict_and_save(
            [vocals_path],
            str(output_dir),
            save_midi=True,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )

        # Find the generated MIDI file
        midi_files = list(output_dir.glob("*.mid"))
        if not midi_files:
            raise RuntimeError("Basic Pitch did not generate a MIDI file")

        midi_path = str(midi_files[0])

        # Move to standard location
        final_midi_path = str(self.temp_dir / "temp_vocal.midi")
        shutil.move(midi_path, final_midi_path)

        # Clean up transcription directory
        shutil.rmtree(output_dir)

        logger.info(f"  MIDI transcription saved to: {final_midi_path}")
        return final_midi_path

    # ========================================================================
    # STAGE 4: POST-PROCESSING & REFINEMENT
    # ========================================================================

    def quantize_notes(self, midi_path: str, quantize_to: float = 0.25) -> music21.stream.Stream:
        """
        Load MIDI and quantize note timings.

        Args:
            midi_path: Path to MIDI file
            quantize_to: Quantization unit in quarter notes (0.25 = 16th note)

        Returns:
            Quantized music21 Stream
        """
        logger.info("Stage 4A: Loading and quantizing MIDI...")

        # Load MIDI file
        midi_stream = music21.converter.parse(midi_path)

        # Flatten to get all notes
        notes = midi_stream.flatten().notesAndRests

        logger.info(f"  Quantizing to nearest {quantize_to} quarter notes (16th notes)...")

        for element in notes:
            if isinstance(element, (music21.note.Note, music21.chord.Chord)):
                # Quantize onset
                original_offset = element.offset
                quantized_offset = round(original_offset / quantize_to) * quantize_to
                element.offset = quantized_offset

                # Quantize duration
                original_duration = element.quarterLength
                quantized_duration = max(quantize_to,
                                        round(original_duration / quantize_to) * quantize_to)
                element.quarterLength = quantized_duration

        logger.info(f"  Quantized {len([n for n in notes if isinstance(n, (music21.note.Note, music21.chord.Chord))])} notes")
        return midi_stream

    def filter_accompaniment_conflicts(self,
                                      vocal_stream: music21.stream.Stream,
                                      accompaniment_ref: List[dict],
                                      overlap_threshold: float = 0.8) -> List[music21.note.Note]:
        """
        Filter out vocal notes that conflict with accompaniment.

        Args:
            vocal_stream: Quantized vocal MIDI stream
            accompaniment_ref: Reference accompaniment notes from MusicXML
            overlap_threshold: Minimum overlap ratio to consider a conflict (0.8 = 80%)

        Returns:
            List of filtered vocal notes (non-conflicting)
        """
        logger.info("Stage 4B: Filtering accompaniment conflicts...")

        vocal_notes = []
        filtered_count = 0

        # Extract all notes from vocal stream
        for element in vocal_stream.flatten().notesAndRests:
            if isinstance(element, music21.note.Note):
                vocal_notes.append(element)

        # Filter conflicts
        filtered_notes = []

        for vocal_note in vocal_notes:
            vocal_pitch = vocal_note.pitch.midi
            vocal_onset = float(vocal_note.offset)
            vocal_duration = float(vocal_note.quarterLength)
            vocal_end = vocal_onset + vocal_duration

            is_conflict = False

            # Check against all accompaniment notes
            for acc_note in accompaniment_ref:
                acc_pitch = acc_note['pitch']
                acc_onset = acc_note['onset']
                acc_duration = acc_note['duration']
                acc_end = acc_onset + acc_duration

                # Check if pitch matches
                if vocal_pitch != acc_pitch:
                    continue

                # Calculate temporal overlap
                overlap_start = max(vocal_onset, acc_onset)
                overlap_end = min(vocal_end, acc_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Calculate overlap ratio (relative to vocal note duration)
                overlap_ratio = overlap_duration / vocal_duration if vocal_duration > 0 else 0

                # If overlap exceeds threshold, it's a conflict
                if overlap_ratio >= overlap_threshold:
                    is_conflict = True
                    filtered_count += 1
                    break

            if not is_conflict:
                filtered_notes.append(vocal_note)

        logger.info(f"  Filtered {filtered_count} conflicting notes")
        logger.info(f"  Remaining vocal notes: {len(filtered_notes)}")

        return filtered_notes

    def divide_melody_harmony(self,
                             vocal_notes: List[music21.note.Note]) -> Tuple[List[music21.note.Note],
                                                                            List[music21.note.Note]]:
        """
        Divide vocal notes into melody and harmony using heuristic rule.

        Heuristic: At each time point, the highest note is melody, others are harmony.

        Args:
            vocal_notes: List of filtered vocal notes

        Returns:
            Tuple of (melody_notes, harmony_notes)
        """
        logger.info("Stage 4C: Dividing melody and harmony...")

        # Group notes by onset time
        from collections import defaultdict
        notes_by_time = defaultdict(list)

        for note in vocal_notes:
            onset = float(note.offset)
            notes_by_time[onset].append(note)

        melody_notes = []
        harmony_notes = []

        # For each time point, select highest note as melody
        for onset, notes in notes_by_time.items():
            if len(notes) == 1:
                # Single note is melody
                melody_notes.append(notes[0])
            else:
                # Multiple notes: highest is melody, rest are harmony
                sorted_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=True)
                melody_notes.append(sorted_notes[0])
                harmony_notes.extend(sorted_notes[1:])

        logger.info(f"  Melody notes: {len(melody_notes)}")
        logger.info(f"  Harmony notes: {len(harmony_notes)}")

        return melody_notes, harmony_notes

    # ========================================================================
    # STAGE 5: MUSICXML GENERATION
    # ========================================================================

    def create_output_score(self,
                           melody_notes: List[music21.note.Note],
                           harmony_notes: List[music21.note.Note],
                           output_path: str):
        """
        Create final MusicXML score with melody and harmony parts.

        Args:
            melody_notes: List of melody notes
            harmony_notes: List of harmony notes
            output_path: Path for output MusicXML file
        """
        logger.info("Stage 5: Creating output MusicXML score...")

        # Create new score
        score = music21.stream.Score()

        # Create melody part
        melody_part = music21.stream.Part()
        melody_part.partName = "Singing Melody"
        melody_part.id = "melody"

        # Add melody notes
        for note in sorted(melody_notes, key=lambda n: n.offset):
            melody_part.insert(note.offset, note)

        # Create harmony part
        harmony_part = music21.stream.Part()
        harmony_part.partName = "Harmony"
        harmony_part.id = "harmony"

        # Add harmony notes
        for note in sorted(harmony_notes, key=lambda n: n.offset):
            harmony_part.insert(note.offset, note)

        # Add parts to score
        score.insert(0, melody_part)
        score.insert(0, harmony_part)

        # Add time signature and key signature if needed
        # (Basic Pitch output might not have these)
        if not melody_part.getElementsByClass(music21.meter.TimeSignature):
            ts = music21.meter.TimeSignature('4/4')
            melody_part.insert(0, ts)
            harmony_part.insert(0, ts)

        # Write to MusicXML
        logger.info(f"  Writing MusicXML to: {output_path}")
        score.write('musicxml', fp=output_path)
        logger.info("  MusicXML score created successfully!")

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def process_music(self, xml_path: str, mp3_path: str, output_path: str = "output_score.musicxml"):
        """
        Main pipeline: Process MusicXML and MP3 to generate output score.

        Args:
            xml_path: Path to input MusicXML file
            mp3_path: Path to input MP3 audio file
            output_path: Path for output MusicXML file
        """
        try:
            logger.info("="*70)
            logger.info("MUSIC TRANSCRIPTION SYSTEM - STARTING PIPELINE")
            logger.info("="*70)

            # Stage 1: Input & Preprocessing
            accompaniment_ref = self.parse_musicxml_reference(xml_path)
            wav_path = self.convert_audio_to_wav(mp3_path)

            # Stage 2: Source Separation
            vocals_path = self.separate_vocals(wav_path)

            # Stage 3: Automatic Music Transcription
            midi_path = self.transcribe_vocals(vocals_path)

            # Stage 4: Post-Processing & Refinement
            quantized_stream = self.quantize_notes(midi_path)
            filtered_notes = self.filter_accompaniment_conflicts(quantized_stream, accompaniment_ref)
            melody_notes, harmony_notes = self.divide_melody_harmony(filtered_notes)

            # Stage 5: MusicXML Generation
            self.create_output_score(melody_notes, harmony_notes, output_path)

            logger.info("="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Output saved to: {output_path}")
            logger.info("="*70)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup temporary files
            self.cleanup()


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python music_transcription.py <input.musicxml> <input.mp3> [output.musicxml]")
        print("\nExample:")
        print("  python music_transcription.py score.musicxml audio.mp3 output.musicxml")
        sys.exit(1)

    xml_path = sys.argv[1]
    mp3_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "output_score.musicxml"

    # Create system and process
    system = MusicTranscriptionSystem()
    system.process_music(xml_path, mp3_path, output_path)


if __name__ == "__main__":
    main()
