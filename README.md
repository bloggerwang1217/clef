# Piano Harmony Harvester

An automated music transcription system that extracts singing melody and harmony parts from audio files using MusicXML reference scores.

## 🎯 Project Overview

This system processes two input files:
- **MusicXML score** (piano accompaniment)
- **MP3 audio file** (vocals + accompaniment)

And produces a new **MusicXML score** containing:
- **Singing Melody** part (main vocal line)
- **Harmony** part (background vocals)

## 🔧 Technical Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **Audio Separation** | `demucs` | Separates vocals from accompaniment using htdemucs model |
| **Audio Conversion** | `ffmpeg` | Converts MP3 to WAV format |
| **Audio Processing** | `librosa`, `soundfile` | Audio I/O and processing |
| **Music Transcription** | `Basic Pitch` | Converts vocals to MIDI |
| **Score Processing** | `music21` | MusicXML parsing and generation |

## 🚀 Installation

### Prerequisites

1. **Python 3.9 or higher**

2. **ffmpeg** (required system dependency)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/
   ```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- demucs (audio source separation)
- librosa & soundfile (audio processing)
- basic-pitch (music transcription)
- music21 (score processing)
- numpy & scipy (numerical computing)

## 📋 Usage

### Command Line

```bash
python music_transcription.py <input.musicxml> <input.mp3> [output.musicxml]
```

**Example:**
```bash
python music_transcription.py piano_score.musicxml song.mp3 output_score.musicxml
```

### Python API

```python
from music_transcription import MusicTranscriptionSystem

# Create system instance
system = MusicTranscriptionSystem()

# Process files
system.process_music(
    xml_path="piano_score.musicxml",
    mp3_path="song.mp3",
    output_path="output_score.musicxml"
)
```

## 🔄 Processing Pipeline

The system performs five stages of processing:

### Stage 1: Input & Preprocessing
- **A. MusicXML Parsing**: Extracts accompaniment notes from reference score
- **B. Audio Conversion**: Converts MP3 to mono WAV (44.1kHz)

### Stage 2: Source Separation
- Uses Demucs `htdemucs` model to separate vocals from accompaniment
- Extracts clean vocal track for transcription

### Stage 3: Automatic Music Transcription (AMT)
- Transcribes separated vocals to MIDI using Basic Pitch
- Generates note-level representation of vocal performance

### Stage 4: Post-Processing & Refinement
- **A. Quantization**: Aligns notes to 16th note grid for readability
- **B. Conflict Filtering**: Removes vocal notes that overlap >80% with accompaniment
- **C. Melody/Harmony Division**: Splits notes into melody (highest) and harmony (others)

### Stage 5: MusicXML Generation
- Creates output score with two parts: "Singing Melody" and "Harmony"
- Exports as standard MusicXML format
- Cleans up all temporary files

## 📊 Output Structure

The generated MusicXML contains:

```
Score
├── Part 1: "Singing Melody"
│   └── Main vocal line (highest notes at each time point)
└── Part 2: "Harmony"
    └── Background vocals (remaining notes)
```

## 🔍 Key Features

### Intelligent Conflict Resolution
- Compares transcribed vocals with piano accompaniment
- Filters out notes that are likely "leaked" accompaniment
- Uses 80% temporal overlap threshold with pitch matching

### Rhythm Quantization
- Quantizes all notes to 16th note grid (0.25 quarter notes)
- Ensures score readability and consistency
- Preserves musical structure while cleaning timing

### Heuristic Melody Extraction
- At each time point, assigns highest note to melody
- Remaining simultaneous notes become harmony
- Simple but effective for most vocal arrangements

## ⚙️ Configuration Options

You can customize the system by modifying parameters:

```python
system = MusicTranscriptionSystem(temp_dir="./custom_temp")

# Custom quantization (8th notes instead of 16th)
quantized_stream = system.quantize_notes(midi_path, quantize_to=0.5)

# Custom overlap threshold (90% instead of 80%)
filtered_notes = system.filter_accompaniment_conflicts(
    vocal_stream,
    accompaniment_ref,
    overlap_threshold=0.9
)
```

## 🐛 Troubleshooting

### Common Issues

1. **"ffmpeg not found"**
   - Make sure ffmpeg is installed and in your PATH
   - Test with: `ffmpeg -version`

2. **"CUDA out of memory"**
   - Demucs will automatically fall back to CPU if CUDA fails
   - For large files, ensure sufficient RAM (4GB+ recommended)

3. **"No MIDI file generated"**
   - Check that vocal separation produced valid output
   - Ensure input audio contains audible vocals

4. **Empty output score**
   - All vocal notes may have been filtered as accompaniment conflicts
   - Try lowering `overlap_threshold` parameter

## 📝 File Formats

### Input Files
- **MusicXML**: `.musicxml` or `.xml` (MusicXML 3.0 or higher recommended)
- **Audio**: `.mp3` (any bitrate, mono or stereo)

### Output Files
- **MusicXML**: Standard MusicXML 3.0 format
- Compatible with: MuseScore, Finale, Sibelius, Dorico, etc.

## 🧪 Testing

To test the system with sample files:

```bash
# Ensure you have test files in the project directory
python music_transcription.py sample_score.musicxml sample_audio.mp3 test_output.musicxml
```

## 📚 Dependencies Documentation

- [Demucs](https://github.com/facebookresearch/demucs) - Music source separation
- [Basic Pitch](https://github.com/spotify/basic-pitch) - Audio-to-MIDI transcription
- [music21](https://web.mit.edu/music21/) - Music score analysis and generation
- [librosa](https://librosa.org/) - Audio processing
- [ffmpeg](https://ffmpeg.org/) - Multimedia processing

## 🤝 Contributing

This project is part of the Piano Harmony Harvester initiative. Contributions are welcome!

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- **Demucs** by Facebook Research for state-of-the-art source separation
- **Basic Pitch** by Spotify for robust pitch detection
- **music21** by MIT for comprehensive music theory support