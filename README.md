# Music Generation System

An AI-powered music generation system that uses hierarchical models to create multi-track music from text prompts, now integrated with Microsoft Muzic for enhanced rap generation capabilities.

## Architecture

The system combines two main approaches:
1. **Hierarchical Music Generation**: Using StripedHyena layers with S4 models for general music generation
2. **Microsoft Muzic Integration**: Leveraging DeepRapper, MusicBERT, and Museformer for specialized rap generation

## Key Features

- Text-to-music generation
- Multi-track output (vocals, drums, bass, other)
- Rap lyrics generation with rhyme and rhythm
- Symbolic music understanding and generation
- REST API for serving the model

## Requirements

- Python 3.8+ for the main system
- Python 3.6 for Microsoft Muzic integration (handled by setup script)
- PyTorch with MPS support for Apple Silicon
- EnCodec for audio tokenization
- FastAPI for API serving

## Setup

For the main system:
```bash
./setup.sh
```

For Microsoft Muzic integration:
```bash
./setup_muzic.sh
```

## Usage

Generate music using the original system:
```bash
python generate.py --prompt "happy pop song in C major" --tempo 120 --key "C" --mode "major" --style "pop"
```

Generate rap music using Microsoft Muzic:
```bash
python generate_muzic.py --prompt "energetic rap about success" --tempo 95 --key "F" --mode "minor" --style "rap"
```

Start the API server:
```bash
python api_muzic.py
```

## API Endpoints

- `POST /generate_song`: Generate a new song
  - Parameters:
    - `prompt`: Text prompt for generation
    - `tempo`: Tempo in BPM
    - `key`: Musical key
    - `mode`: Musical mode (major/minor)
    - `style`: Music style
    - `use_muzic`: Whether to use Microsoft Muzic (for rap) or the original system

- `GET /songs/{job_id}`: Check generation status
- `GET /songs/{job_id}/download`: Download generated song

## Model Architecture

The system uses a combination of:
- Transformer attention mechanisms
- StripedHyena layers with S4 (Structured State Space) models
- EnCodec for audio tokenization and generation
- Microsoft Muzic models for specialized rap generation

## License

Copyright (c) 2025 BBZ(AiArtist)

## Audio Catalog Structure

The project includes real audio files from BBZ's music catalog:

```
data/
└── catalog/
    ├── Do What I Want/
    │   ├── audio/
    │   │   └── Do What I Want.mp3
    │   ├── lyrics/
    │   │   └── lyrics.txt
    │   └── features.json
    ├── Taste It/
    │   ├── audio/
    │   │   └── standardized.wav
    │   └── features.json
    └── Friended/
        ├── audio/
        │   └── Friended.wav
        └── features.json
```

Each song includes:
- Audio files (MP3/WAV)
- Lyrics (when available)
- Features metadata (tempo, key, time signature)

This structure mirrors the original catalog organization while providing a clean interface for the music generation system.
