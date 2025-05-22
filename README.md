# Music Generation System

An AI-powered music generation system that uses hierarchical models to create multi-track music from text prompts.

## Architecture

The system uses a hierarchical approach with three main components:
- Structure Generator: Creates the overall song structure
- Phrase Generator: Generates musical phrases
- EnCodec Generator: Produces the final audio output

## Key Features

- Text-to-music generation
- Multi-track output (vocals, drums, bass, other)
- Lyrics generation using Gemini API
- Audio preprocessing and feature extraction
- REST API for serving the model

## Requirements

- Python 3.8+
- PyTorch with MPS support (for Apple Silicon)
- EnCodec for audio tokenization
- FastAPI for API serving

## Setup

```bash
./setup.sh
```

## Usage

Generate music from a text prompt:

```bash
python generate.py --prompt "happy pop song in C major" --tempo 120 --key "C" --mode "major" --style "pop"
```

Start the API server:

```bash
python api.py
```

## Model Architecture

The model uses a combination of:
- Transformer attention mechanisms
- StripedHyena layers with S4 (Structured State Space) models
- EnCodec for audio tokenization and generation

## License

Copyright (c) 2025 BBZ(AiArtist)
