#!/bin/bash

# setup.sh - Enhanced setup for music generation with MusicGen/EnCodec,
#                 text conditioning, and lyrics on macOS M3 Pro
# Date: March 12, 2025
# Purpose: Automate setup for a hierarchical music generation model with lyrics and
#          full song synthesis using PyTorch, including MusicGen/EnCodec integration

# Exit on any error
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Directories
BASE_DIR="/Users/blblackboyzeusackboyzeus/BBZ(AiArtist)/music_generator"
A_AUDIO_DIR="$BASE_DIR/A_AUDIO"
AUGMENTED_AUDIO_DIR="$BASE_DIR/augmented_AUDIO"
MIDI_DIR="$BASE_DIR/MIDI"
SPEC_DIR="$BASE_DIR/spectrograms"
LYRICS_DIR="$BASE_DIR/lyrics"
MODEL_DIR="$BASE_DIR/models"
CHECKPOINTS_DIR="$BASE_DIR/checkpoints"
OUTPUT_DIR="$BASE_DIR/output"
LOG_DIR="$BASE_DIR/logs"

# Export directories
export BASE_DIR
export A_AUDIO_DIR
export AUGMENTED_AUDIO_DIR
export MIDI_DIR
export SPEC_DIR
export LYRICS_DIR
export MODEL_DIR
export CHECKPOINTS_DIR
export OUTPUT_DIR
export LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR"

# Logging and status functions
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

check_status() {
  if [ $? -ne 0 ]; then
    log "${RED}Error: $1 failed${NC}"
    exit 1
  else
    log "${GREEN}$1 succeeded${NC}"
  fi
}

log "Starting enhanced setup for music generation with MusicGen/EnCodec, text conditioning, and lyrics on macOS M3 Pro (PyTorch)..."

# Step 1: Install Homebrew
if ! command -v brew &>/dev/null; then
  log "${YELLOW}Installing Homebrew...${NC}"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  check_status "Homebrew installation"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
else
  log "${GREEN}Homebrew already installed.${NC}"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
HOMEBREW_PREFIX=$(brew --prefix)

# Step 2: Install Homebrew dependencies
log "Installing Homebrew dependencies..."
brew install python@3.9 fluidsynth ffmpeg
check_status "Homebrew dependencies installation"

# Step 3: Ensure project directories
log "Ensuring project directories exist..."
mkdir -p "$A_AUDIO_DIR" "$AUGMENTED_AUDIO_DIR" "$MIDI_DIR" "$SPEC_DIR" "$LYRICS_DIR" "$MODEL_DIR" "$CHECKPOINTS_DIR" "$OUTPUT_DIR" "$LOG_DIR"
check_status "Directory creation"

# Step 4: Set up virtual environment
PYTHON_COMMAND="${HOMEBREW_PREFIX}/opt/python@3.9/bin/python3.9"
log "Setting up virtual environment with $PYTHON_COMMAND..."
$PYTHON_COMMAND -m venv "$BASE_DIR/ai_music_env"
check_status "Virtual environment creation"

# Activate virtual environment
log "Activating virtual environment..."
source "$BASE_DIR/ai_music_env/bin/activate"
check_status "Virtual environment activation"

# Step 5: Install Python dependencies with PyTorch and AudioCraft support
log "Installing Python dependencies (M3 GPU mode)..."
pip install --upgrade pip

cat <<EOF > "$BASE_DIR/requirements.txt"
torch==2.1.0
torchaudio==2.1.0
librosa==0.10.1
numpy==1.23.5
soundfile==0.12.1
fastapi==0.111.0
uvicorn==0.30.1
demucs==4.0.1
basic-pitch==0.3.1
pyjwt==2.8.0
loguru==0.7.2
matplotlib==3.9.0
mido==1.3.3
music21==8.3.0
python-dotenv==1.0.1
sentencepiece==0.2.0
transformers==4.38.2
speechbrain==1.0.1
google-generativeai==0.8.3
google-ai-generativelanguage==0.6.10
einops==0.8.0
audiocraft==1.3.0
protobuf==3.20.3
scikit-learn==1.3.2
EOF

pip install -r "$BASE_DIR/requirements.txt" --no-cache-dir
check_status "Dependency installation"

# Step 6: Verify AudioCraft installation
log "Verifying AudioCraft installation..."
python -c "import audiocraft; print('AudioCraft version:', audiocraft.__version__)" >> "$LOG_FILE" 2>&1
check_status "AudioCraft installation check"

# Step 7: Verify PyTorch GPU (MPS) support
log "Verifying PyTorch MPS support..."
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())" >> "$LOG_FILE" 2>&1
check_status "PyTorch MPS check"

# Step 8: Download soundfont (might not be needed anymore, kept for backward compatibility)
log "Downloading FluidSynth soundfont..."
SOUNDFONT_URL="https://github.com/FluidSynth/fluidsynth/raw/master/sf2/GeneralUser_GS_v1.471.sf2"
SOUNDFONT_PATH="$BASE_DIR/GeneralUser_GS_v1.471.sf2"
curl -L "$SOUNDFONT_URL" -o "$SOUNDFONT_PATH" --progress-bar
check_status "Soundfont download"

# Step 9: Create preprocessing script
log "Creating preprocessing script..."
cat << 'EOF' > "$BASE_DIR/preprocess.py"
import torch
torch.set_num_threads(1)
import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from demucs import separate
from basic_pitch.inference import predict
import librosa
import numpy as np
from loguru import logger
import mido
from music21 import converter, key
from speechbrain.pretrained import EncoderClassifier
import re
from audiocraft.models import EncodecModel
from torchaudio.functional import resample

A_AUDIO_DIR = os.getenv("A_AUDIO_DIR")
AUGMENTED_AUDIO_DIR = os.getenv("AUGMENTED_AUDIO_DIR")
MIDI_DIR = os.getenv("MIDI_DIR")
SPEC_DIR = os.getenv("SPEC_DIR")
LYRICS_DIR = os.getenv("LYRICS_DIR")
BASE_DIR = os.getenv("BASE_DIR")
LOG_DIR = os.getenv("LOG_DIR")
NUM_THREADS = 4  # Reduced to avoid resource contention on M3 Pro
LOG_FILE = os.path.join(LOG_DIR, "preprocess.log")

logger.add(LOG_FILE, rotation="500 MB")

# Initialize speech-to-text model
asr_model = EncoderClassifier.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr")

# Initialize EnCodec model
encodec_model = EncodecModel.from_pretrain('facebook/encodec_24khz')
encodec_model.eval()

# Move EnCodec to CPU to avoid MPS errors
encodec_model.to("cpu")

def validate_AUDIO(audio_file):
    """Validate audio file before processing."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) == 0:
            logger.error(f"Audio file {audio_file} is empty.")
            return False
        if sr != 24000:
            y = resample(torch.from_numpy(y).unsqueeze(0).float(), sr, 24000).numpy().squeeze()
        librosa.output.write_wav(audio_file, y, 24000)
        return True
    except Exception as e:
        logger.error(f"Failed to validate audio file {audio_file}: {e}")
        return False

def detect_rhyme_scheme(lines):
    """Simple rhyme scheme detection based on last words."""
    if not lines:
        return "Unknown"
    last_words = []
    for line in lines:
        words = line.strip().split()
        last_words.append(words[-1].lower() if words else "")
    rhyme_letters = []
    rhyme_dict = {}
    current_letter = "A"
    for word in last_words:
        found_rhyme = False
        for key, val in rhyme_dict.items():
            if word and key.endswith(word[-2:]):  # Simple rhyme check
                rhyme_letters.append(val)
                found_rhyme = True
                break
        if not found_rhyme and word:
            rhyme_dict[word] = current_letter
            rhyme_letters.append(current_letter)
            current_letter = chr(ord(current_letter) + 1)
    return "".join(rhyme_letters) if rhyme_letters else "Unknown"

def annotate_lyrics(lyrics):
    """Annotate lyrics with section (Verse/Chorus) and rhyme scheme."""
    lines = lyrics.strip().split("\n")
    if not lines:
        return "Unknown", "Unknown", lines
    # Simple heuristic: Assume repeated lines indicate a chorus
    line_counts = {}
    for line in lines:
        line = line.strip()
        if line:
            line_counts[line] = line_counts.get(line, 0) + 1
    # If more than 2 lines are repeated, likely a chorus
    repeated_lines = sum(1 for count in line_counts.values() if count > 1)
    section = "Chorus" if repeated_lines >= 2 else "Verse"
    rhyme_scheme = detect_rhyme_scheme(lines)
    return section, rhyme_scheme, lines

def extract_lyrics(audio_file, output_file):
    try:
        y, sr = librosa.load(audio_file, sr=24000, dtype=np.float32)
        if sr != 24000:
            y = resample(torch.from_numpy(y).unsqueeze(0).float(), sr, 24000).numpy().squeeze()
        signal = torch.from_numpy(y).unsqueeze(0)
        transcription = asr_model.transcribe_batch(signal, torch.tensor([len(y)]))[0]
        with open(output_file, "w") as f:
            f.write(transcription[0])
        logger.info(f"Extracted lyrics from {audio_file} to {output_file}")
    except Exception as e:
        logger.error(f"Failed to extract lyrics from {audio_file}: {e}")

def augment_AUDIO(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=24000, dtype=np.float32)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        y_stretched = librosa.effects.time_stretch(y, rate=0.9)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        np.save(os.path.join(AUGMENTED_AUDIO_DIR, f"{base_name}_shifted.npy"), y_shifted)
        np.save(os.path.join(AUGMENTED_AUDIO_DIR, f"{base_name}_stretched.npy"), y_stretched)
    except Exception as e:
        logger.error(f"Augmentation failed for {audio_file}: {e}")

def analyze_music_theory(midi_file):
    try:
        mf = converter.parse(midi_file)
        key_obj = mf.analyze("key")
        tonic = key_obj.tonic.name
        mode = key_obj.mode
        return tonic, mode
    except Exception as e:
        logger.error(f"Music theory analysis failed for {midi_file}: {e}")
        return None, None

def process_AUDIO(audio_file):
    try:
        if not validate_AUDIO(audio_file):
            logger.error(f"Skipping {audio_file} due to validation failure.")
            return

        song_name = os.path.splitext(os.path.basename(audio_file))[0]
        stem_dir = os.path.join(MIDI_DIR, song_name)
        spec_stem_dir = os.path.join(SPEC_DIR, song_name)
        lyrics_file = os.path.join(LYRICS_DIR, f"{song_name}_vocals.txt")
        os.makedirs(stem_dir, exist_ok=True)
        os.makedirs(spec_stem_dir, exist_ok=True)

        # Run demucs separation
        logger.info(f"Running demucs separation for {audio_file}...")
        separate.main(["-o", stem_dir, "-n", "htdemucs", audio_file])

        stem_subdir = next((d for d in glob.glob(os.path.join(stem_dir, "*")) if os.path.isdir(d)), None)
        if not stem_subdir:
            logger.error(f"No demucs output for {audio_file}")
            return

        for stem in ["vocals", "drums", "bass", "other"]:
            stem_file = os.path.join(stem_subdir, f"{stem}.wav")
            if os.path.exists(stem_file):
                if stem == "vocals":
                    logger.info(f"Extracting lyrics from {stem_file}...")
                    extract_lyrics(stem_file, lyrics_file)

                midi_file = os.path.join(MIDI_DIR, f"{song_name}_{stem}.mid")
                midi_data, _, _ = predict(stem_file)
                midi_data.write(midi_file)

                tonic, mode = analyze_music_theory(midi_file)
                if tonic and mode:
                    logger.info(f"Detected key for {stem}: {tonic} {mode}")

                # Load the stem audio
                y, sr = librosa.load(stem_file, sr=24000, dtype=np.float32)
                if sr != 24000:
                    y = resample(torch.from_numpy(y).unsqueeze(0).float(), sr, 24000).numpy().squeeze()

                y = torch.from_numpy(y).float().unsqueeze(0)
                y = y.to("cpu")  # Move to CPU before encoding
                with torch.no_grad():
                    encoded = encodec_model.encode(y)

                # Save EnCodec encodings
                spec_file = os.path.join(spec_stem_dir, f"{stem}_encodec.pt")
                torch.save(encoded, spec_file)
                logger.info(f"Encoded audio with EnCodec to {spec_file}")
            else:
                logger.warning(f"Stem file {stem_file} not found after demucs separation.")
    except Exception as e:
        logger.error(f"Failed to process {audio_file}: {e}")

def prepare_lyrics_dataset():
    lyrics_files = glob.glob(os.path.join(LYRICS_DIR, "*.txt"))
    if not lyrics_files:
        logger.error(f"No lyrics files found in {LYRICS_DIR}")
        exit(1)
    dataset = []
    for lyrics_file in lyrics_files:
        with open(lyrics_file, "r") as f:
            lyrics = f.read().strip()
        section, rhyme_scheme, lines = annotate_lyrics(lyrics)
        # Format for Gemini fine-tuning with system/user/model messages
        formatted_lyrics = f"[Section: {section}, Rhyme: {rhyme_scheme}]\n" + "\n".join(lines)
        dataset.append({
            "messages": [
                {"role": "system", "content": "Generate lyrics for a song with structural annotations."},
                {"role": "user", "content": f"Write a {section.lower()} for a pop song with a {rhyme_scheme} rhyme scheme."},
                {"role": "model", "content": formatted_lyrics}
            ]
        })
    dataset_path = os.path.join(BASE_DIR, "lyrics_dataset.jsonl")
    with open(dataset_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Lyrics dataset prepared at {dataset_path}")

def main():
    audio_files = glob.glob(os.path.join(A_AUDIO_DIR, "*.wav")) + glob.glob(os.path.join(A_AUDIO_DIR, "*.mp3"))
    if not audio_files:
        logger.error(f"No audio files found in $A_AUDIO_DIR")
        exit(1)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        list(executor.map(augment_AUDIO, audio_files))
        list(executor.map(process_AUDIO, audio_files))
    prepare_lyrics_dataset()
    logger.success("Preprocessing succeeded")

if __name__ == "__main__":
    main()
EOF
check_status "Preprocessing script creation"

    # Step 11: Create generation script
    log "Creating generation script..."
    cat << 'EOF' > "$BASE_DIR/generate.py"
import torch
import numpy as np
import os
import librosa
from loguru import logger
import argparse
from model import HierarchicalMusicGenerator, encode_text_prompt #This may require changes with S4 and StripesHyena Gone!
from lyrics_generator import generate_lyrics
from speechbrain.pretrained import HIFIGAN
import soundfile as sf
from audiocraft.models import EncodecModel
from audiocraft.utils.notebook import display_AUDIO
from transformers import AutoTokenizer, AutoModel
import mido
from torchaudio.functional import resample

LOG_DIR = os.getenv('LOG_DIR')
LOG_FILE = os.path.join(LOG_DIR, "generate.log")
logger.add(LOG_FILE, rotation="500 MB")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# VOCODER = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/hifigan")

# Initialize EnCodec model
encodec_model = EncodecModel.from_pretrain('facebook/encodec_24khz').to(device)
encodec_model.eval()
encodec_model.to("cpu")  # Load Encoder to CPU because it seems to work better for the system!

# Load tokenizer for text embedding
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

def generate_song(prompt, lyrics=None, output_file=os.path.join(os.getenv('OUTPUT_DIR'), "generated_song.wav"), tempo=120, key="C", mode="major", style="pop"):
    try:
        model = HierarchicalMusicGenerator().to(device) #This will likely require big changes based on your implementations
        model.load_state_dict(torch.load(os.path.join(os.getenv('MODEL_DIR'), "music_generator.pt")))
        model.eval()

        # Encode text prompt
        text_data = encode_text_prompt(prompt)
        text_tensor = torch.tensor(text_data, dtype=torch.float32).to(device)

        # Initialize input tensors
        num_tracks = 4  # Vocals, drums, bass, other
        sequence_length = 500  # EnCodec sequence length (5 seconds)
        encodec_dim = 60  # EnCodec dimension
        phrase_input = torch.zeros((1, sequence_length, encodec_dim * num_tracks), dtype=torch.float32).to(device)
        encodec_input = torch.zeros((1, sequence_length, encodec_dim * num_tracks), dtype=torch.float32).to(device)

        with torch.no_grad():
            structure, phrase_output, encodec_output = model(text_tensor, phrase_input, encodec_input)

        structure_labels = ["Intro", "Verse1", "Chorus1", "Verse2", "Chorus2", "Bridge", "Outro"]
        sections = torch.argmax(structure, dim=-1).cpu().numpy().tolist()

        # Generate lyrics if not provided
        if lyrics is None:
            lyrics_prompt = f"Write a {style} song in {key} {mode} about {prompt}"
            lyrics = generate_lyrics(lyrics_prompt)
            if not lyrics:
                logger.error("Failed to generate lyrics")
                return
            logger.info(f"Generated lyrics:\n{lyrics}")
        else:
            logger.info("Using provided lyrics.")

        lyrics_lines = lyrics.split("\n")
        lyrics_per_section = len(lyrics_lines) // len(sections) if sections else len(lyrics_lines)
        section_lyrics = [lyrics_lines[i:i + lyrics_per_section] for i in range(0, len(lyrics_lines), lyrics_per_section)]

        tracks = ["vocals", "drums", "bass", "other"]

        # Generate Encodec output
        generated_AUDIO = {}
        for idx, track in enumerate(tracks):
            track_encodec_output = encodec_output[:, :, idx*encodec_dim:(idx+1)*encodec_dim]
            # Convert output to proper format for EnCodec decoding
            track_encodec_output = track_encodec_output.permute(0, 2, 1)  # Swap dimensions
            track_encodec_output = track_encodec_output.unsqueeze(0)

            with torch.no_grad():
                track_encodec_output = track_encodec_output.to("cpu")  # Move to CPU for running
                decoded_AUDIO = encodec_model.decode(track_encodec_output)

            # Resample to 16kHz for output
            decoded_AUDIO_resampled = resample(decoded_AUDIO, 24000, 16000)

            # Store the generated audio for each track
            generated_AUDIO[track] = decoded_AUDIO_resampled.cpu().numpy().squeeze()

        # Mixing and output
        mixed_AUDIO = (
            generated_AUDIO["vocals"] * 0.4 +
            generated_AUDIO["drums"] * 0.2 +
            generated_AUDIO["bass"] * 0.2 +
            generated_AUDIO["other"] * 0.2
        )
        sf.write(output_file, mixed_AUDIO, 16000)  # Important: write at 16kHz
        logger.info(f"Generated full song saved to {output_file}")

    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a song with a text prompt")
    parser.add_argument("--prompt", type=str, default="happy pop song in C major", help="Text prompt")
    parser.add_argument("--lyrics", type=str, default=None, help="Optional lyrics to use")
    parser.add_argument("--tempo", type=int, default=120, help="Tempo in BPM")
    parser.add_argument("--key", type=str, default="C", help="Key of the song")
    parser.add_argument("--mode", type=str, default="major", help="Mode of the song")
    parser.add_argument("--style", type=str, default="pop", help="Style of the song")
    args = parser.parse_args()
    generate_song(prompt=args.prompt, lyrics=args.lyrics, tempo=args.tempo, key=args.key, mode=args.mode, style=args.style)
EOF
check_status "Generation script creation"

    # Step 12: Fix TensorFlow dependency conflicts
    log "Attempting to resolve TensorFlow dependency conflicts..."
    pip install "keras==2.15.0" "ml-dtypes==0.2.0" "tensorboard==2.15.2" || log "${YELLOW}Warning: Could not fix TensorFlow dependencies. May cause errors later.${NC}"

    # Step 13: Explicitly reinstall demucs
    log "Explicitly reinstalling demucs..."
    pip install demucs --no-cache-dir || log "${YELLOW}Warning: demucs reinstallation failed, may cause errors later.${NC}"

    # Step 14: Check for audio files and GEMINI_API_KEY
    log "Checking for audio files in $A_AUDIO_DIR..."
    if [ -z "$(ls -A "$A_AUDIO_DIR")" ]; then
        log "${YELLOW}Warning: No audio files found in $A_AUDIO_DIR. Add WAV or MP3 files with vocals before preprocessing.${NC}"
    else
        log "${GREEN}Audio files detected in $A_AUDIO_DIR${NC}"
    fi

    log "Checking for GEMINI_API_KEY..."
    if [ -z "$GEMINI_API_KEY" ]; then
        log "${YELLOW}Warning: GEMINI_API_KEY not set. Set it with 'export GEMINI_API_KEY=your-key' for lyrics generation.${NC}"
    else
        log "${GREEN}GEMINI_API_KEY is set${NC}"
    fi

    # Step 15: Run preprocessing
    log "Running preprocessing on audio files..."
    python "$BASE_DIR/preprocess.py" >> "$LOG_DIR/preprocess.log" 2>&1 || {
        log "${RED}Preprocessing failed. Check $LOG_DIR/preprocess.log for details.${NC}"
        exit 1
    }
    log "${GREEN}Preprocessing succeeded${NC}"

    # Step 16: Fine-tune lyrics model (PaLM API does not support fine-tuning)
    pip install --force-reinstall google-generativeai==0.8.3
    log "Fine-tuning lyrics model..."
    python "$BASE_DIR/lyrics_generator.py" >> "$LOG_DIR/lyrics_finetune.log" 2>&1 || {
        log "${RED}Lyrics model training failed. Check $LOG_DIR/lyrics_finetune.log for details.${NC}"
        exit 1
    }
    log "${GREEN}Lyrics model training succeeded${NC}"

    # Step 17: Train hierarchical music model (Adapt this to MusicGen)
    log "Training hierarchical music model..."
    python "$BASE_DIR/model.py" >> "$LOG_DIR/model_training.log" 2>&1 || {
        log "${RED}Model training failed. Check $LOG_DIR/model_training.log for details.${NC}"
        exit 1
    }
    log "${GREEN}Model training succeeded${NC}"

    # Step 18: Test generation
    log "Testing song generation..."
    python "$BASE_DIR/generate.py" --prompt "happy pop song in C major" --tempo 120 --key "C" --mode "major" --style "pop" >> "$LOG_DIR/generate_test.log" 2>&1 || {
        log "${RED}Generation test failed. Check $LOG_DIR/generate_test.log for details.${NC}"
        exit 1
    }
    log "${GREEN}Generation test succeeded${NC}"

    # Step 19: Start API (Commented out for now, uncomment when ready to use)
    # log "Starting API..."
    # echo "Generate a JWT token for API access:"
    # echo "python3 -c \"import jwt; print(jwt.encode({'user': 'artist'}, '$
    # echo "Then use the token to call the API, e.g.:"
    # echo "curl -H \"Authorization: Bearer <token>\" \"http://localhost:8000/generate_song?prompt=happy%20pop%20song&tempo=120&key=C&mode=major&style=pop\""
    # uvicorn api:app --host 0.0.0.0 --port 8000 &
    # check_status "API startup"

    log "${GREEN}Setup completed successfully! Activate the virtual environment with 'source $BASE_DIR/ai_music_env/bin/activate' for future runs.${NC}"