import torch
torch.set_num_threads(1)  # Limit PyTorch to 1 thread to avoid contention
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
from speechbrain.inference import EncoderClassifier  # Updated import to fix deprecation warning
import re
from audiocraft.models import get_pretrained_compression_model  # Updated import for correct model loading
from torchaudio.functional import resample

# Set device to MPS if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define directories using environment variables
A_AUDIO_DIR = os.getenv("A_AUDIO_DIR")
AUGMENTED_AUDIO_DIR = os.getenv("AUGMENTED_AUDIO_DIR")
MIDI_DIR = os.getenv("MIDI_DIR")
SPEC_DIR = os.getenv("SPEC_DIR")
LYRICS_DIR = os.getenv("LYRICS_DIR")
BASE_DIR = os.getenv("BASE_DIR")
LOG_DIR = os.getenv("LOG_DIR")

# Fallback for LOG_DIR if not set in environment
if LOG_DIR is None:
    LOG_DIR = "/Users/blblackboyzeusackboyzeus/BBZ(AiArtist)/music_generator/logs"
    os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the logs directory exists

NUM_THREADS = 4  # Reduced to avoid resource contention on M3 Pro
LOG_FILE = os.path.join(LOG_DIR, "preprocess.log")

# Configure logger
logger.add(LOG_FILE, rotation="500 MB")

# Initialize speech-to-text model
asr_model = EncoderClassifier.from_hparams(
    source="speechbrain/asr-crdnn-rnnlm-librispeech",
    savedir="pretrained_models/asr"
)
# Move ASR model to device (MPS or CPU)
asr_model = asr_model.to(device)

# Initialize EnCodec model
encodec_model = get_pretrained_compression_model('facebook/encodec_24khz')  # Updated to use correct method
encodec_model.eval()

# Move EnCodec to CPU to avoid MPS errors (as per your script)
encodec_model = encodec_model.to("cpu")

def validate_audio(audio_file):
    """Validate audio file before processing."""
    try:
        y, sr = librosa.load(audio_file, sr=None)
        if len(y) == 0:
            logger.error(f"Audio file {audio_file} is empty.")
            return False
        if sr != 24000:
            y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)
            y = resample(y_tensor, sr, 24000).cpu().numpy().squeeze()
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
    """Extract lyrics from audio using SpeechBrain ASR."""
    try:
        y, sr = librosa.load(audio_file, sr=24000, dtype=np.float32)
        if sr != 24000:
            y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)
            y = resample(y_tensor, sr, 24000).cpu().numpy().squeeze()
        signal = torch.from_numpy(y).unsqueeze(0).to(device)
        wav_lens = torch.tensor([len(y) / 24000]).to(device)  # Length in seconds
        transcription = asr_model.transcribe_batch(signal, wav_lens)[0]
        with open(output_file, "w") as f:
            f.write(transcription[0])
        logger.info(f"Extracted lyrics from {audio_file} to {output_file}")
    except Exception as e:
        logger.error(f"Failed to extract lyrics from {audio_file}: {e}")

def augment_audio(audio_file):
    """Augment audio with pitch shift and time stretch."""
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
    """Analyze MIDI file for key and mode using music21."""
    try:
        mf = converter.parse(midi_file)
        key_obj = mf.analyze("key")
        tonic = key_obj.tonic.name
        mode = key_obj.mode
        return tonic, mode
    except Exception as e:
        logger.error(f"Music theory analysis failed for {midi_file}: {e}")
        return None, None

def process_audio(audio_file):
    """Process audio file: separate stems, extract lyrics, convert to MIDI, and encode with EnCodec."""
    try:
        if not validate_audio(audio_file):
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
                    y_tensor = torch.from_numpy(y).unsqueeze(0).float().to(device)
                    y = resample(y_tensor, sr, 24000).cpu().numpy().squeeze()

                y = torch.from_numpy(y).float().unsqueeze(0)
                y = y.to("cpu")  # Move to CPU before encoding (as per your script)
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
    """Prepare a dataset of lyrics with annotations for fine-tuning."""
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
    """Main function to process audio files and prepare lyrics dataset."""
    audio_files = glob.glob(os.path.join(A_AUDIO_DIR, "*.wav")) + glob.glob(os.path.join(A_AUDIO_DIR, "*.mp3"))
    if not audio_files:
        logger.error(f"No audio files found in $A_AUDIO_DIR")
        exit(1)
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        list(executor.map(augment_audio, audio_files))
        list(executor.map(process_audio, audio_files))
    prepare_lyrics_dataset()
    logger.success("Preprocessing succeeded")

if __name__ == "__main__":
    main()