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
