import os
import sys
import argparse
import torch
import numpy as np
from loguru import logger
import tempfile
from muzic_integration import MuzicIntegration

# Import original generation code
try:
    from generate import generate_song as original_generate_song
except ImportError:
    # Define a placeholder if the original function is not available
    def original_generate_song(prompt, lyrics=None, output_file=None, tempo=120, key="C", mode="major", style="pop"):
        logger.error("Original generate_song function not found")
        return None

def generate_song_with_muzic(prompt, output_file=None, tempo=120, key="C", mode="major", style="rap"):
    """Generate a song using Microsoft Muzic models"""
    try:
        # Initialize Muzic integration
        muzic = MuzicIntegration()
        
        # Set default output file if not provided
        if output_file is None:
            output_dir = os.getenv('OUTPUT_DIR', 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "generated_song_muzic.wav")
        
        # Step 1: Generate lyrics
        logger.info(f"Generating lyrics with prompt: {prompt}")
        lyrics, lyrics_file = muzic.generate_lyrics(prompt)
        
        # Step 2: Generate music
        logger.info(f"Generating music with tempo: {tempo}, key: {key} {mode}")
        midi_file = muzic.generate_music(tempo, f"{key} {mode}", 180)
        
        # Step 3: Convert MIDI to audio
        logger.info("Converting MIDI to audio")
        audio_file = muzic.convert_midi_to_audio(midi_file)
        
        # Step 4: For now, just use the instrumental as the final output
        # In a full implementation, you would generate vocals from lyrics and mix them
        logger.info(f"Copying audio to output file: {output_file}")
        import shutil
        shutil.copy(audio_file, output_file)
        
        # Save lyrics alongside the audio
        lyrics_output = output_file.replace(".wav", ".txt")
        with open(lyrics_output, 'w') as f:
            f.write(lyrics)
        
        logger.info(f"Song generation complete: {output_file}")
        logger.info(f"Lyrics saved to: {lyrics_output}")
        
        return output_file
    except Exception as e:
        logger.error(f"Error during Muzic song generation: {e}", exc_info=True)
        # Fall back to original generation method
        logger.info("Falling back to original generation method")
        return original_generate_song(prompt, lyrics=None, output_file=output_file, 
                                     tempo=tempo, key=key, mode=mode, style=style)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a song with Microsoft Muzic")
    parser.add_argument("--prompt", type=str, default="happy rap song in C major", help="Text prompt")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--tempo", type=int, default=120, help="Tempo in BPM")
    parser.add_argument("--key", type=str, default="C", help="Key of the song")
    parser.add_argument("--mode", type=str, default="major", help="Mode of the song")
    parser.add_argument("--style", type=str, default="rap", help="Style of the song")
    args = parser.parse_args()
    
    generate_song_with_muzic(
        prompt=args.prompt,
        output_file=args.output,
        tempo=args.tempo,
        key=args.key,
        mode=args.mode,
        style=args.style
    )
