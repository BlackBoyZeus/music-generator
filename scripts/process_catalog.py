#!/usr/bin/env python3
import os
import argparse
import glob
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_song(song_dir, output_dir):
    """Process a single song directory"""
    song_id = os.path.basename(song_dir)
    print(f"Processing {song_id}...")
    
    # Create output directories
    lyrics_output_dir = os.path.join(output_dir, "lyrics", song_id)
    midi_output_dir = os.path.join(output_dir, "midi", song_id)
    os.makedirs(lyrics_output_dir, exist_ok=True)
    os.makedirs(midi_output_dir, exist_ok=True)
    
    # Find lyrics file
    lyrics_file = None
    for path in [
        os.path.join(song_dir, "lyrics", f"{song_id}_Lyrics.txt"),
        os.path.join(song_dir, "audio", "lyrics.txt"),
        os.path.join(song_dir, "dataset", "lyrics.txt")
    ]:
        if os.path.exists(path):
            lyrics_file = path
            break
    
    # Find audio file
    audio_file = os.path.join(song_dir, "audio", f"{song_id}.wav")
    if not os.path.exists(audio_file):
        audio_file = None
        for path in glob.glob(os.path.join(song_dir, "audio", "*.wav")):
            audio_file = path
            break
    
    # Find features file
    features_file = os.path.join(song_dir, "features.json")
    if not os.path.exists(features_file):
        features_file = os.path.join(song_dir, "audio", "features.json")
        if not os.path.exists(features_file):
            # Create a default features file
            features = {
                "tempo": 120,
                "key": "C major",
                "time_signature": "4/4"
            }
            features_file = os.path.join(song_dir, "features.json")
            with open(features_file, 'w') as f:
                json.dump(features, f, indent=2)
    
    # Process lyrics if available
    if lyrics_file and audio_file:
        cmd = [
            "python", "scripts/preprocess_lyrics.py",
            f"--lyrics_file={lyrics_file}",
            f"--audio_file={audio_file}",
            f"--features_file={features_file}",
            f"--output_dir={lyrics_output_dir}"
        ]
        subprocess.run(cmd)
    
    # Process MIDI files
    for midi_file in glob.glob(os.path.join(song_dir, "midi", "*.mid")):
        cmd = [
            "python", "scripts/preprocess_midi.py",
            f"--midi_file={midi_file}",
            f"--features_file={features_file}",
            f"--output_dir={midi_output_dir}"
        ]
        subprocess.run(cmd)
    
    return song_id

def process_catalog(catalog_dir, output_dir, num_workers=4):
    """Process all songs in the catalog"""
    # Create output directories
    os.makedirs(os.path.join(output_dir, "lyrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "midi"), exist_ok=True)
    
    # Get all song directories
    song_dirs = [d for d in glob.glob(os.path.join(catalog_dir, "*")) if os.path.isdir(d)]
    
    # Process songs in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda d: process_song(d, output_dir), song_dirs), total=len(song_dirs)))
    
    # Combine all MIDI files for MusicBERT training
    all_midi_files = glob.glob(os.path.join(output_dir, "midi", "*", "*_octuplemidi.txt"))
    combined_midi_file = os.path.join(output_dir, "midi", "combined_octuplemidi.txt")
    with open(combined_midi_file, 'w') as outfile:
        for midi_file in all_midi_files:
            with open(midi_file, 'r') as infile:
                outfile.write(infile.read() + "\n")
    
    print(f"Catalog processing complete! Output in {output_dir}")
    print(f"Combined MIDI file: {combined_midi_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process music catalog for training")
    parser.add_argument("--catalog_dir", type=str, required=True, help="Path to catalog directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    process_catalog(args.catalog_dir, args.output_dir, args.num_workers)
