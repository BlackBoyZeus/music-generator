import os
import json
import numpy as np
import pretty_midi
from glob import glob
import sys
import argparse

def extract_features_from_json(features_file):
    """Extract relevant features from features.json"""
    with open(features_file, 'r') as f:
        features = json.load(f)
    
    tempo = features.get('tempo', 120)
    key = features.get('key', 'C major')
    time_signature = features.get('time_signature', '4/4')
    
    return tempo, key, time_signature

def parse_key(key_str):
    """Parse key string to get root note and mode"""
    key_str = key_str.strip()
    
    # Map of key names to MIDI pitch numbers (C=0, C#=1, etc.)
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    # Determine mode (major or minor)
    if 'minor' in key_str.lower():
        mode = 1  # minor
    else:
        mode = 0  # major
    
    # Extract root note
    for root in key_map:
        if key_str.startswith(root):
            return key_map[root], mode
    
    # Default to C major if parsing fails
    return 0, 0

def parse_time_signature(time_sig_str):
    """Parse time signature string to get numerator and denominator"""
    try:
        numerator, denominator = map(int, time_sig_str.split('/'))
        return numerator, denominator
    except:
        # Default to 4/4 if parsing fails
        return 4, 4

def midi_to_octuplemidi(midi_file, features_file, output_file):
    """Convert MIDI file to OctupleMIDI format"""
    # Load MIDI file
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    except Exception as e:
        print(f"Error loading MIDI file {midi_file}: {e}")
        return False
    
    # Extract features
    try:
        tempo, key_str, time_sig_str = extract_features_from_json(features_file)
    except Exception as e:
        print(f"Error loading features from {features_file}: {e}")
        # Use defaults
        tempo = 120
        key_str = 'C major'
        time_sig_str = '4/4'
    
    # Parse key and time signature
    key_root, key_mode = parse_key(key_str)
    time_sig_num, time_sig_denom = parse_time_signature(time_sig_str)
    
    # Initialize OctupleMIDI tokens
    octuple_tokens = []
    
    # Process each instrument
    for instrument in midi_data.instruments:
        # Skip drum tracks for melody/chord processing
        if instrument.is_drum:
            continue
        
        # Determine track type (0 for melody, 1 for chord)
        track_type = 1 if "chord" in midi_file.lower() else 0
        
        # Get instrument program number
        instrument_program = instrument.program
        
        # Process notes
        for note in instrument.notes:
            # Calculate bar and position
            position_in_seconds = note.start
            bar = int(position_in_seconds / (60 * time_sig_num / tempo))
            
            # Position within bar (0-127)
            bar_start_time = bar * (60 * time_sig_num / tempo)
            position_in_bar = int(((position_in_seconds - bar_start_time) / 
                                  (60 * time_sig_num / tempo)) * 128)
            
            # Ensure position is within valid range
            position_in_bar = min(127, max(0, position_in_bar))
            
            # Note pitch (0-127)
            pitch = note.pitch
            
            # Note duration (in 1/128 of a bar, capped at 127)
            duration_in_seconds = note.end - note.start
            duration_in_128th = int((duration_in_seconds / (60 * time_sig_num / tempo)) * 128)
            duration = min(127, max(1, duration_in_128th))
            
            # Create OctupleMIDI token
            token = [
                bar,                # Bar
                position_in_bar,    # Position
                pitch,              # Pitch
                duration,           # Duration
                instrument_program, # Instrument
                tempo,              # Tempo
                key_root + (key_mode * 12),  # Key (0-23)
                track_type          # Track type
            ]
            
            octuple_tokens.append(token)
    
    # Sort tokens by bar, then position
    octuple_tokens.sort(key=lambda x: (x[0], x[1]))
    
    # Convert to string format
    token_strings = []
    for token in octuple_tokens:
        token_strings.append(' '.join(map(str, token)))
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(token_strings))
    
    return True

def preprocess_midi(midi_file, features_file, output_dir):
    """Process MIDI for MusicBERT training"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to OctupleMIDI format
    output_file = os.path.join(output_dir, f"{os.path.basename(midi_file).split('.')[0]}_octuplemidi.txt")
    midi_to_octuplemidi(midi_file, features_file, output_file)
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MIDI for MusicBERT")
    parser.add_argument("--midi_file", type=str, required=True, help="Path to MIDI file")
    parser.add_argument("--features_file", type=str, required=True, help="Path to features file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    preprocess_midi(args.midi_file, args.features_file, args.output_dir)
