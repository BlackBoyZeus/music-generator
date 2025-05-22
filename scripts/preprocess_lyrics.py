import os
import re
import nltk
from nltk.corpus import cmudict
import librosa
import numpy as np
import json
import sys
import argparse

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict')

# Load CMU Pronouncing Dictionary
prondict = cmudict.dict()

def extract_beats(audio_file, tempo=None):
    """Extract beat positions from audio file"""
    y, sr = librosa.load(audio_file, sr=None)
    
    # If tempo is provided, use it to guide beat tracking
    if tempo:
        # Convert tempo from BPM to beat period in seconds
        beat_period = 60.0 / tempo
        # Use tempo as prior for beat tracking
        beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=tempo)[1]
    else:
        # Use librosa's default beat tracking
        beats = librosa.beat.beat_track(y=y, sr=sr)[1]
    
    # Convert frame indices to time
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return beat_times

def format_time(seconds):
    """Format seconds to [MM:SS.xx] format"""
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"[{minutes:02d}:{seconds:05.2f}]"

def add_beat_markers(lyrics_file, audio_file, features_file, output_file):
    """Add beat markers to lyrics based on audio beat tracking"""
    # Load lyrics
    with open(lyrics_file, 'r', encoding='utf-8') as f:
        lyrics = f.read().strip().split('\n')
    
    # Load features to get tempo if available
    tempo = None
    if os.path.exists(features_file):
        with open(features_file, 'r') as f:
            features = json.load(f)
            if 'tempo' in features:
                tempo = features['tempo']
    
    # Extract beats
    beat_times = extract_beats(audio_file, tempo)
    
    # Estimate average syllables per beat for rap (typically 1-3)
    avg_syllables_per_beat = 2
    
    # Process lyrics line by line
    beat_annotated_lyrics = []
    line_number = 0
    
    for line in lyrics:
        if not line.strip():
            continue
            
        line_number += 1
        
        # Estimate line duration based on syllable count
        # This is a simplification; actual timing would require vocal alignment
        words = line.split()
        syllable_count = sum([count_syllables(word) for word in words])
        estimated_beats_in_line = max(1, syllable_count // avg_syllables_per_beat)
        
        # Find beats that might correspond to this line
        # This is an approximation; for production, use actual vocal timing
        line_beat_indices = np.linspace(0, len(beat_times)-1, estimated_beats_in_line+2, dtype=int)[1:-1]
        
        # Insert beat markers into the line
        modified_line = line
        beat_positions = []
        
        # Simple algorithm: insert beats roughly evenly through the line
        words = line.split()
        if len(words) <= 1:
            # If only one word, put beat at the beginning
            modified_line = f"[BEAT]{modified_line}"
        else:
            # Distribute beats among words
            positions = np.linspace(0, len(words)-1, min(len(line_beat_indices), len(words)), dtype=int)
            
            for pos in sorted(positions, reverse=True):
                if pos < len(words):
                    words[pos] = f"[BEAT]{words[pos]}"
            
            modified_line = " ".join(words)
        
        # Add line number and timestamp of first beat
        if line_beat_indices.size > 0:
            first_beat_time = beat_times[line_beat_indices[0]]
            time_str = format_time(first_beat_time)
            beat_annotated_lyrics.append(f"{line_number}_{time_str}{modified_line}")
        else:
            # Fallback if no beats assigned
            beat_annotated_lyrics.append(f"{line_number}_[00:00.00]{modified_line}")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(beat_annotated_lyrics))
    
    return beat_annotated_lyrics

def count_syllables(word):
    """Estimate syllable count in a word (English)"""
    # This is a simple estimation; for production use a proper syllable counter
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def get_phonemes(word):
    """Get phonemes for a word using CMUdict"""
    word = word.lower()
    
    # Remove non-alphabetic characters
    clean_word = re.sub(r'[^a-z]', '', word)
    
    if not clean_word:
        return []
    
    # Try to get pronunciation from CMUdict
    if clean_word in prondict:
        # Get first pronunciation (most common)
        phones = prondict[clean_word][0]
        # Extract only the vowel part of each phoneme (e.g., 'AA1' -> 'AA')
        vowels = []
        for phone in phones:
            # Keep only vowel sounds
            if any(phone.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
                # Remove stress marker (number at the end)
                vowel = ''.join([c for c in phone if not c.isdigit()]).lower()
                vowels.append(vowel)
        
        # If no vowels found, return a default vowel
        if not vowels:
            return ['a']
        
        return vowels
    else:
        # For unknown words, make a simple guess based on vowels
        vowels = []
        for char in clean_word:
            if char in 'aeiou':
                vowels.append(char)
        
        # If no vowels found, return a default vowel
        if not vowels:
            return ['a']
        
        return vowels

def process_beat_lyrics(input_file, output_file):
    """Process lyrics with beat markers to create phonetic representation"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    phonetic_lines = []
    
    for line in lines:
        if not line.strip():
            continue
        
        # Extract line number and timestamp
        parts = line.split(']', 1)
        if len(parts) < 2 or '_[' not in parts[0]:
            continue
        
        prefix = parts[0] + ']'
        text = parts[1]
        
        # Process text with beat markers
        words = []
        current_word = ""
        phonetic_result = prefix
        
        # Split by spaces but preserve [BEAT] markers
        i = 0
        while i < len(text):
            if text[i:i+6] == "[BEAT]":
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append("[BEAT]")
                i += 6
            elif text[i] == ' ':
                if current_word:
                    words.append(current_word)
                    current_word = ""
                i += 1
            else:
                current_word += text[i]
                i += 1
        
        if current_word:
            words.append(current_word)
        
        # Process each word to get phonemes
        for word in words:
            if word == "[BEAT]":
                phonetic_result += " [BEAT]"
            else:
                phonemes = get_phonemes(word)
                phonetic_result += " " + " ".join(phonemes)
        
        phonetic_lines.append(phonetic_result)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(phonetic_lines))
    
    return phonetic_lines

def preprocess_lyrics(lyrics_file, audio_file, features_file, output_dir):
    """Process lyrics for DeepRapper training"""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Add beat markers
    beat_lyrics_file = os.path.join(output_dir, f"{os.path.basename(lyrics_file).split('.')[0]}_lyric_with_beat_global.txt")
    add_beat_markers(lyrics_file, audio_file, features_file, beat_lyrics_file)
    
    # Convert to phonemes
    phoneme_lyrics_file = os.path.join(output_dir, f"{os.path.basename(lyrics_file).split('.')[0]}_mapped_final_with_beat_global.txt")
    process_beat_lyrics(beat_lyrics_file, phoneme_lyrics_file)
    
    return phoneme_lyrics_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess lyrics for DeepRapper")
    parser.add_argument("--lyrics_file", type=str, required=True, help="Path to lyrics file")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--features_file", type=str, required=True, help="Path to features file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    preprocess_lyrics(args.lyrics_file, args.audio_file, args.features_file, args.output_dir)
