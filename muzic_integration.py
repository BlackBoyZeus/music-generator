import os
import sys
import logging
import subprocess
import tempfile
from loguru import logger

class MuzicIntegration:
    """Integration with Microsoft Muzic models"""
    
    def __init__(self, muzic_dir="muzic_env/muzic"):
        self.muzic_dir = muzic_dir
        self.deeprapper_dir = os.path.join(self.muzic_dir, "deeprapper")
        self.musicbert_dir = os.path.join(self.muzic_dir, "musicbert")
        self.museformer_dir = os.path.join(self.muzic_dir, "museformer")
        
        # Check if Muzic repository exists
        if not os.path.exists(self.muzic_dir):
            logger.error(f"Microsoft Muzic repository not found at {self.muzic_dir}")
            logger.error("Please run setup_muzic.sh first")
            raise FileNotFoundError(f"Microsoft Muzic repository not found at {self.muzic_dir}")
    
    def generate_lyrics(self, prompt, output_file=None):
        """Generate rap lyrics using DeepRapper"""
        if output_file is None:
            output_file = tempfile.mktemp(suffix=".txt")
        
        # Prepare command
        cmd = [
            "cd", self.deeprapper_dir, "&&",
            "python", "generate_samples.py",
            "--model_type=gpt2",
            "--model_name_or_path=checkpoints/deeprapper-model",
            "--num_samples=1",
            "--beam=5",
            "--length=512",
            "--temperature=1.0",
            "--top_k=50",
            "--top_p=0.95",
            "--reverse",
            f"--output_file={output_file}"
        ]
        
        # Add prompt conditioning if provided
        if prompt:
            cmd.append(f"--prefix={prompt}")
        
        # Run command
        logger.info(f"Generating lyrics with prompt: {prompt}")
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error generating lyrics: {result.stderr}")
            raise RuntimeError(f"Error generating lyrics: {result.stderr}")
        
        # Read generated lyrics
        with open(output_file, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        
        logger.info(f"Generated lyrics saved to {output_file}")
        return lyrics, output_file
    
    def generate_music(self, tempo=120, key="C major", duration=180, output_file=None):
        """Generate music using Museformer"""
        if output_file is None:
            output_file = tempfile.mktemp(suffix=".mid")
        
        # Prepare command
        cmd = [
            "cd", self.museformer_dir, "&&",
            "python", "generate.py",
            "--model_path=checkpoints/museformer_model.pt",
            f"--tempo={tempo}",
            f"--key={key}",
            f"--duration={duration}",
            f"--output={output_file}"
        ]
        
        # Run command
        logger.info(f"Generating music with tempo: {tempo}, key: {key}")
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error generating music: {result.stderr}")
            raise RuntimeError(f"Error generating music: {result.stderr}")
        
        logger.info(f"Generated music saved to {output_file}")
        return output_file
    
    def convert_midi_to_audio(self, midi_file, output_file=None):
        """Convert MIDI to audio using FluidSynth"""
        if output_file is None:
            output_file = midi_file.replace(".mid", ".wav")
        
        # Check if FluidSynth is installed
        try:
            subprocess.run(["fluidsynth", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FluidSynth not found. Please install FluidSynth.")
            raise RuntimeError("FluidSynth not found. Please install FluidSynth.")
        
        # Prepare command
        cmd = [
            "fluidsynth",
            "-ni",
            "soundfonts/GeneralUser_GS_v1.471.sf2",
            midi_file,
            "-F", output_file,
            "-r", "44100"
        ]
        
        # Run command
        logger.info(f"Converting MIDI to audio: {midi_file} -> {output_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error converting MIDI to audio: {result.stderr}")
            raise RuntimeError(f"Error converting MIDI to audio: {result.stderr}")
        
        logger.info(f"MIDI converted to audio: {output_file}")
        return output_file
    
    def process_lyrics_for_deeprapper(self, lyrics_file, audio_file, features_file, output_dir):
        """Process lyrics for DeepRapper training"""
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command
        cmd = [
            "cd", self.deeprapper_dir, "&&",
            "python", "preprocess_lyrics.py",
            f"--lyrics_file={lyrics_file}",
            f"--audio_file={audio_file}",
            f"--features_file={features_file}",
            f"--output_dir={output_dir}"
        ]
        
        # Run command
        logger.info(f"Processing lyrics for DeepRapper: {lyrics_file}")
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error processing lyrics: {result.stderr}")
            raise RuntimeError(f"Error processing lyrics: {result.stderr}")
        
        output_file = os.path.join(output_dir, f"{os.path.basename(lyrics_file).split('.')[0]}_processed.txt")
        logger.info(f"Lyrics processed for DeepRapper: {output_file}")
        return output_file
    
    def process_midi_for_musicbert(self, midi_file, features_file, output_dir):
        """Process MIDI for MusicBERT training"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare command
        cmd = [
            "cd", self.musicbert_dir, "&&",
            "python", "preprocess_midi.py",
            f"--midi_file={midi_file}",
            f"--features_file={features_file}",
            f"--output_dir={output_dir}"
        ]
        
        # Run command
        logger.info(f"Processing MIDI for MusicBERT: {midi_file}")
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error processing MIDI: {result.stderr}")
            raise RuntimeError(f"Error processing MIDI: {result.stderr}")
        
        output_file = os.path.join(output_dir, f"{os.path.basename(midi_file).split('.')[0]}_octuplemidi.txt")
        logger.info(f"MIDI processed for MusicBERT: {output_file}")
        return output_file
