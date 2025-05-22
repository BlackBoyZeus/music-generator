import os
# For Apple Silicon, we’ll see if MPS is available
import torch
USE_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
device = torch.device("mps" if USE_MPS else "cpu")

os.environ["LIBROSA_CACHE_LEVEL"] = "0"
import sys
import json
import shutil
import logging
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wave
from vosk import Model as VoskModel, KaldiRecognizer
from pydub import AudioSegment
import torchaudio
from mutagen import File
from sklearn.cluster import KMeans
from scipy.signal.windows import hann
from scipy.stats import pearsonr
import whisper

# ------------------ Krumhansl-Schmuckler Key Profiles ------------------ #
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# ------------------ Command Line Arguments ------------------ #
# Command Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Music Processing Tool")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing song folders")
    parser.add_argument("--vosk-model", type=str, default="", help="Path to Vosk model directory (optional if using Whisper)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()

# Logging Setup (unchanged from the first script)
def setup_logging(log_level: str):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    fh = logging.FileHandler(log_dir / "music_processing.log", mode='w')
    fh.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ------------------ Audio Loading & Standardization ------------------ #
def load_audio(audio_path: str, logger) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Loaded audio: {audio_path} (Sample Rate: {sr})")
        return y, sr
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return None, None

def standardize_audio(audio_file, output_path, target_sr, logger):
    """
    Converts audio_file to WAV at target_sr, 1 channel, normalized volume.
    Skips if output_path exists.
    """
    if os.path.exists(output_path):
        logger.info(f"[SKIP] Already standardized: {output_path}")
        return output_path

    try:
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_frame_rate(target_sr).set_channels(1).normalize()
        audio.export(output_path, format="wav")
        logger.info(f"Audio standardized to {target_sr}Hz => {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Audio standardization failed for {audio_file}: {e}")
        return None

# ------------------ Vocal Isolation (Demucs) ------------------ #
def isolate_vocals(audio_path, logger):
    """
    Uses Demucs (htdemucs) to isolate vocals.  
    Moves model & waveform to MPS if available.
    Skips if separated vocals already exist.
    """
    try:
        from demucs.separate import Separator
        output_dir = Path(audio_path).parent / "separated"
        output_dir.mkdir(exist_ok=True)
        base_name = Path(audio_path).stem
        vocal_path = output_dir / f"{base_name}_vocals.wav"

        # Skip if already isolated
        if vocal_path.exists():
            logger.info(f"[SKIP] Vocals already isolated: {vocal_path}")
            return str(vocal_path)

        separator = Separator(model='htdemucs')
        if USE_MPS:
            separator.to(device)

        waveform, sample_rate = torchaudio.load(audio_path)
        if USE_MPS:
            waveform = waveform.to(device)

        sources = separator.separate(waveform, sample_rate)
        vocals = sources['vocals']
        if vocals.shape[0] > 1:
            vocals = vocals.mean(dim=0, keepdim=True)

        # Move back to CPU for saving
        vocals = vocals.cpu()
        torchaudio.save(str(vocal_path), vocals, sample_rate)
        logger.info(f"Vocals isolated => {vocal_path}")
        return str(vocal_path)
    except ImportError as e:
        logger.error(f"Demucs import failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Vocal isolation failed for {audio_path}: {e}")
        return None

# ------------------ Transcription (Whisper w/ Fallback to Vosk) ------------------ #
def transcribe_lyrics_whisper(audio_path, logger):
    """
    Transcribes using Whisper (medium) on MPS if available.
    """
    try:
        # Load or reuse a global model
        # For demonstration, load each time. Ideally load once at the top-level for speed.
        w_model = whisper.load_model("medium", device=device)
        result = w_model.transcribe(audio_path)
        lyrics = result["text"].strip()
        logger.info(f"[OK] Whisper transcribed => {audio_path}")
        return lyrics if lyrics else "No lyrics detected"
    except Exception as e:
        logger.error(f"Whisper transcription failed for {audio_path}: {e}")
        return None

def transcribe_lyrics_vosk(audio_path, vosk_model, logger):
    """
    Transcribes using Vosk. Expects somewhat isolated vocals.
    """
    if not vosk_model:
        logger.error("[FAIL] No Vosk model loaded; cannot transcribe.")
        return "No lyrics detected"

    try:
        # Attempt vocal isolation first
        vocal_path = isolate_vocals(audio_path, logger)
        if not vocal_path:
            return "Vocal isolation failed"

        wf = wave.open(vocal_path, "rb")
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)
        transcription = []

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcription.append(result.get("text", ""))

        final_result = json.loads(rec.FinalResult())
        transcription.append(final_result.get("text", ""))
        lyrics = " ".join(transcription).strip()
        logger.info(f"[OK] Vosk transcribed => {vocal_path}")
        return lyrics if lyrics else "No lyrics detected"
    except Exception as e:
        logger.error(f"Vosk transcription error: {e}")
        return "No lyrics detected"

def transcribe_lyrics(audio_path, vosk_model, logger):
    """
    Primary: Whisper  
    Fallback: Vosk (if Whisper fails)
    """
    whisper_result = transcribe_lyrics_whisper(audio_path, logger)
    if whisper_result is not None:
        return whisper_result

    logger.warning("[Fallback] Whisper failed; using Vosk")
    return transcribe_lyrics_vosk(audio_path, vosk_model, logger)

# ------------------ Feature Extraction (Tempo, Key, etc.) ------------------ #
def estimate_tempo(y: np.ndarray, sr: int, logger: logging.Logger) -> float:
    try:
        if len(y) < sr * 3:
            logger.warning("Audio too short for reliable tempo estimation, using default")
            return 120.0
        if np.max(np.abs(y)) < 0.01:
            logger.warning("Audio signal too quiet for tempo estimation")
            return 120.0
        y_harmonic, _ = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(
            y=y_harmonic, sr=sr, hop_length=512, n_fft=2048, aggregate=np.median
        )
        if len(onset_env) < sr // 512:
            logger.warning("Not enough onset data for tempo estimation")
            return 120.0
        onset_env = librosa.util.normalize(onset_env)
        try:
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            if not np.isfinite(tempo):
                raise ValueError("Tempo is infinite or NaN")
            if not (60 <= tempo <= 240):
                logger.warning(f"Estimated tempo outside range: {tempo:.2f} BPM, adjusting with fallback")
                ac = librosa.autocorrelate(onset_env, max_size=sr // 512 * 4)
                min_lag = max(sr // 512 * 60 // 240, 1)
                max_lag = min(sr // 512 * 60 // 60, len(ac) - 1)
                if min_lag >= max_lag:
                    logger.warning("Invalid lag bounds, using default tempo")
                    return 120.0
                peaks = librosa.util.peak_pick(ac[min_lag:max_lag+1],
                                              pre_max=3, post_max=3,
                                              pre_avg=3, post_avg=3,
                                              delta=0.5, wait=10)
                if len(peaks) > 0:
                    peaks = peaks + min_lag
                    best_lag = peaks[np.argmax(ac[peaks])]
                    tempo = 60 * sr / (512 * best_lag)
                else:
                    tempo = 120.0
        except Exception as e:
            logger.warning(f"Primary tempo estimation failed: {e}, trying fallback")
            try:
                tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
                tempo = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)[np.argmax(np.median(tempogram, axis=1))]
            except Exception as inner_e:
                logger.warning(f"Fallback tempo estimation failed: {inner_e}, using default tempo")
                tempo = 120.0
        if not np.isfinite(tempo):
            logger.warning("Estimated tempo is infinite or NaN, defaulting to 120 BPM")
            tempo = 120.0
        elif not (60 <= tempo <= 240):
            logger.warning(f"Estimated tempo outside range: {tempo:.2f} BPM, defaulting to 120 BPM")
            tempo = 120.0
        logger.info(f"Estimated tempo: {tempo:.2f} BPM")
        return float(tempo)
    except Exception as e:
        logger.error(f"Tempo estimation failed: {e}")
        return 120.0
    
def estimate_time_signature(y: np.ndarray, sr: int, logger: logging.Logger) -> str:
    try:
        if len(y) < sr * 5:
            logger.warning("Audio too short for time signature estimation")
            return "4/4"
        y_harmonic, _ = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y_harmonic, sr=sr, hop_length=512, n_fft=2048)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512, backtrack=True,
            pre_max=30, post_max=30, pre_avg=100, post_avg=100, delta=0.07, wait=30
        )
        if len(onset_frames) < 8:
            logger.warning("Not enough onsets for time signature estimation")
            return "4/4"
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
        intervals = np.diff(onset_times)
        intervals = intervals[intervals > 0.1]
        if len(intervals) < 4:
            logger.warning("Not enough valid intervals for time signature estimation")
            return "4/4"
        try:
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            beat_period = 60 / tempo
            beat_multiples = np.round(intervals / beat_period)
            unique_multiples, counts = np.unique(beat_multiples, return_counts=True)
            if len(unique_multiples) > 0 and np.max(counts) > len(intervals) * 0.3:
                dominant_multiple = unique_multiples[np.argmax(counts)]
                if dominant_multiple == 2:
                    return "2/4"
                elif dominant_multiple == 3:
                    return "3/4"
                elif dominant_multiple == 4:
                    return "4/4"
                elif dominant_multiple == 6:
                    return "6/8"
                elif dominant_multiple == 9:
                    return "9/8"
                elif dominant_multiple == 12:
                    return "12/8"
                else:
                    logger.info(f"Uncommon time signature multiple: {dominant_multiple}, defaulting to 4/4")
                    return "4/4"
            beat_variance = np.var(intervals)
            normalized_variance = beat_variance / np.mean(intervals)
            if normalized_variance > 0.3:
                logger.info("High beat variance detected, defaulting to 4/4")
                return "4/4"
            hist, _ = np.histogram(intervals, bins=20)
            if len(np.where(hist > np.max(hist) * 0.5)[0]) >= 2:
                return "Compound"
        except Exception as e:
            logger.warning(f"Beat tracking failed: {e}, defaulting to 4/4")
        logger.info("Defaulting to 4/4 time signature")
        return "4/4"
    except Exception as e:
        logger.error(f"Time signature estimation failed: {e}")
        return "4/4"
    
def estimate_key(y: np.ndarray, sr: int, logger, chromagram=None):
    try:
        if chromagram is None:
            y_harmonic = librosa.effects.harmonic(y)
            chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        average_chroma = np.mean(chromagram, axis=1)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        def rotate_profile(profile, steps):
            return np.roll(profile, steps)
        major_keys = [rotate_profile(major_profile, -i) for i in range(12)]
        minor_keys = [rotate_profile(minor_profile, -i) for i in range(12)]
        all_profiles = major_keys + minor_keys
        correlations = [np.corrcoef(average_chroma, profile)[0, 1] for profile in all_profiles]
        best_match = np.argmax(correlations)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if best_match < 12:
            key_type = 'major'
            key_index = best_match
        else:
            key_type = 'minor'
            key_index = best_match - 12
        detected_key = f"{key_names[key_index]} {key_type}"
        logger.info(f"Estimated key: {detected_key}")
        return detected_key
    except Exception as e:
        logger.warning(f"Key estimation failed: {e}, defaulting to 'C major'")
        return 'C major'

def extract_chord_progression(y, sr, logger):
    try:
        hop_length = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        if chroma.size == 0:
            logger.warning("Empty chroma for chord progression")
            return []
        n_frames = min(chroma.shape[1], 500)
        chroma = chroma[:, :n_frames]
        try:
            chroma_frames = librosa.util.frame(chroma, frame_length=8, hop_length=1, axis=1)
        except ValueError:
            logger.warning("Could not frame chroma, using original")
            chroma_frames = chroma.T.reshape(-1, 12, 1)
        chord_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        chords = []
        for i, frame in enumerate(chroma_frames):
            pitch_profile = np.sum(frame, axis=1) if frame.ndim > 1 else frame
            if np.sum(pitch_profile) == 0:
                continue
            root_idx = np.argmax(pitch_profile)
            root = chord_names[root_idx]
            third_energy = pitch_profile[(root_idx + 4) % 12]
            minor_third_energy = pitch_profile[(root_idx + 3) % 12]
            seventh_energy = pitch_profile[(root_idx + 10) % 12]
            if third_energy > minor_third_energy:
                quality = "" if seventh_energy < 0.5 * third_energy else "7"
            else:
                quality = "m" if seventh_energy < 0.5 * minor_third_energy else "m7"
            start_time = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(i + 1, sr=sr, hop_length=hop_length)
            chords.append({
                "root": root,
                "quality": quality,
                "start_time": float(start_time),
                "end_time": float(end_time)
            })
        simplified_chords = [chords[0]] if chords else []
        for chord in chords[1:]:
            if chord['root'] != simplified_chords[-1]['root'] or chord['quality'] != simplified_chords[-1]['quality']:
                simplified_chords.append(chord)
        return simplified_chords[:100]
    except Exception as e:
        logger.error(f"Chord progression extraction failed: {e}")
        return []
def extract_extended_features(y, sr, logger, vis_dir):
    features = {}
    try:
        # Check if audio is silent or too short
        if len(y) < sr * 3:
            logger.warning("Audio too short for feature extraction")
            raise ValueError("Audio too short")
        if np.max(np.abs(y)) < 0.01:
            logger.warning("Audio signal too quiet for feature extraction")
            raise ValueError("Audio too quiet")

        # Duration
        features['duration'] = float(librosa.get_duration(y=y, sr=sr))
        
        # Tempo
        features['tempo'] = estimate_tempo(y, sr, logger)
        
        # Time Signature
        features['time_signature'] = estimate_time_signature(y, sr, logger)
        
        # Key
        features['key'] = estimate_key(y, sr, logger)
        
        # Spectral Features
        try:
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        except Exception as e:
            logger.error(f"Spectral centroid failed: {e}")
            features['spectral_centroid'] = 0.0

        try:
            features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        except Exception as e:
            logger.error(f"Spectral bandwidth failed: {e}")
            features['spectral_bandwidth'] = 0.0

        try:
            features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        except Exception as e:
            logger.error(f"Spectral rolloff failed: {e}")
            features['spectral_rolloff'] = 0.0

        try:
            features['rms'] = float(np.mean(librosa.feature.rms(y=y)))
        except Exception as e:
            logger.error(f"RMS failed: {e}")
            features['rms'] = 0.0

        try:
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
        except Exception as e:
            logger.error(f"Zero crossing rate failed: {e}")
            features['zero_crossing_rate'] = 0.0
        
        # Advanced Features
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc'] = mfcc.mean(axis=1).tolist()
        except Exception as e:
            logger.error(f"MFCC failed: {e}")
            features['mfcc'] = [0.0] * 13

        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma'] = chroma.mean(axis=1).tolist()
        except Exception as e:
            logger.error(f"Chroma failed: {e}")
            features['chroma'] = [0.0] * 12

        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = spectral_contrast.mean(axis=1).tolist()
        except Exception as e:
            logger.error(f"Spectral contrast failed: {e}")
            features['spectral_contrast'] = [0.0] * 7

        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features['tonnetz'] = tonnetz.mean(axis=1).tolist()
        except Exception as e:
            logger.error(f"Tonnetz failed: {e}")
            features['tonnetz'] = [0.0] * 6

        try:
            features['chord_progression'] = extract_chord_progression(y, sr, logger)
        except Exception as e:
            logger.error(f"Chord progression failed: {e}")
            features['chord_progression'] = []

        # Chromagram Visualization
        try:
            y_harmonic, _ = librosa.effects.hpss(y)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.title('Chromagram')
            plt.tight_layout()
            chroma_path = os.path.join(vis_dir, "chromagram.png")
            plt.savefig(chroma_path)
            plt.close()
            features['chromagram_path'] = chroma_path
        except Exception as e:
            logger.error(f"Chromagram visualization failed: {e}")
            features['chromagram_path'] = "error"

        # Waveform Visualization
        try:
            plt.figure(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr)
            plt.title('Waveform')
            plt.tight_layout()
            wave_path = os.path.join(vis_dir, "waveform.png")
            plt.savefig(wave_path)
            plt.close()
            features['waveform_path'] = wave_path
        except Exception as e:
            logger.error(f"Waveform visualization failed: {e}")
            features['waveform_path'] = "error"
        
        return features
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return {
            'duration': 0.0, 'tempo': 120.0, 'time_signature': "4/4", 'key': "C major",
            'spectral_centroid': 0.0, 'spectral_bandwidth': 0.0, 'spectral_rolloff': 0.0,
            'rms': 0.0, 'zero_crossing_rate': 0.0, 'mfcc': [0.0]*13, 'chroma': [0.0]*12,
            'spectral_contrast': [0.0]*7, 'tonnetz': [0.0]*6, 'chord_progression': [],
            'chromagram_path': "error", 'waveform_path': "error"
        }
    
def plot_spectrogram(y, sr, output_path, logger):
    try:
        plt.figure(figsize=(12, 6))
        max_samples = sr * 60
        y_segment = y[:max_samples] if len(y) > max_samples else y
        S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', hop_length=512)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency Spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Spectrogram saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Spectrogram plotting failed: {e}")
        try:
            plt.close()
        except:
            pass
        return False

# ------------------ Metadata Extraction ------------------ #
def extract_metadata(audio_file: str, logger) -> Dict[str, str]:
    try:
        file_ext = Path(audio_file).suffix.lower()
        if file_ext == '.mp3':
            audio = EasyID3(audio_file)
            return {key: audio.get(key, ["Unknown"])[0] for key in ['title', 'artist', 'album', 'genre']}
        elif file_ext == '.wav':
            return {'title': 'Unknown', 'artist': 'Unknown', 'album': 'Unknown', 'genre': 'Unknown'}
        elif file_ext == '.flac':
            audio = FLAC(audio_file)
            return {'title': audio.get('title', ['Unknown'])[0],
                    'artist': audio.get('artist', ['Unknown'])[0],
                    'album': audio.get('album', ['Unknown'])[0],
                    'genre': audio.get('genre', ['Unknown'])[0]}
        elif file_ext == '.ogg':
            audio = OggVorbis(audio_file)
            return {'title': audio.get('title', ['Unknown'])[0],
                    'artist': audio.get('artist', ['Unknown'])[0],
                    'album': audio.get('album', ['Unknown'])[0],
                    'genre': audio.get('genre', ['Unknown'])[0]}
        elif file_ext == '.m4a':
            audio = MP4(audio_file)
            return {'title': audio.get('\xa9nam', ['Unknown'])[0],
                    'artist': audio.get('\xa9ART', ['Unknown'])[0],
                    'album': audio.get('\xa9alb', ['Unknown'])[0],
                    'genre': audio.get('\xa9gen', ['Unknown'])[0]}
        else:
            logger.warning(f"Unsupported file format for metadata: {audio_file}")
            return {'title': 'Unknown', 'artist': 'Unknown', 'album': 'Unknown', 'genre': 'Unknown'}
    except Exception as e:
        logger.warning(f"Metadata extraction failed for {audio_file}: {e}")
        return {'title': 'Unknown', 'artist': 'Unknown', 'album': 'Unknown', 'genre': 'Unknown'}

def verify_tempo(audio_path, features, logger):
    """
    Re-check the tempo on the standardized file to confirm.
    """
    try:
        y, sr = load_audio(audio_path, logger)
        if y is None or sr is None:
            logger.warning(f"Could not load audio for tempo verification: {audio_path}")
            return features.get('tempo', 120.0)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        if tempo <= 10 or tempo > 300:
            logger.warning(f"Tempo {tempo} out of range, falling back to features['tempo'] or default")
            return features.get('tempo', 120.0) if features.get('tempo', 0.0) > 0 else 120.0
        logger.info(f"Verified tempo for {audio_path}: {tempo} BPM")
        return tempo
    except Exception as e:
        logger.error(f"Tempo verification failed for {audio_path}: {e}")
        return features.get('tempo', 120.0)

# ------------------ Main Song Directory Processing ------------------ #
def process_song_directory(song_dir, root_dir, vosk_model, logger):
    try:
        song_dir_path = os.path.join(root_dir, song_dir)
        audio_dir = os.path.join(song_dir_path, "audio")
        if not os.path.exists(audio_dir):
            return False, f"No audio directory found in {song_dir}"

        # Check for audio files, excluding vocals.wav to avoid using potentially corrupted files
        audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a')) and f != "vocals.wav"]
        audio_files.sort(key=lambda x: {'.wav': 0, '.flac': 1, '.m4a': 2, '.aac': 3, '.mp3': 4, '.ogg': 5}.get(os.path.splitext(x.lower())[1], 99))
        audio_file = os.path.join(audio_dir, audio_files[0]) if audio_files else None
        
        if not audio_file or not os.path.exists(audio_file):
            return False, f"No audio files found in {audio_dir}"

        logger.info(f"Selected {os.path.basename(audio_file)} for processing in {song_dir}")

        visuals_dir = os.path.join(song_dir_path, "visuals")
        dataset_dir = os.path.join(song_dir_path, "dataset")
        features_dir = os.path.join(dataset_dir, "features")
        lyrics_dir = os.path.join(song_dir_path, "lyrics")
        
        os.makedirs(visuals_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(lyrics_dir, exist_ok=True)

        # Standardize
        standardized_path = os.path.join(audio_dir, "standardized.wav")
        standardized_audio = standardize_audio(audio_file, standardized_path, target_sr=32000, logger=logger)
        if standardized_audio is None or not os.path.exists(standardized_audio):
            return False, f"Failed to standardize audio for {song_dir}"

        # Extract Metadata
        metadata = extract_metadata(audio_file, logger) or {}
        with open(os.path.join(dataset_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        # Load standardized audio
        y, sr = load_audio(standardized_audio, logger)
        if y is None or sr is None:
            return False, f"Failed to load standardized audio {standardized_audio}"

        # Extract features
        features = extract_extended_features(y, sr, logger, visuals_dir)
        if not features:
            logger.warning(f"Features extraction returned empty result for {song_dir}")
            features = {}

    

        # Add spectrogram path
        features_with_spectrogram = features.copy()
        features_with_spectrogram['spectrogram_path'] = os.path.join("visuals", "spectrogram.png")
        with open(os.path.join(features_dir, "features.json"), 'w') as f:
            json.dump(features_with_spectrogram, f, indent=4)

        # Plot spectrogram
        spectrogram_path = os.path.join(visuals_dir, "spectrogram.png")
        plot_spectrogram(y, sr, spectrogram_path, logger)

        # Transcribe
        lyrics_file = os.path.join(lyrics_dir, f"{song_dir}_Lyrics.txt")
        if os.path.exists(lyrics_file):
            logger.info(f"[SKIP] Already have lyrics => {lyrics_file}")
        else:
            lyrics = transcribe_lyrics(standardized_audio, vosk_model, logger)
            with open(lyrics_file, 'w') as f:
                f.write(lyrics)
            logger.info(f"Transcribed lyrics => {lyrics_file}")

        # Cleanup separated dir
        demucs_dir = os.path.join(audio_dir, "separated")
        if os.path.exists(demucs_dir):
            try:
                shutil.rmtree(demucs_dir)
                logger.info(f"Cleaned up Demucs directory: {demucs_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up Demucs directory: {e}")

        logger.info(f"Successfully processed: {song_dir}")
        return True, f"Processed {song_dir}"

    except Exception as e:
        logger.error(f"Failed to process {song_dir}: {e}")
        traceback.print_exc()
        return False, f"Failed to process {song_dir}: {str(e)}"

# ------------------ Main Function ------------------ #
def main():
    args = parse_arguments()
    global logger
    logger = setup_logging()

    print(f"Using device: {device}")

    # Load Vosk model
    try:
        vosk_model = VoskModel(args.vosk_model)
        logger.info(f"Loaded Vosk model from {args.vosk_model}")
    except Exception as e:
        logger.warning(f"Failed to load Vosk model: {e}. Will rely on Whisper only.")
        vosk_model = None

    root_dir = Path(args.root_dir)
    song_dirs = [
        d.name for d in root_dir.iterdir()
        if d.is_dir() and (d / "audio").exists() and any(
            f.name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a'))
            for f in (d / "audio").iterdir()
        )
    ]
    if not song_dirs:
        logger.critical("No valid song directories found!")
        return

    logger.info(f"Found {len(song_dirs)} song directories to process")

    catalog = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_song_directory, d, str(root_dir), vosk_model, logger): d for d in song_dirs}
        success_count = 0
        failed_count = 0
        for i, future in enumerate(as_completed(futures), 1):
            song_dir = futures[future]
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                    catalog.append({
                        "song_id": song_dir,
                        "audio_path": str(Path(song_dir) / "audio" / "standardized.wav"),
                        "features_path": str(Path(song_dir) / "dataset" / "features" / "features.json"),
                        "lyrics_path": str(Path(song_dir) / "lyrics" / f"{song_dir}_Lyrics.txt"),
                        "metadata_path": str(Path(song_dir) / "dataset" / "metadata.json"),
                        "spectrogram_path": str(Path(song_dir) / "visuals" / "spectrogram.png"),
                        "chromagram_path": str(Path(song_dir) / "visuals" / "chromagram.png"),
                        "waveform_path": str(Path(song_dir) / "visuals" / "waveform.png")
                    })
                else:
                    failed_count += 1
                logger.info(f"[{i}/{len(song_dirs)}] {msg}")
            except Exception as e:
                logger.error(f"Error processing {song_dir}: {e}")
                failed_count += 1

    # Write catalog
    with open(root_dir / "catalog.json", 'w') as f:
        json.dump(catalog, f, indent=4)
    logger.info(f"Catalog generated with {len(catalog)} songs")
    logger.info(f"Processing complete. Success: {success_count}, Failed: {failed_count}, Total: {len(song_dirs)}")
    if failed_count > 0:
        logger.warning(f"{failed_count} directories failed. Check the logs for details.")

if __name__ == "__main__":
    main()
