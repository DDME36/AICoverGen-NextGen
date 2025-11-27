"""
AICoverGen NextGen - Audio Mixer
Audio effects and mixing with optimized performance
"""

import os
import numpy as np
import soundfile as sf
import sox
from concurrent.futures import ThreadPoolExecutor

# Handle both module and script execution
try:
    from . import config
except ImportError:
    import config

# Lazy imports for faster startup
_pydub_loaded = False
_pedalboard_loaded = False


def _load_pydub():
    global _pydub_loaded, AudioSegment, normalize, compress_dynamic_range
    if not _pydub_loaded:
        from pydub import AudioSegment
        from pydub.effects import normalize, compress_dynamic_range
        _pydub_loaded = True


def _load_pedalboard():
    global _pedalboard_loaded, Pedalboard, Reverb, Compressor, HighpassFilter, AudioFile
    if not _pedalboard_loaded:
        from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
        from pedalboard.io import AudioFile
        _pedalboard_loaded = True


def add_vocal_effects(audio_path: str, reverb_room_size: float = None, 
                      reverb_wet: float = None, reverb_dry: float = None,
                      reverb_damping: float = None) -> str:
    """Apply effects to vocals (highpass, compression, reverb)"""
    _load_pedalboard()
    
    # Use defaults if not specified
    reverb_room_size = reverb_room_size or config.DEFAULT_REVERB_ROOM_SIZE
    reverb_wet = reverb_wet or config.DEFAULT_REVERB_WET
    reverb_dry = reverb_dry or config.DEFAULT_REVERB_DRY
    reverb_damping = reverb_damping or config.DEFAULT_REVERB_DAMPING
    
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    board = Pedalboard([
        HighpassFilter(),
        Compressor(ratio=4, threshold_db=-15),
        Reverb(room_size=reverb_room_size, dry_level=reverb_dry, 
               wet_level=reverb_wet, damping=reverb_damping)
    ])

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def pitch_shift(audio_path: str, pitch_change: int) -> str:
    """Shift pitch of audio file"""
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def _load_audio_numpy(path: str) -> tuple[np.ndarray, int]:
    """Load audio as numpy array (faster than pydub)"""
    data, sr = sf.read(path)
    if data.ndim == 1:
        data = np.column_stack([data, data])  # mono to stereo
    return data, sr


def _normalize_numpy(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Normalize audio to target dB level"""
    max_val = np.abs(audio).max()
    if max_val > 0:
        target_linear = 10 ** (target_db / 20)
        audio = audio * (target_linear / max_val)
    return audio


def _apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in dB"""
    return audio * (10 ** (gain_db / 20))


def _mix_numpy(tracks: list[np.ndarray], gains: list[float]) -> np.ndarray:
    """Mix multiple tracks with gains (numpy-based, fast)"""
    # Find minimum length
    min_len = min(len(t) for t in tracks)
    
    # Mix with gains
    mixed = np.zeros((min_len, 2), dtype=np.float32)
    for track, gain in zip(tracks, gains):
        mixed += _apply_gain(track[:min_len], gain)
    
    return mixed


def _soft_limit(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Soft limiter to prevent clipping"""
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (threshold + (1 - threshold) * np.tanh((np.abs(audio[mask]) - threshold) / (1 - threshold)))
    return audio


def auto_mix(main_vocals_path: str, instrumental_path: str, output_path: str,
             main_gain: int = 0, inst_gain: int = 0, 
             output_format: str = None, backing_vocals_path: str = None,
             backing_gain: int = -6) -> str:
    """
    Auto-mix vocals and instrumental - OPTIMIZED with numpy
    Much faster than pydub-based mixing
    """
    output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
    
    print("[~] Auto-mixing (optimized)...")
    
    # Load audio in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            'vocal': executor.submit(_load_audio_numpy, main_vocals_path),
            'inst': executor.submit(_load_audio_numpy, instrumental_path),
        }
        if backing_vocals_path and os.path.exists(backing_vocals_path):
            futures['backing'] = executor.submit(_load_audio_numpy, backing_vocals_path)
        
        main_vocal, sr = futures['vocal'].result()
        instrumental, _ = futures['inst'].result()
        backing_vocal = futures['backing'].result()[0] if 'backing' in futures else None
    
    # Normalize
    main_vocal = _normalize_numpy(main_vocal, -3.0)
    instrumental = _normalize_numpy(instrumental, -5.0)
    
    # Prepare tracks and gains
    tracks = [instrumental, main_vocal]
    gains = [inst_gain, main_gain]
    
    if backing_vocal is not None:
        print("    Including backing vocals")
        backing_vocal = _normalize_numpy(backing_vocal, -6.0)
        tracks.append(backing_vocal)
        gains.append(backing_gain)
    
    # Mix
    mixed = _mix_numpy(tracks, gains)
    
    # Normalize and limit
    mixed = _normalize_numpy(mixed, -1.0)
    mixed = _soft_limit(mixed)
    
    # Export
    if output_format == 'mp3':
        # For MP3, use pydub (required for encoding)
        _load_pydub()
        temp_wav = output_path.replace('.mp3', '_temp.wav')
        sf.write(temp_wav, mixed, sr)
        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format='mp3', bitrate='320k')
        os.remove(temp_wav)
    else:
        sf.write(output_path, mixed, sr)
    
    print(f"✓ Auto-mix complete: {os.path.basename(output_path)}")
    return output_path


def auto_mix_pydub(main_vocals_path: str, instrumental_path: str, output_path: str,
                   main_gain: int = 0, inst_gain: int = 0, 
                   output_format: str = None, backing_vocals_path: str = None,
                   backing_gain: int = -6) -> str:
    """
    Auto-mix using pydub (legacy, slower but more features)
    """
    _load_pydub()
    output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
    
    print("[~] Auto-mixing (pydub)...")
    
    # Load audio
    main_vocal = AudioSegment.from_wav(main_vocals_path)
    instrumental = AudioSegment.from_wav(instrumental_path)
    
    # Load backing vocals if available
    backing_vocal = None
    if backing_vocals_path and os.path.exists(backing_vocals_path):
        print(f"    Including backing vocals")
        backing_vocal = AudioSegment.from_wav(backing_vocals_path)
        backing_vocal = normalize(backing_vocal)
        backing_vocal = compress_dynamic_range(
            backing_vocal, threshold=-18.0, ratio=2.5, attack=10.0, release=100.0
        )
        backing_vocal = backing_vocal + backing_gain
    
    # Normalize and compress
    main_vocal = normalize(main_vocal)
    instrumental = normalize(instrumental)
    
    main_vocal = compress_dynamic_range(
        main_vocal, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0
    )
    instrumental = compress_dynamic_range(
        instrumental, threshold=-15.0, ratio=2.0, attack=20.0, release=200.0
    )
    
    # Gain staging
    main_vocal = main_vocal - 3 + main_gain
    instrumental = instrumental - 5 + inst_gain
    
    # Match lengths
    min_length = min(len(main_vocal), len(instrumental))
    main_vocal = main_vocal[:min_length]
    instrumental = instrumental[:min_length]
    
    # Mix
    mixed = instrumental
    if backing_vocal:
        backing_vocal = backing_vocal[:min_length]
        mixed = mixed.overlay(backing_vocal)
    mixed = mixed.overlay(main_vocal)
    
    # Final normalize
    mixed = normalize(mixed)
    if mixed.dBFS > -1.0:
        mixed = mixed - (mixed.dBFS + 1.0)
    
    # Export
    mixed.export(output_path, format=output_format, bitrate="320k")
    print(f"✓ Auto-mix complete: {os.path.basename(output_path)}")
    
    return output_path
