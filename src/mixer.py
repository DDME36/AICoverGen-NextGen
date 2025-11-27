"""
AICoverGen NextGen - Audio Mixer
Audio effects and mixing
"""

import os
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
import soundfile as sf
import sox

# Handle both module and script execution
try:
    from . import config
except ImportError:
    import config


def add_vocal_effects(audio_path: str, reverb_room_size: float = None, 
                      reverb_wet: float = None, reverb_dry: float = None,
                      reverb_damping: float = None) -> str:
    """Apply effects to vocals (highpass, compression, reverb)"""
    
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


def auto_mix(main_vocals_path: str, instrumental_path: str, output_path: str,
             main_gain: int = 0, inst_gain: int = 0, 
             output_format: str = None, backing_vocals_path: str = None,
             backing_gain: int = -6) -> str:
    """
    Auto-mix vocals and instrumental with compression and gain staging
    Optionally includes backing vocals for smoother blend
    """
    output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
    
    print("[~] Auto-mixing with compression and EQ...")
    
    # Load audio
    main_vocal = AudioSegment.from_wav(main_vocals_path)
    instrumental = AudioSegment.from_wav(instrumental_path)
    
    # Load backing vocals if available
    backing_vocal = None
    if backing_vocals_path and os.path.exists(backing_vocals_path):
        print(f"    Including backing vocals for smoother mix")
        backing_vocal = AudioSegment.from_wav(backing_vocals_path)
        backing_vocal = normalize(backing_vocal)
        # Compress backing vocals lightly
        backing_vocal = compress_dynamic_range(
            backing_vocal, threshold=-18.0, ratio=2.5, attack=10.0, release=100.0
        )
        # Lower backing vocals in mix
        backing_vocal = backing_vocal + backing_gain
    
    # Normalize
    main_vocal = normalize(main_vocal)
    instrumental = normalize(instrumental)
    
    # Compression
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
    
    # Mix instrumental first
    mixed = instrumental
    
    # Add backing vocals if available (blend with instrumental)
    if backing_vocal:
        backing_vocal = backing_vocal[:min_length]
        mixed = mixed.overlay(backing_vocal)
    
    # Overlay main vocals on top
    mixed = mixed.overlay(main_vocal)
    
    # Final normalize and limit
    mixed = normalize(mixed)
    if mixed.dBFS > -1.0:
        mixed = mixed - (mixed.dBFS + 1.0)
    
    # Export
    mixed.export(output_path, format=output_format, bitrate="320k")
    print(f"✓ Auto-mix complete: {os.path.basename(output_path)}")
    
    return output_path
