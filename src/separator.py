"""
AICoverGen NextGen - Audio Separator
Vocal separation and DeReverb processing with optimizations
"""

import gc
import os
from pathlib import Path

import torch

# Handle both module and script execution
try:
    from . import config
except ImportError:
    import config


def _clear_gpu_memory():
    """Clear GPU memory after heavy operations"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Try to import audio-separator
try:
    from audio_separator.separator import Separator
    SEPARATOR_AVAILABLE = True
except ImportError:
    SEPARATOR_AVAILABLE = False
    print("⚠️  audio-separator not available. Install with: pip install audio-separator[gpu]")


def separate_vocals_bs_roformer(input_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Separate vocals using BS-RoFormer (SDR 12.97 - BEST quality)
    Returns: (vocals_path, instrumental_path)
    """
    if not SEPARATOR_AVAILABLE:
        return None, None
    
    print("[~] Separating vocals (BS-RoFormer SDR 12.97)...")
    
    separator = Separator(
        output_dir=output_dir,
        output_format="wav"
    )
    separator.load_model(model_filename=config.BS_ROFORMER_MODEL)
    
    output_files = separator.separate(input_path)
    
    vocals_path = None
    instrumental_path = None
    
    for f in output_files:
        f_lower = os.path.basename(f).lower()
        if not os.path.isabs(f):
            f = os.path.join(output_dir, os.path.basename(f))
        
        if '(vocals)' in f_lower:
            vocals_path = f
        elif '(instrumental)' in f_lower or '(other)' in f_lower:
            instrumental_path = f
    
    # Fallback: scan directory
    if vocals_path and not os.path.exists(vocals_path):
        for file in os.listdir(output_dir):
            if '(Vocals)' in file and file.endswith('.wav'):
                vocals_path = os.path.join(output_dir, file)
                break
    
    if instrumental_path and not os.path.exists(instrumental_path):
        for file in os.listdir(output_dir):
            if '(Instrumental)' in file and file.endswith('.wav'):
                instrumental_path = os.path.join(output_dir, file)
                break
    
    if vocals_path:
        print(f"✓ Vocals: {os.path.basename(vocals_path)}")
    if instrumental_path:
        print(f"✓ Instrumental: {os.path.basename(instrumental_path)}")
    
    # Clear GPU memory after separation
    _clear_gpu_memory()
    
    return vocals_path, instrumental_path


def remove_reverb(vocals_path: str, output_dir: str) -> str:
    """
    Remove reverb and echo from vocals using UVR-DeEcho-DeReverb
    Returns: clean_vocals_path (or original if failed)
    """
    if not SEPARATOR_AVAILABLE:
        print("⚠️  audio-separator not available, skipping DeReverb")
        return vocals_path
    
    dereverb_model = config.get_model_path(config.DEREVERB_MODEL, "mdx")
    if not dereverb_model.exists():
        print(f"⚠️  DeReverb model not found, skipping")
        return vocals_path
    
    try:
        print("[~] Removing reverb/echo (UVR-DeEcho-DeReverb)...")
        
        separator = Separator(
            output_dir=output_dir,
            output_format="wav"
        )
        separator.load_model(model_filename=config.DEREVERB_MODEL)
        
        output_files = separator.separate(vocals_path)
        
        # Find clean vocals (No Reverb)
        clean_vocals_path = None
        for f in output_files:
            f_lower = os.path.basename(f).lower()
            if not os.path.isabs(f):
                f = os.path.join(output_dir, os.path.basename(f))
            
            if 'no reverb' in f_lower or 'noreverb' in f_lower:
                clean_vocals_path = f
                break
        
        # Fallback
        if clean_vocals_path is None:
            for f in output_files:
                f_lower = os.path.basename(f).lower()
                if not os.path.isabs(f):
                    f = os.path.join(output_dir, os.path.basename(f))
                if 'reverb' not in f_lower or 'no' in f_lower:
                    clean_vocals_path = f
                    break
        
        if clean_vocals_path and os.path.exists(clean_vocals_path):
            print(f"✓ DeReverb complete: {os.path.basename(clean_vocals_path)}")
            _clear_gpu_memory()
            return clean_vocals_path
        else:
            print("⚠️  DeReverb output not found, using original")
            return vocals_path
            
    except Exception as e:
        print(f"⚠️  DeReverb failed: {e}")
        return vocals_path
    finally:
        _clear_gpu_memory()


def separate_backing_vocals(instrumental_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Separate backing vocals from instrumental using mel_band_roformer_karaoke
    This creates a cleaner instrumental without backing vocals bleeding through
    Returns: (clean_instrumental_path, backing_vocals_path)
    """
    if not SEPARATOR_AVAILABLE:
        print("⚠️  audio-separator not available, skipping backing vocal separation")
        return instrumental_path, None
    
    try:
        print("[~] Separating backing vocals (mel_band_roformer_karaoke SDR 10.19)...")
        
        separator = Separator(
            output_dir=output_dir,
            output_format="wav"
        )
        separator.load_model(model_filename=config.MEL_BAND_ROFORMER_KARAOKE)
        
        output_files = separator.separate(instrumental_path)
        
        clean_instrumental = None
        backing_vocals = None
        
        for f in output_files:
            f_lower = os.path.basename(f).lower()
            if not os.path.isabs(f):
                f = os.path.join(output_dir, os.path.basename(f))
            
            # Karaoke model outputs: (Instrumental) = clean, (Vocals) = backing
            if '(instrumental)' in f_lower or 'karaoke' in f_lower:
                clean_instrumental = f
            elif '(vocals)' in f_lower:
                backing_vocals = f
        
        # Fallback: scan directory for karaoke outputs
        if clean_instrumental is None or backing_vocals is None:
            for file in os.listdir(output_dir):
                file_lower = file.lower()
                if 'karaoke' in file_lower or ('instrumental' in file_lower and 'mel_band' in file_lower):
                    if '(instrumental)' in file_lower:
                        clean_instrumental = os.path.join(output_dir, file)
                    elif '(vocals)' in file_lower:
                        backing_vocals = os.path.join(output_dir, file)
        
        if clean_instrumental and os.path.exists(clean_instrumental):
            print(f"✓ Clean instrumental: {os.path.basename(clean_instrumental)}")
        else:
            print("⚠️  Clean instrumental not found, using original")
            clean_instrumental = instrumental_path
            
        if backing_vocals and os.path.exists(backing_vocals):
            print(f"✓ Backing vocals extracted: {os.path.basename(backing_vocals)}")
        
        _clear_gpu_memory()
        return clean_instrumental, backing_vocals
        
    except Exception as e:
        print(f"⚠️  Backing vocal separation failed: {e}")
        return instrumental_path, None
    finally:
        _clear_gpu_memory()


def find_cached_audio(song_dir: str) -> tuple[str | None, str | None, str | None]:
    """
    Find cached audio files in song directory
    Returns: (orig_song_path, instrumentals_path, vocals_path)
    """
    orig_song_path = None
    instrumentals_path = None
    vocals_path = None
    vocals_dereverb_path = None

    for file in os.listdir(song_dir):
        file_lower = file.lower()
        
        # Priority 1: DeReverb vocals
        if 'no reverb' in file_lower and file.endswith('.wav'):
            vocals_dereverb_path = os.path.join(song_dir, file)
        # Priority 2: Raw vocals
        elif '(vocals)' in file_lower and file.endswith('.wav') and 'reverb' not in file_lower:
            if vocals_path is None:
                vocals_path = os.path.join(song_dir, file)
        elif '(instrumental)' in file_lower and file.endswith('.wav'):
            instrumentals_path = os.path.join(song_dir, file)
        # MDX-Net fallback
        elif file.endswith('_Vocals.wav'):
            if vocals_path is None:
                vocals_path = os.path.join(song_dir, file)
        elif file.endswith('_Instrumental.wav'):
            if instrumentals_path is None:
                instrumentals_path = os.path.join(song_dir, file)

    # Find original song
    if orig_song_path is None:
        for file in os.listdir(song_dir):
            if file.endswith('.mp3'):
                orig_song_path = os.path.join(song_dir, file)
                break
            elif file.endswith('_stereo.wav'):
                orig_song_path = os.path.join(song_dir, file)
                break
            elif file.endswith('.wav') and '(' not in file and '_' not in file:
                orig_song_path = os.path.join(song_dir, file)
                break

    # Use best available vocals
    final_vocals = vocals_dereverb_path or vocals_path

    return orig_song_path, instrumentals_path, final_vocals
