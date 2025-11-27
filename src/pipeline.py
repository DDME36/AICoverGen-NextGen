"""
AICoverGen NextGen - Main Pipeline
Orchestrates the entire AI cover generation process
"""

import json
import os
from urllib.parse import urlparse

# Handle both module and script execution
try:
    from . import config
    from .downloader import get_youtube_video_id, download_from_youtube, convert_to_stereo, get_file_hash
    from .separator import separate_vocals_bs_roformer, remove_reverb, separate_backing_vocals, find_cached_audio, SEPARATOR_AVAILABLE
    from .voice_converter import convert_voice
    from .mixer import add_vocal_effects, pitch_shift, auto_mix
    from .mdx import run_mdx
except ImportError:
    import config
    from downloader import get_youtube_video_id, download_from_youtube, convert_to_stereo, get_file_hash
    from separator import separate_vocals_bs_roformer, remove_reverb, separate_backing_vocals, find_cached_audio, SEPARATOR_AVAILABLE
    from voice_converter import convert_voice
    from mixer import add_vocal_effects, pitch_shift, auto_mix
    from mdx import run_mdx


def preprocess_song(song_input: str, song_id: str, input_type: str, 
                    progress_callback=None, separate_backing: bool = True) -> tuple:
    """
    Download and separate vocals from song
    Returns: (orig_path, vocals_path, instrumental_path, dereverb_vocals_path, backing_vocals_path)
    """
    song_output_dir = str(config.get_output_dir(song_id))
    
    # Download or use local file
    if input_type == 'yt':
        if progress_callback:
            progress_callback('[~] Downloading song...', 0)
        song_link = song_input.split('&')[0]
        orig_song_path = download_from_youtube(song_link)
    else:
        orig_song_path = song_input
    
    # Convert to stereo
    orig_song_path = convert_to_stereo(orig_song_path)
    
    # Separate vocals
    if progress_callback:
        progress_callback('[~] Separating vocals (BS-RoFormer)...', 0.1)
    
    if SEPARATOR_AVAILABLE:
        try:
            vocals_path, instrumental_path = separate_vocals_bs_roformer(
                orig_song_path, song_output_dir
            )
            print("✓ BS-RoFormer separation complete (SDR 12.97)")
        except Exception as e:
            print(f"⚠️ BS-RoFormer failed: {e}, falling back to MDX-Net")
            vocals_path, instrumental_path = _fallback_mdx_separation(
                orig_song_path, song_output_dir
            )
    else:
        vocals_path, instrumental_path = _fallback_mdx_separation(
            orig_song_path, song_output_dir
        )
    
    # DeReverb on main vocals
    if progress_callback:
        progress_callback('[~] Removing reverb/echo...', 0.2)
    
    dereverb_vocals_path = remove_reverb(vocals_path, song_output_dir)
    
    # Separate backing vocals from instrumental (optional)
    # Note: We keep the original instrumental for mixing to preserve timing
    # The backing_vocals can be mixed back in at lower volume if needed
    backing_vocals_path = None
    if separate_backing and instrumental_path:
        if progress_callback:
            progress_callback('[~] Separating backing vocals (mel_band_roformer)...', 0.3)
        
        # Get clean instrumental and backing vocals
        # But we'll use original instrumental to preserve timing
        clean_instrumental, backing_vocals_path = separate_backing_vocals(
            instrumental_path, song_output_dir
        )
        # DON'T replace instrumental - keep original for correct timing
        # instrumental_path stays as BS-RoFormer output
    
    return orig_song_path, vocals_path, instrumental_path, dereverb_vocals_path, backing_vocals_path


def _fallback_mdx_separation(orig_path: str, output_dir: str) -> tuple:
    """Fallback to MDX-Net for vocal separation"""
    mdx_params_path = config.MDX_MODELS_DIR / "model_data.json"
    with open(mdx_params_path) as f:
        mdx_params = json.load(f)
    
    mdx_model = config.get_model_path(config.MDX_VOCAL_MODEL, "mdx")
    return run_mdx(mdx_params, output_dir, str(mdx_model), orig_path, denoise=True)


def generate_cover(song_input: str, voice_model: str, pitch_change: int = 0,
                   keep_files: bool = False, main_gain: int = 0, inst_gain: int = 0,
                   index_rate: float = None, filter_radius: int = None,
                   rms_mix_rate: float = None, f0_method: str = None,
                   crepe_hop_length: int = None, protect: float = None,
                   pitch_change_all: int = 0, reverb_room_size: float = None,
                   reverb_wet: float = None, reverb_dry: float = None,
                   reverb_damping: float = None, output_format: str = None,
                   separate_backing: bool = False, 
                   clean_audio: bool = False, clean_strength: float = 0.7,
                   progress_callback=None) -> str:
    """
    Main pipeline: Generate AI cover from song input
    Returns: path to final cover audio
    """
    if not song_input or not voice_model:
        raise ValueError('Song input and voice model are required')
    
    if progress_callback:
        progress_callback('[~] Starting AI Cover Generation...', 0)
    
    # Determine input type and song ID
    if urlparse(song_input).scheme == 'https':
        input_type = 'yt'
        song_id = get_youtube_video_id(song_input)
        if song_id is None:
            raise ValueError('Invalid YouTube URL')
    else:
        input_type = 'local'
        song_input = song_input.strip('"')
        if not os.path.exists(song_input):
            raise FileNotFoundError(f'{song_input} does not exist')
        song_id = get_file_hash(song_input)
    
    song_dir = str(config.get_output_dir(song_id))
    
    # Check cache or preprocess
    backing_vocals_path = None
    if not os.path.exists(song_dir) or not os.listdir(song_dir):
        os.makedirs(song_dir, exist_ok=True)
        orig_path, vocals_path, instrumental_path, dereverb_path, backing_vocals_path = preprocess_song(
            song_input, song_id, input_type, progress_callback, separate_backing
        )
    else:
        # Use cached files
        orig_path, instrumental_path, dereverb_path = find_cached_audio(song_dir)
        
        # Check for cached backing vocals and clean instrumental
        backing_vocals_path = _find_backing_vocals(song_dir)
        clean_instrumental = _find_clean_instrumental(song_dir)
        if clean_instrumental:
            instrumental_path = clean_instrumental
        
        if instrumental_path is None or dereverb_path is None:
            print("[~] Cache incomplete, re-processing...")
            orig_path, vocals_path, instrumental_path, dereverb_path, backing_vocals_path = preprocess_song(
                song_input, song_id, input_type, progress_callback, separate_backing
            )
        else:
            print(f"[✓] Using cached files from {song_dir}")
            print(f"    Vocals: {os.path.basename(dereverb_path)}")
            if backing_vocals_path:
                print(f"    Backing: {os.path.basename(backing_vocals_path)}")
    
    # Calculate final pitch
    final_pitch = pitch_change * 12 + pitch_change_all
    
    # Generate output paths - extract base name from available files
    if orig_path:
        base_name = os.path.splitext(os.path.basename(orig_path))[0]
    elif instrumental_path:
        # Extract base name from instrumental (remove suffix like "(Instrumental)_model...")
        inst_name = os.path.basename(instrumental_path)
        # Find the part before "(Instrumental)" or "(Vocals)"
        for marker in ['_(Instrumental)', '_(Vocals)', '(Instrumental)', '(Vocals)']:
            if marker in inst_name:
                base_name = inst_name.split(marker)[0]
                break
        else:
            base_name = os.path.splitext(inst_name)[0]
    else:
        base_name = "cover"
    
    output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
    
    ai_vocals_path = os.path.join(
        song_dir,
        f'{base_name}_{voice_model}_p{final_pitch}.wav'
    )
    ai_cover_path = os.path.join(
        song_dir,
        f'{base_name} ({voice_model} Ver).{output_format}'
    )
    
    # Voice conversion
    if not os.path.exists(ai_vocals_path):
        if progress_callback:
            progress_callback('[~] Converting voice (RVC)...', 0.5)
        
        convert_voice(
            voice_model, dereverb_path, ai_vocals_path,
            pitch_change=final_pitch, f0_method=f0_method,
            index_rate=index_rate, filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate, protect=protect,
            crepe_hop_length=crepe_hop_length,
            clean_audio=clean_audio, clean_strength=clean_strength
        )
    
    # Apply effects
    if progress_callback:
        progress_callback('[~] Applying audio effects...', 0.8)
    
    ai_vocals_mixed = add_vocal_effects(
        ai_vocals_path, reverb_room_size, reverb_wet, reverb_dry, reverb_damping
    )
    
    # Pitch shift instrumental if needed
    if pitch_change_all != 0:
        if progress_callback:
            progress_callback('[~] Applying overall pitch change...', 0.85)
        instrumental_path = pitch_shift(instrumental_path, pitch_change_all)
    
    # Final mix
    if progress_callback:
        progress_callback('[~] Mixing final output...', 0.9)
    
    auto_mix(ai_vocals_mixed, instrumental_path, ai_cover_path,
             main_gain, inst_gain, output_format, backing_vocals_path)
    
    # Cleanup intermediate files
    if not keep_files:
        if progress_callback:
            progress_callback('[~] Cleaning up...', 0.95)
        _cleanup_intermediate_files(ai_vocals_mixed)
    
    return ai_cover_path


def _find_backing_vocals(song_dir: str) -> str | None:
    """Find cached backing vocals file"""
    for file in os.listdir(song_dir):
        file_lower = file.lower()
        if 'karaoke' in file_lower and '(vocals)' in file_lower and file.endswith('.wav'):
            return os.path.join(song_dir, file)
    return None


def _find_clean_instrumental(song_dir: str) -> str | None:
    """Find cached clean instrumental (from karaoke separation)"""
    for file in os.listdir(song_dir):
        file_lower = file.lower()
        if 'karaoke' in file_lower and '(instrumental)' in file_lower and file.endswith('.wav'):
            return os.path.join(song_dir, file)
    return None


def _cleanup_intermediate_files(*files):
    """Remove intermediate audio files"""
    for f in files:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
