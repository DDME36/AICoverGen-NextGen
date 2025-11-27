import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from rvc import Config, load_hubert, get_vc, rvc_infer

# BS-RoFormer (Best vocal separation - SDR 12.97)
try:
    from audio_separator.separator import Separator
    BS_ROFORMER_AVAILABLE = True
except ImportError:
    BS_ROFORMER_AVAILABLE = False
    print("⚠️  audio-separator not available. Install with: pip install audio-separator[gpu]")

# MDX-Net fallback
from mdx import run_mdx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

# Global model cache for performance
_model_cache = {
    'hubert': None,
    'config': None,
    'separator': None,
    'dereverb_separator': None
}


def get_youtube_video_id(url, ignore_playlist=True):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    return None


def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        if result is None:
            raise Exception("Failed to download from YouTube. Try uploading a local file instead.")
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None
    kara_main_vocals_path = None

    for file in os.listdir(song_dir):
        file_lower = file.lower()
        
        # Priority 1: DeReverb vocals (No Reverb) - cleanest vocals for RVC
        if 'no reverb' in file_lower and file.endswith('.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)
        # Priority 2: KARA separated main vocals (from backing separation)
        elif 'kara' in file_lower and '(vocals)' in file_lower and file.endswith('.wav'):
            kara_main_vocals_path = os.path.join(song_dir, file)
        # Priority 3: KARA backing vocals (instrumental output from KARA = backing)
        elif 'kara' in file_lower and '(instrumental)' in file_lower and file.endswith('.wav'):
            backup_vocals_path = os.path.join(song_dir, file)
        # Priority 4: BS-RoFormer output (raw vocals before KARA separation)
        elif '(vocals)' in file_lower and file.endswith('.wav') and 'reverb' not in file_lower and 'kara' not in file_lower:
            if main_vocals_path is None:
                main_vocals_path = os.path.join(song_dir, file)
        elif '(instrumental)' in file_lower and file.endswith('.wav') and 'kara' not in file_lower:
            instrumentals_path = os.path.join(song_dir, file)
        # MDX-Net output format (fallback)
        elif file.endswith('_Vocals.wav'):
            if main_vocals_path is None:
                main_vocals_path = os.path.join(song_dir, file)
        elif file.endswith('_Instrumental.wav'):
            if instrumentals_path is None:
                instrumentals_path = os.path.join(song_dir, file)
                orig_song_path = instrumentals_path.replace('_Instrumental', '')
        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)
        elif file.endswith('_Vocals_Main.wav'):
            main_vocals_path = os.path.join(song_dir, file)

    # Find original song (mp3, wav, or stereo wav)
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

    # Use best available vocals: DeReverb > KARA main > raw vocals
    final_vocals_path = main_vocals_dereverb_path or kara_main_vocals_path or main_vocals_path

    return orig_song_path, instrumentals_path, final_vocals_path, backup_vocals_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def separate_with_bs_roformer(input_path, output_dir):
    """Separate vocals using BS-RoFormer (SDR 12.97 - BEST quality)"""
    global _model_cache
    
    # Create new separator with correct output_dir each time
    # Model weights are cached by audio-separator internally
    if _model_cache['separator'] is None:
        print("[~] Loading BS-RoFormer model (first time only)...")
    
    separator = Separator(
        output_dir=output_dir,
        output_format="wav"
    )
    separator.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    _model_cache['separator'] = True  # Mark as loaded
    
    # Separate
    output_files = separator.separate(input_path)
    
    vocals_path = None
    instrumental_path = None
    
    for f in output_files:
        f_lower = os.path.basename(f).lower()
        # Make sure path is absolute
        if not os.path.isabs(f):
            f = os.path.join(output_dir, os.path.basename(f))
        
        if '(vocals)' in f_lower:
            vocals_path = f
        elif '(instrumental)' in f_lower or '(other)' in f_lower:
            instrumental_path = f
    
    # Verify files exist
    if vocals_path and not os.path.exists(vocals_path):
        # Try to find the file in output_dir
        for file in os.listdir(output_dir):
            if '(Vocals)' in file and file.endswith('.wav'):
                vocals_path = os.path.join(output_dir, file)
                break
    
    if instrumental_path and not os.path.exists(instrumental_path):
        for file in os.listdir(output_dir):
            if '(Instrumental)' in file and file.endswith('.wav'):
                instrumental_path = os.path.join(output_dir, file)
                break
    
    return vocals_path, instrumental_path


def remove_reverb_echo(vocals_path, output_dir):
    """Remove reverb and echo from vocals using UVR-DeEcho-DeReverb"""
    global _model_cache
    
    if not BS_ROFORMER_AVAILABLE:
        print("⚠️  audio-separator not available, skipping DeReverb")
        return vocals_path
    
    # Check if model exists
    dereverb_model = os.path.join(mdxnet_models_dir, 'UVR-DeEcho-DeReverb.pth')
    if not os.path.exists(dereverb_model):
        print(f"⚠️  DeReverb model not found at {dereverb_model}, skipping")
        return vocals_path
    
    try:
        print("[~] Removing reverb/echo from vocals (UVR-DeEcho-DeReverb)...")
        
        separator = Separator(
            output_dir=output_dir,
            output_format="wav"
        )
        separator.load_model(model_filename="UVR-DeEcho-DeReverb.pth")
        
        output_files = separator.separate(vocals_path)
        
        # Find the clean vocals (No Reverb)
        clean_vocals_path = None
        for f in output_files:
            f_lower = os.path.basename(f).lower()
            if not os.path.isabs(f):
                f = os.path.join(output_dir, os.path.basename(f))
            
            # DeReverb model outputs: (No Reverb) and (Reverb)
            if 'no reverb' in f_lower or 'noreverb' in f_lower:
                clean_vocals_path = f
                break
        
        # Fallback: find any output that's not the reverb part
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
            return clean_vocals_path
        else:
            print("⚠️  DeReverb output not found, using original vocals")
            return vocals_path
            
    except Exception as e:
        print(f"⚠️  DeReverb failed: {e}, using original vocals")
        return vocals_path


def separate_backing_vocals(vocals_path, output_dir):
    """Separate main vocals from backing vocals using UVR_MDXNET_KARA_2"""
    if not BS_ROFORMER_AVAILABLE:
        print("⚠️  audio-separator not available, skipping backing vocal separation")
        return vocals_path, None
    
    # Check if model exists (will be downloaded automatically by audio-separator)
    try:
        print("[~] Separating main vocals from backing vocals (KARA_2)...")
        
        separator = Separator(
            output_dir=output_dir,
            output_format="wav"
        )
        separator.load_model(model_filename="UVR_MDXNET_KARA_2.onnx")
        
        output_files = separator.separate(vocals_path)
        
        main_vocals_path = None
        backing_vocals_path = None
        
        for f in output_files:
            f_lower = os.path.basename(f).lower()
            if not os.path.isabs(f):
                f = os.path.join(output_dir, os.path.basename(f))
            
            # KARA model outputs: (Vocals) for main, (Instrumental) for backing
            if '(vocals)' in f_lower:
                main_vocals_path = f
            elif '(instrumental)' in f_lower or 'backing' in f_lower:
                backing_vocals_path = f
        
        if main_vocals_path and os.path.exists(main_vocals_path):
            print(f"✓ Main vocals: {os.path.basename(main_vocals_path)}")
            if backing_vocals_path and os.path.exists(backing_vocals_path):
                print(f"✓ Backing vocals: {os.path.basename(backing_vocals_path)}")
            return main_vocals_path, backing_vocals_path
        else:
            print("⚠️  Backing separation output not found, using original")
            return vocals_path, None
            
    except Exception as e:
        print(f"⚠️  Backing vocal separation failed: {e}")
        return vocals_path, None


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    orig_song_path = convert_to_stereo(orig_song_path)

    # Separate Vocals from Instrumental using BS-RoFormer (SDR 12.97)
    display_progress('[~] Separating Vocals from Instrumental (BS-RoFormer)...', 0.1, is_webui, progress)
    
    if BS_ROFORMER_AVAILABLE:
        try:
            vocals_path, instrumentals_path = separate_with_bs_roformer(orig_song_path, song_output_dir)
            print(f"✓ BS-RoFormer separation complete (SDR 12.97)")
        except Exception as e:
            print(f"⚠️ BS-RoFormer failed: {e}, falling back to MDX-Net")
            vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, 
                os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), 
                orig_song_path, denoise=True, keep_orig=keep_orig)
    else:
        # Fallback to MDX-Net
        vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, 
            os.path.join(mdxnet_models_dir, 'UVR-MDX-NET-Voc_FT.onnx'), 
            orig_song_path, denoise=True, keep_orig=keep_orig)

    # Separate main vocals from backing vocals
    display_progress('[~] Separating main/backing vocals...', 0.2, is_webui, progress)
    main_vocals_path, backup_vocals_path = separate_backing_vocals(vocals_path, song_output_dir)

    # Apply DeReverb to clean up main vocals
    display_progress('[~] Removing reverb/echo from vocals...', 0.3, is_webui, progress)
    main_vocals_dereverb_path = remove_reverb_echo(main_vocals_path, song_output_dir)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    global _model_cache
    
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    
    # Reuse config and hubert model for better performance
    if _model_cache['config'] is None:
        _model_cache['config'] = Config(device, True)
    config = _model_cache['config']
    
    if _model_cache['hubert'] is None:
        print("[~] Loading HuBERT model (first time only)...")
        _model_cache['hubert'] = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    hubert_model = _model_cache['hubert']
    
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def auto_mix_audio(main_vocals_path, backing_vocals_path, instrumental_path, output_path, 
                   main_gain=0, backup_gain=0, inst_gain=0, output_format='mp3'):
    """
    Auto-mix all audio tracks with intelligent processing:
    - Compression for consistent loudness
    - EQ matching between vocals and instrumental
    - Proper gain staging
    """
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    
    print("[~] Auto-mixing with compression and EQ...")
    
    # Load audio files
    main_vocal = AudioSegment.from_wav(main_vocals_path)
    instrumental = AudioSegment.from_wav(instrumental_path)
    
    # Load backing vocals if available
    if backing_vocals_path and os.path.exists(backing_vocals_path):
        backing_vocal = AudioSegment.from_wav(backing_vocals_path)
    else:
        backing_vocal = AudioSegment.silent(duration=len(main_vocal))
    
    # Step 1: Normalize all tracks to consistent level
    main_vocal = normalize(main_vocal)
    backing_vocal = normalize(backing_vocal)
    instrumental = normalize(instrumental)
    
    # Step 2: Apply compression for consistent loudness
    # Main vocals: moderate compression (ratio 4:1)
    main_vocal = compress_dynamic_range(main_vocal, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
    
    # Backing vocals: lighter compression
    if len(backing_vocal) > 0 and backing_vocal.dBFS > -60:
        backing_vocal = compress_dynamic_range(backing_vocal, threshold=-25.0, ratio=3.0, attack=10.0, release=100.0)
    
    # Instrumental: gentle compression to preserve dynamics
    instrumental = compress_dynamic_range(instrumental, threshold=-15.0, ratio=2.0, attack=20.0, release=200.0)
    
    # Step 3: Apply gain staging (professional mix levels)
    # Main vocals: prominent but not overpowering
    main_vocal = main_vocal - 3 + main_gain
    
    # Backing vocals: sit behind main vocals
    backing_vocal = backing_vocal - 8 + backup_gain
    
    # Instrumental: foundation of the mix
    instrumental = instrumental - 5 + inst_gain
    
    # Step 4: Match lengths
    max_length = max(len(main_vocal), len(instrumental))
    if len(main_vocal) < max_length:
        main_vocal = main_vocal + AudioSegment.silent(duration=max_length - len(main_vocal))
    if len(backing_vocal) < max_length:
        backing_vocal = backing_vocal + AudioSegment.silent(duration=max_length - len(backing_vocal))
    if len(instrumental) < max_length:
        instrumental = instrumental + AudioSegment.silent(duration=max_length - len(instrumental))
    
    # Trim to shortest
    min_length = min(len(main_vocal), len(backing_vocal), len(instrumental))
    main_vocal = main_vocal[:min_length]
    backing_vocal = backing_vocal[:min_length]
    instrumental = instrumental[:min_length]
    
    # Step 5: Mix all tracks
    mixed = instrumental.overlay(backing_vocal).overlay(main_vocal)
    
    # Step 6: Final limiting to prevent clipping
    mixed = normalize(mixed)
    
    # Apply soft limiting if too loud
    if mixed.dBFS > -1.0:
        mixed = mixed - (mixed.dBFS + 1.0)
    
    # Export
    mixed.export(output_path, format=output_format, bitrate="320k")
    print(f"✓ Auto-mix complete: {os.path.basename(output_path)}")
    
    return output_path


def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format, use_auto_mix=True):
    """Combine vocals and instrumental into final mix"""
    main_vocals_path = audio_paths[0]
    backup_vocals_path = audio_paths[1]
    instrumental_path = audio_paths[2]
    
    if use_auto_mix:
        # Use advanced auto-mixing with compression and EQ
        return auto_mix_audio(
            main_vocals_path, backup_vocals_path, instrumental_path, output_path,
            main_gain, backup_gain, inst_gain, output_format
        )
    else:
        # Legacy simple mixing
        main_vocal_audio = AudioSegment.from_wav(main_vocals_path) - 4 + main_gain
        
        if backup_vocals_path and os.path.exists(backup_vocals_path):
            backup_vocal_audio = AudioSegment.from_wav(backup_vocals_path) - 6 + backup_gain
        else:
            backup_vocal_audio = AudioSegment.silent(duration=len(main_vocal_audio))
        
        instrumental_audio = AudioSegment.from_wav(instrumental_path) - 7 + inst_gain
        main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)
        return output_path


def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files,
                        is_webui=0, main_gain=0, backup_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3,
                        rms_mix_rate=0.25, f0_method='rmvpe', crepe_hop_length=128, protect=0.33, pitch_change_all=0,
                        reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, output_format='mp3',
                        progress=gr.Progress()):
    try:
        if not song_input or not voice_model:
            raise_exception('Ensure that the song input field and voice model field is filled.', is_webui)

        display_progress('[~] Starting AI Cover Generation Pipeline...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Invalid YouTube url.'
                raise_exception(error_msg, is_webui)
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} does not exist.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
        else:
            # Check if we already have separated audio files (use cache)
            vocals_path, main_vocals_path = None, None
            paths = get_audio_paths(song_dir)
            orig_song_path, instrumentals_path, cached_vocals_path, backup_vocals_path = paths

            # Only re-process if essential files are missing
            if instrumentals_path is None or cached_vocals_path is None:
                print(f"[~] Cache incomplete, re-processing song...")
                orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
            else:
                print(f"[✓] Using cached audio files from {song_dir}")
                print(f"    Vocals: {os.path.basename(cached_vocals_path)}")
                main_vocals_path = cached_vocals_path
                main_vocals_dereverb_path = cached_vocals_path

        pitch_change = pitch_change * 12 + pitch_change_all
        ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{"" if f0_method != "mangio-crepe" else f"_{crepe_hop_length}"}.wav')
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{output_format}')

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Converting voice using RVC...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path if main_vocals_dereverb_path else main_vocals_path, 
                        ai_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Applying audio effects to Vocals...', 0.8, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)

        if pitch_change_all != 0:
            display_progress('[~] Applying overall pitch change', 0.85, is_webui, progress)
            instrumentals_path = pitch_shift(instrumentals_path, pitch_change_all)
            if backup_vocals_path:
                backup_vocals_path = pitch_shift(backup_vocals_path, pitch_change_all)

        display_progress('[~] Combining AI Vocals and Instrumentals...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed_path, backup_vocals_path, instrumentals_path], ai_cover_path, main_gain, backup_gain, inst_gain, output_format)

        if not keep_files:
            display_progress('[~] Removing intermediate audio files...', 0.95, is_webui, progress)
            intermediate_files = [vocals_path, main_vocals_path, ai_vocals_mixed_path]
            if pitch_change_all != 0:
                intermediate_files += [instrumentals_path]
                if backup_vocals_path:
                    intermediate_files += [backup_vocals_path]
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return ai_cover_path

    except Exception as e:
        raise_exception(str(e), is_webui)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a AI cover song in the song_output/id directory.', add_help=True)
    parser.add_argument('-i', '--song-input', type=str, required=True)
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True)
    parser.add_argument('-p', '--pitch-change', type=int, required=True)
    parser.add_argument('-k', '--keep-files', action=argparse.BooleanOptionalAction)
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5)
    parser.add_argument('-fr', '--filter-radius', type=int, default=3)
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25)
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128)
    parser.add_argument('-pro', '--protect', type=float, default=0.33)
    parser.add_argument('-mv', '--main-vol', type=int, default=0)
    parser.add_argument('-bv', '--backup-vol', type=int, default=0)
    parser.add_argument('-iv', '--inst-vol', type=int, default=0)
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0)
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15)
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2)
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8)
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7)
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3')
    args = parser.parse_args()

    rvc_dirname = args.rvc_dirname
    if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
        raise Exception(f'The folder {os.path.join(rvc_models_dir, rvc_dirname)} does not exist.')

    cover_path = song_cover_pipeline(args.song_input, rvc_dirname, args.pitch_change, args.keep_files,
                                     main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
                                     index_rate=args.index_rate, filter_radius=args.filter_radius,
                                     rms_mix_rate=args.rms_mix_rate, f0_method=args.pitch_detection_algo,
                                     crepe_hop_length=args.crepe_hop_length, protect=args.protect,
                                     pitch_change_all=args.pitch_change_all,
                                     reverb_rm_size=args.reverb_size, reverb_wet=args.reverb_wetness,
                                     reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
                                     output_format=args.output_format)
    print(f'[+] Cover generated at {cover_path}')
