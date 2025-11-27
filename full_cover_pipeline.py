"""
🎵 Full AI Cover Pipeline - One Script Does All!
1. Download from YouTube
2. Separate vocals (Mel-RoFormer SDR 12.6)
3. Denoise vocals (DeepFilterNet3)
4. Voice conversion (RVC)
5. Mix final output

Usage: python full_cover_pipeline.py <youtube_url> <rvc_model_folder>
"""

import os
import sys
import shutil
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "song_output"
RVC_MODELS_DIR = BASE_DIR / "rvc_models"

def download_youtube(url, output_path):
    """Download audio from YouTube in WAV format (better quality)"""
    print("\n" + "=" * 60)
    print("📥 Step 1: Downloading from YouTube (WAV for best quality)...")
    print("=" * 60)
    
    output_template = str(output_path / "input.%(ext)s")
    output_file = output_path / "input.wav"
    
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",  # WAV instead of MP3 (+0.3 SDR)
        "--audio-quality", "0",
        "-o", output_template,
        url
    ]
    
    subprocess.run(cmd, check=True)
    
    # Check if WAV exists, fallback to MP3
    if not output_file.exists():
        output_file = output_path / "input.mp3"
        if not output_file.exists():
            # Find any audio file
            for ext in ['wav', 'mp3', 'webm', 'opus', 'm4a']:
                f = output_path / f"input.{ext}"
                if f.exists():
                    output_file = f
                    break
    
    print(f"✓ Downloaded: {output_file}")
    return output_file


def separate_vocals(input_file, output_path):
    """Separate vocals using Mel-RoFormer (BEST SDR 12.6)"""
    print("\n" + "=" * 60)
    print("🎼 Step 2: Separating Vocals (Mel-RoFormer SDR 12.6)...")
    print("=" * 60)
    
    from audio_separator.separator import Separator
    
    separator = Separator(
        output_dir=str(output_path),
        output_format="wav"
    )
    
    # Use BEST model
    print("📥 Loading Mel-RoFormer (best quality)...")
    separator.load_model(model_filename="vocals_mel_band_roformer.ckpt")
    
    print(f"🎵 Separating: {input_file.name}")
    output_files = separator.separate(str(input_file))
    
    # Find vocals and instrumental
    vocals_file = None
    instrumental_file = None
    
    print(f"  Output files: {output_files}")
    
    output_path = Path(output_path)
    
    for f in output_files:
        f_lower = os.path.basename(f).lower()
        # Make sure we have full path
        f_path = Path(f)
        if not f_path.is_absolute():
            f_path = output_path / f_path
        
        # Vocals file contains "vocals" but NOT "other"
        if "(vocals)" in f_lower:
            vocals_file = f_path
        # Instrumental is "other" or "instrumental"
        elif "(other)" in f_lower or "(instrumental)" in f_lower:
            instrumental_file = f_path
    
    print(f"✓ Vocals: {vocals_file if vocals_file else 'Not found'}")
    print(f"✓ Instrumental: {instrumental_file if instrumental_file else 'Not found'}")
    
    return vocals_file, instrumental_file


def denoise_vocals(vocals_file, output_path, skip_denoise=False):
    """Denoise vocals using DeepFilterNet3"""
    print("\n" + "=" * 60)
    print("🔇 Step 3a: Denoising Vocals (DeepFilterNet3)...")
    print("=" * 60)
    
    if skip_denoise:
        print("⏭ Skipping denoise (disabled)")
        return vocals_file
    
    # Check if denoised file already exists
    output_file = output_path / "vocals_denoised.wav"
    if output_file.exists():
        print(f"✓ Using existing denoised file: {output_file.name}")
        return output_file
    
    try:
        from df import enhance, init_df
        from df.io import load_audio, save_audio
        
        print("📥 Loading DeepFilterNet3...")
        model, df_state, _ = init_df()
        
        print("🎵 Loading vocals...")
        audio, _ = load_audio(str(vocals_file), sr=df_state.sr())
        
        print("🔧 Applying noise reduction...")
        enhanced = enhance(model, df_state, audio)
        
        save_audio(str(output_file), enhanced, df_state.sr())
        
        print(f"✓ Denoised: {output_file.name}")
        return output_file
        
    except Exception as e:
        print(f"⚠ DeepFilterNet3 failed: {e}")
        print("  Using original vocals...")
        return vocals_file


def remove_reverb_echo(vocals_file, output_path, skip_dereverb=False):
    """Remove reverb and echo from vocals using UVR-DeEcho-DeReverb"""
    print("\n" + "=" * 60)
    print("🔇 Step 3b: Removing Reverb/Echo (UVR-DeEcho-DeReverb)...")
    print("=" * 60)
    
    if skip_dereverb:
        print("⏭ Skipping DeReverb (disabled)")
        return vocals_file
    
    # Check if model exists
    dereverb_model = BASE_DIR / "mdxnet_models" / "UVR-DeEcho-DeReverb.pth"
    if not dereverb_model.exists():
        print(f"⚠ DeReverb model not found, skipping")
        return vocals_file
    
    try:
        from audio_separator.separator import Separator
        
        print("📥 Loading UVR-DeEcho-DeReverb...")
        separator = Separator(
            output_dir=str(output_path),
            output_format="wav"
        )
        separator.load_model(model_filename="UVR-DeEcho-DeReverb.pth")
        
        print(f"🎵 Processing: {vocals_file.name}")
        output_files = separator.separate(str(vocals_file))
        
        # Find clean vocals (No Reverb)
        clean_vocals = None
        for f in output_files:
            f_lower = os.path.basename(f).lower()
            f_path = Path(f)
            if not f_path.is_absolute():
                f_path = output_path / f_path
            
            if 'no reverb' in f_lower or 'noreverb' in f_lower:
                clean_vocals = f_path
                break
        
        # Fallback
        if clean_vocals is None:
            for f in output_files:
                f_lower = os.path.basename(f).lower()
                f_path = Path(f)
                if not f_path.is_absolute():
                    f_path = output_path / f_path
                if 'reverb' not in f_lower or 'no' in f_lower:
                    clean_vocals = f_path
                    break
        
        if clean_vocals and clean_vocals.exists():
            print(f"✓ DeReverb complete: {clean_vocals.name}")
            return clean_vocals
        else:
            print("⚠ DeReverb output not found, using original")
            return vocals_file
            
    except Exception as e:
        print(f"⚠ DeReverb failed: {e}")
        return vocals_file


def convert_voice(vocals_file, rvc_model_path, output_path, pitch=0, f0_method="rmvpe"):
    """Convert voice using RVC"""
    print("\n" + "=" * 60)
    print(f"🎤 Step 4: Voice Conversion (RVC, pitch={pitch}, f0={f0_method})...")
    print("=" * 60)
    
    # Ensure vocals_file is absolute path
    vocals_file = Path(vocals_file)
    if not vocals_file.is_absolute():
        vocals_file = output_path / vocals_file
    
    print(f"📂 Vocals file: {vocals_file}")
    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_file}")
    
    # Add src to path
    sys.path.insert(0, str(BASE_DIR / "src"))
    
    from rvc import Config, load_hubert, get_vc, rvc_infer
    
    # Find model files
    model_path = Path(rvc_model_path)
    pth_file = None
    index_file = None
    
    for f in model_path.iterdir():
        if f.suffix == ".pth":
            pth_file = f
        elif f.suffix == ".index":
            index_file = f
    
    if not pth_file:
        raise FileNotFoundError(f"No .pth file found in {model_path}")
    
    print(f"📥 Loading RVC model: {pth_file.name}")
    
    # Setup
    device = "cuda:0"
    is_half = True
    
    config = Config(device, is_half)
    hubert_path = RVC_MODELS_DIR / "hubert_base.pt"
    
    print("📥 Loading Hubert...")
    hubert_model = load_hubert(device, is_half, str(hubert_path))
    
    print("📥 Loading voice model...")
    cpt, version, net_g, tgt_sr, vc = get_vc(device, is_half, config, str(pth_file))
    
    # Convert
    output_file = output_path / "vocals_converted.wav"
    
    print(f"🎵 Converting voice (pitch: {pitch})...")
    rvc_infer(
        index_path=str(index_file) if index_file else "",
        index_rate=0.5,
        input_path=str(vocals_file),
        output_path=str(output_file),
        pitch_change=pitch,
        f0_method=f0_method,  # rmvpe, fcpe, or hybrid
        cpt=cpt,
        version=version,
        net_g=net_g,
        filter_radius=3,
        tgt_sr=tgt_sr,
        rms_mix_rate=0.25,
        protect=0.33,
        crepe_hop_length=128,
        vc=vc,
        hubert_model=hubert_model
    )
    
    print(f"✓ Converted: {output_file.name}")
    return output_file


def mix_audio(vocals_file, instrumental_file, output_path, 
              vocal_gain=0, inst_gain=-2):
    """Mix vocals and instrumental with auto-normalization"""
    print("\n" + "=" * 60)
    print("🎛️ Step 5: Mixing Final Output (with normalization)...")
    print("=" * 60)
    
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    
    # Ensure paths are absolute
    vocals_file = Path(vocals_file)
    instrumental_file = Path(instrumental_file)
    output_path = Path(output_path)
    
    if not vocals_file.is_absolute():
        vocals_file = output_path / vocals_file
    if not instrumental_file.is_absolute():
        instrumental_file = output_path / instrumental_file
    
    print(f"📂 Vocals: {vocals_file}")
    print(f"📂 Instrumental: {instrumental_file}")
    
    if not vocals_file.exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals_file}")
    if not instrumental_file.exists():
        raise FileNotFoundError(f"Instrumental file not found: {instrumental_file}")
    
    print("📥 Loading audio files...")
    vocals = AudioSegment.from_wav(str(vocals_file))
    instrumental = AudioSegment.from_wav(str(instrumental_file))
    
    # Normalize both tracks
    print("🔧 Normalizing tracks...")
    vocals = normalize(vocals)
    instrumental = normalize(instrumental)
    
    # Apply gain (vocals slightly louder, instrumental slightly lower)
    vocals = vocals + vocal_gain
    instrumental = instrumental + inst_gain
    
    # Match lengths
    if len(vocals) > len(instrumental):
        vocals = vocals[:len(instrumental)]
    elif len(instrumental) > len(vocals):
        instrumental = instrumental[:len(vocals)]
    
    # Mix
    print("🎵 Mixing...")
    mixed = instrumental.overlay(vocals)
    
    # Final normalization
    mixed = normalize(mixed)
    
    # Export both MP3 and WAV
    output_mp3 = output_path / "final_cover.mp3"
    output_wav = output_path / "final_cover.wav"
    
    mixed.export(str(output_mp3), format="mp3", bitrate="320k")
    mixed.export(str(output_wav), format="wav")
    
    print(f"✓ Final output: {output_mp3}")
    print(f"✓ Lossless: {output_wav}")
    return output_mp3


def main():
    # Default values
    youtube_url = "https://www.youtube.com/watch?v=TB0R9bzODiE"
    rvc_model = str(RVC_MODELS_DIR / "HRK")
    pitch = 0
    
    # Parse args
    if len(sys.argv) >= 2:
        youtube_url = sys.argv[1]
    if len(sys.argv) >= 3:
        rvc_model = sys.argv[2]
    if len(sys.argv) >= 4:
        pitch = int(sys.argv[3])
    
    print("=" * 60)
    print("🎵 Full AI Cover Pipeline")
    print("=" * 60)
    print(f"YouTube: {youtube_url}")
    print(f"RVC Model: {rvc_model}")
    print(f"Pitch: {pitch}")
    print("=" * 60)
    
    # Create output directory
    import hashlib
    song_id = hashlib.md5(youtube_url.encode()).hexdigest()[:8]
    work_dir = OUTPUT_DIR / song_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Working directory: {work_dir}")
    
    try:
        # Check if files already exist (skip download/separation)
        existing_vocals = list(work_dir.glob("*_(vocals)_*.wav"))
        existing_inst = list(work_dir.glob("*_(other)_*.wav")) + list(work_dir.glob("*_(instrumental)_*.wav"))
        
        if existing_vocals and existing_inst:
            print("\n✓ Found existing separated files, skipping download & separation...")
            vocals_file = existing_vocals[0]
            instrumental_file = existing_inst[0]
            input_file = work_dir / "input.mp3"
        else:
            # Step 1: Download
            input_file = work_dir / "input.mp3"
            if not input_file.exists():
                input_file = download_youtube(youtube_url, work_dir)
            else:
                print(f"\n✓ Using existing: {input_file}")
            
            # Step 2: Separate
            vocals_file, instrumental_file = separate_vocals(input_file, work_dir)
        
        if not vocals_file or not instrumental_file:
            raise Exception("Separation failed! Check output files.")
        
        # Step 3a: Denoise (skip for now to test pipeline)
        denoised_vocals = denoise_vocals(vocals_file, work_dir, skip_denoise=True)
        
        # Step 3b: Remove Reverb/Echo
        clean_vocals = remove_reverb_echo(denoised_vocals, work_dir, skip_dereverb=False)
        
        # Step 4: Voice conversion
        converted_vocals = convert_voice(clean_vocals, rvc_model, work_dir, pitch)
        
        # Step 5: Mix
        final_output = mix_audio(converted_vocals, instrumental_file, work_dir)
        
        print("\n" + "=" * 60)
        print("✅ AI COVER COMPLETE!")
        print("=" * 60)
        print(f"\n🎵 Final output: {final_output}")
        print(f"📁 All files in: {work_dir}")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
