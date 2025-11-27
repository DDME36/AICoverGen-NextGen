"""
AICoverGen NextGen - Model Downloader
Downloads all required AI models for voice conversion
"""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent
RVC_MODELS_DIR = BASE_DIR / "rvc_models"
MDX_MODELS_DIR = BASE_DIR / "mdxnet_models"

# Create directories
RVC_MODELS_DIR.mkdir(exist_ok=True)
MDX_MODELS_DIR.mkdir(exist_ok=True)

# Model definitions
MODELS = {
    # ContentVec - Better feature extraction for RVC v2
    "hubert_base.pt": {
        "url": "https://huggingface.co/innnky/contentvec/resolve/main/checkpoint_best_legacy_500.pt",
        "dir": RVC_MODELS_DIR,
        "description": "ContentVec (Feature Extraction)"
    },
    # RMVPE - Best pitch detection
    "rmvpe.pt": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "dir": RVC_MODELS_DIR,
        "description": "RMVPE (Pitch Detection)"
    },
    # Note: FCPE uses torchfcpe library (installed via pip) which has bundled model
    # MDX-Net Vocal separation (fallback only)
    "UVR-MDX-NET-Voc_FT.onnx": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
        "dir": MDX_MODELS_DIR,
        "description": "MDX-Net Vocal (Fallback)"
    },
    # UVR DeEcho-DeReverb - Remove echo and reverb from vocals
    "UVR-DeEcho-DeReverb.pth": {
        "url": "https://huggingface.co/seanghay/uvr_models/resolve/main/UVR-DeEcho-DeReverb.pth",
        "dir": MDX_MODELS_DIR,
        "description": "UVR DeEcho-DeReverb (Vocal Cleanup)"
    },
    # Mel-Band RoFormer Instrumental - Separate backing vocals from instrumental
    # แยก instrumental (ไม่มี vocals) สำหรับ backing track
    "mel_band_roformer_instrumental_becruily.ckpt": {
        "url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt",
        "dir": MDX_MODELS_DIR,
        "description": "Mel-Band RoFormer Instrumental (Backing Vocal Removal)"
    },
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indicator"""
    if dest.exists():
        print(f"  ✓ {desc} (already exists)")
        return True
    
    print(f"  ↓ Downloading {desc}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ {desc}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {desc}: {e}")
        return False


def get_separator_model_dir():
    """Get the audio-separator model directory"""
    import tempfile
    # audio-separator uses /tmp/audio-separator-models/ by default
    return Path(tempfile.gettempdir()) / "audio-separator-models"


def pre_download_separator_models():
    """
    Pre-download audio-separator models to avoid slow first-run
    Downloads custom models directly to audio-separator's cache directory
    """
    print()
    print("[~] Pre-downloading audio-separator models...")
    print("    (This may take a few minutes on first run)")
    print()
    
    # Get audio-separator model directory
    separator_model_dir = get_separator_model_dir()
    separator_model_dir.mkdir(exist_ok=True)
    print(f"    Model directory: {separator_model_dir}")
    print()
    
    # Custom models to download directly (not in audio-separator's repo)
    custom_models = {
        "mel_band_roformer_instrumental_becruily.ckpt": {
            "url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt",
            "description": "Mel-Band RoFormer Instrumental (Backing Vocal Removal)"
        }
    }
    
    # Download custom models to audio-separator directory
    for filename, info in custom_models.items():
        dest = separator_model_dir / filename
        download_file(info["url"], dest, info["description"])
    
    # Pre-load standard models via audio-separator
    try:
        from audio_separator.separator import Separator
        
        # Standard models that audio-separator can download
        separator_models = [
            ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "BS-RoFormer (Vocal Separation)"),
            ("UVR-DeEcho-DeReverb.pth", "DeEcho-DeReverb (Vocal Cleanup)"),
        ]
        
        # Create separator instance
        separator = Separator(output_format="wav")
        
        for model_name, description in separator_models:
            print(f"  ↓ Loading {description}...")
            try:
                separator.load_model(model_filename=model_name)
                print(f"  ✓ {description}")
            except Exception as e:
                print(f"  ⚠️  {description}: {e}")
        
        print()
        print("✅ Audio separator models ready!")
        
    except ImportError:
        print("⚠️  audio-separator not installed, skipping standard model pre-download")
        print("   Install with: pip install audio-separator[gpu]")
    except Exception as e:
        print(f"⚠️  Error pre-downloading separator models: {e}")


def main():
    print("=" * 60)
    print("🎵 AICoverGen NextGen - Model Downloader")
    print("=" * 60)
    print()
    
    success_count = 0
    total_count = len(MODELS)
    
    for filename, info in MODELS.items():
        dest = info["dir"] / filename
        if download_file(info["url"], dest, info["description"]):
            success_count += 1
    
    print()
    print("=" * 60)
    if success_count == total_count:
        print(f"✅ All {total_count} RVC models downloaded!")
    else:
        print(f"⚠️  {success_count}/{total_count} RVC models downloaded")
    print("=" * 60)
    
    # Pre-download audio-separator models
    pre_download_separator_models()
    
    print()
    print("=" * 60)
    print("🎉 Setup complete! You can now run the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
