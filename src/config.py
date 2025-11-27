"""
AICoverGen NextGen - Configuration
Centralized configuration and paths
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent

# Model directories
RVC_MODELS_DIR = BASE_DIR / "rvc_models"
MDX_MODELS_DIR = BASE_DIR / "mdxnet_models"
OUTPUT_DIR = BASE_DIR / "song_output"

# Ensure directories exist
RVC_MODELS_DIR.mkdir(exist_ok=True)
MDX_MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model filenames
HUBERT_MODEL = "hubert_base.pt"
RMVPE_MODEL = "rmvpe.pt"
MDX_VOCAL_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
DEREVERB_MODEL = "UVR-DeEcho-DeReverb.pth"
BS_ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# Karaoke model - separates backing vocals from instrumental for cleaner mix
MEL_BAND_ROFORMER_KARAOKE = "mel_band_roformer_karaoke_becruily.ckpt"

# Default RVC parameters
DEFAULT_INDEX_RATE = 0.5
DEFAULT_FILTER_RADIUS = 3
DEFAULT_RMS_MIX_RATE = 0.25
DEFAULT_PROTECT = 0.33
# Pitch detection methods: rmvpe, fcpe, hybrid, mangio-crepe
# rmvpe: clear vocals (default)
# fcpe: smoother, less robotic
# hybrid: best quality (rmvpe + fcpe combined)
# mangio-crepe: smoothest but slowest
DEFAULT_F0_METHOD = "rmvpe"
DEFAULT_CREPE_HOP_LENGTH = 128

# Embedder models for speaker embedding extraction
# contentvec: default, good quality
# hubert_base: alternative embedder
# custom: use custom embedder model
DEFAULT_EMBEDDER_MODEL = "contentvec"

# Noise reduction settings
DEFAULT_CLEAN_AUDIO = False
DEFAULT_CLEAN_STRENGTH = 0.7

# Default audio parameters
DEFAULT_REVERB_ROOM_SIZE = 0.15
DEFAULT_REVERB_WET = 0.2
DEFAULT_REVERB_DRY = 0.8
DEFAULT_REVERB_DAMPING = 0.7
DEFAULT_OUTPUT_FORMAT = "mp3"

# Device settings
DEVICE = "cuda:0"
IS_HALF = True

# Caching settings
CACHE_ENABLED = True
CACHE_DIR = OUTPUT_DIR / ".cache"


def get_model_path(model_name: str, model_dir: str = "rvc") -> Path:
    """Get full path to a model file"""
    if model_dir == "rvc":
        return RVC_MODELS_DIR / model_name
    elif model_dir == "mdx":
        return MDX_MODELS_DIR / model_name
    else:
        raise ValueError(f"Unknown model directory: {model_dir}")


def get_output_dir(song_id: str) -> Path:
    """Get output directory for a song"""
    output_path = OUTPUT_DIR / song_id
    output_path.mkdir(exist_ok=True)
    return output_path


# ============ Caching System ============

import hashlib
import json

def get_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
    return hashlib.blake2b(data.encode(), digest_size=8).hexdigest()


def get_file_hash(file_path: str, size: int = 8) -> str:
    """Get hash of a file for caching"""
    with open(file_path, 'rb') as f:
        file_hash = hashlib.blake2b(digest_size=size)
        for chunk in iter(lambda: f.read(4096), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def get_cached_path(song_dir: Path, prefix: str, cache_key: str, ext: str = '.wav') -> Path:
    """Get path for cached file"""
    return song_dir / f"{prefix}_{cache_key}{ext}"


def is_cached(cache_path: Path) -> bool:
    """Check if cached file exists"""
    return CACHE_ENABLED and cache_path.exists()


def save_cache_metadata(cache_path: Path, metadata: dict):
    """Save metadata for cached file"""
    json_path = cache_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def load_cache_metadata(cache_path: Path) -> dict | None:
    """Load metadata for cached file"""
    json_path = cache_path.with_suffix('.json')
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None
