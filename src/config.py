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

# Default audio parameters
DEFAULT_REVERB_ROOM_SIZE = 0.15
DEFAULT_REVERB_WET = 0.2
DEFAULT_REVERB_DRY = 0.8
DEFAULT_REVERB_DAMPING = 0.7
DEFAULT_OUTPUT_FORMAT = "mp3"

# Device settings
DEVICE = "cuda:0"
IS_HALF = True


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
