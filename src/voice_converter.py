"""
AICoverGen NextGen - Voice Converter
RVC voice conversion with optimizations
"""

import gc
import os
import torch

# Handle both module and script execution
try:
    from . import config
    from .rvc import Config, load_hubert, get_vc, rvc_infer
except ImportError:
    import config
    from rvc import Config, load_hubert, get_vc, rvc_infer

# Global model cache
_model_cache = {
    'hubert': None,
    'config': None,
}

# Enable PyTorch optimizations
def _enable_optimizations():
    """Enable PyTorch performance optimizations"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmark for faster convolutions
        torch.backends.cudnn.benchmark = True
        # Enable TF32 for faster matrix operations on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

_enable_optimizations()


def get_rvc_model_files(voice_model: str) -> tuple[str, str]:
    """Get .pth and .index files for a voice model"""
    model_dir = config.RVC_MODELS_DIR / voice_model
    
    pth_file = None
    index_file = None
    
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            pth_file = str(model_dir / file)
        if ext == '.index':
            index_file = str(model_dir / file)

    if pth_file is None:
        raise FileNotFoundError(f'No .pth model file in {model_dir}')

    return pth_file, index_file or ''


def convert_voice(voice_model: str, input_path: str, output_path: str,
                  pitch_change: int = 0, f0_method: str = None,
                  index_rate: float = None, filter_radius: int = None,
                  rms_mix_rate: float = None, protect: float = None,
                  crepe_hop_length: int = None) -> str:
    """
    Convert voice using RVC
    """
    global _model_cache
    
    # Use defaults
    f0_method = f0_method or config.DEFAULT_F0_METHOD
    index_rate = index_rate if index_rate is not None else config.DEFAULT_INDEX_RATE
    filter_radius = filter_radius if filter_radius is not None else config.DEFAULT_FILTER_RADIUS
    rms_mix_rate = rms_mix_rate if rms_mix_rate is not None else config.DEFAULT_RMS_MIX_RATE
    protect = protect if protect is not None else config.DEFAULT_PROTECT
    crepe_hop_length = crepe_hop_length if crepe_hop_length is not None else config.DEFAULT_CREPE_HOP_LENGTH
    
    # Get model files
    rvc_model_path, rvc_index_path = get_rvc_model_files(voice_model)
    
    # Setup config and hubert (cached)
    if _model_cache['config'] is None:
        _model_cache['config'] = Config(config.DEVICE, config.IS_HALF)
    rvc_config = _model_cache['config']
    
    if _model_cache['hubert'] is None:
        print("[~] Loading HuBERT model (first time only)...")
        hubert_path = config.get_model_path(config.HUBERT_MODEL, "rvc")
        _model_cache['hubert'] = load_hubert(config.DEVICE, config.IS_HALF, str(hubert_path))
    hubert_model = _model_cache['hubert']
    
    # Load voice model
    cpt, version, net_g, tgt_sr, vc = get_vc(
        config.DEVICE, config.IS_HALF, rvc_config, rvc_model_path
    )

    # Run inference
    print(f"[~] Converting voice (pitch: {pitch_change}, method: {f0_method})...")
    rvc_infer(
        rvc_index_path, index_rate, input_path, output_path,
        pitch_change, f0_method, cpt, version, net_g,
        filter_radius, tgt_sr, rms_mix_rate, protect,
        crepe_hop_length, vc, hubert_model
    )
    
    print(f"✓ Voice conversion complete")
    
    # Cleanup memory
    del cpt, net_g, vc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def list_voice_models() -> list[str]:
    """List available voice models"""
    models = []
    exclude = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt', 'fcpe.pt']
    
    for item in os.listdir(config.RVC_MODELS_DIR):
        if item not in exclude:
            models.append(item)
    
    return models
