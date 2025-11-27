"""
AICoverGen NextGen - Pitch Extraction
FCPE (Fast Context-aware Pitch Estimation) implementation
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FCPE:
    """
    FCPE - Fast Context-aware Pitch Estimation
    Smoother pitch detection than RMVPE, less robotic artifacts
    """
    
    def __init__(self, model_path=None, device="cuda", hop_length=160, sampling_rate=16000):
        self.device = device
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.model = None
        
        if model_path is None:
            model_path = os.path.join(BASE_DIR, 'rvc_models', 'fcpe.pt')
        
        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"⚠️  FCPE model not found at {model_path}")
            print("   Will use fallback pitch detection")
    
    def _load_model(self, model_path):
        """Load FCPE model"""
        try:
            # FCPE uses a simple CNN architecture
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print("✓ FCPE model loaded")
        except Exception as e:
            print(f"⚠️  Failed to load FCPE model: {e}")
            self.model = None
    
    def compute_f0(self, audio, p_len=None, f0_min=50, f0_max=1100):
        """
        Compute F0 using FCPE
        
        Args:
            audio: numpy array of audio samples (16kHz)
            p_len: expected output length
            f0_min: minimum F0 frequency
            f0_max: maximum F0 frequency
            
        Returns:
            f0: numpy array of F0 values
        """
        if self.model is None:
            # Fallback to simple pitch detection
            return self._fallback_f0(audio, p_len, f0_min, f0_max)
        
        try:
            # Prepare audio tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                f0 = self.model(audio_tensor, self.sampling_rate, self.hop_length)
                f0 = f0.squeeze().cpu().numpy()
            
            # Post-process
            f0[f0 < f0_min] = 0
            f0[f0 > f0_max] = 0
            
            # Interpolate to target length if needed
            if p_len is not None and len(f0) != p_len:
                f0 = np.interp(
                    np.linspace(0, len(f0), p_len),
                    np.arange(len(f0)),
                    f0
                )
            
            return f0
            
        except Exception as e:
            print(f"⚠️  FCPE inference failed: {e}")
            return self._fallback_f0(audio, p_len, f0_min, f0_max)
    
    def _fallback_f0(self, audio, p_len, f0_min, f0_max):
        """Fallback pitch detection using autocorrelation"""
        import librosa
        
        # Use librosa's pyin for fallback
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio.astype(np.float32),
                fmin=f0_min,
                fmax=f0_max,
                sr=self.sampling_rate,
                hop_length=self.hop_length
            )
            f0 = np.nan_to_num(f0)
            
            if p_len is not None and len(f0) != p_len:
                f0 = np.interp(
                    np.linspace(0, len(f0), p_len),
                    np.arange(len(f0)),
                    f0
                )
            
            return f0
        except:
            # Ultimate fallback - zeros
            return np.zeros(p_len if p_len else len(audio) // self.hop_length)


class HybridPitchExtractor:
    """
    Hybrid pitch extraction combining RMVPE and FCPE
    Takes median of both for more robust results
    """
    
    def __init__(self, device="cuda"):
        self.device = device
        self.rmvpe = None
        self.fcpe = None
    
    def _load_rmvpe(self):
        if self.rmvpe is None:
            from rmvpe import RMVPE
            rmvpe_path = os.path.join(BASE_DIR, 'rvc_models', 'rmvpe.pt')
            self.rmvpe = RMVPE(rmvpe_path, is_half=True, device=self.device)
    
    def _load_fcpe(self):
        if self.fcpe is None:
            fcpe_path = os.path.join(BASE_DIR, 'rvc_models', 'fcpe.pt')
            self.fcpe = FCPE(fcpe_path, device=self.device)
    
    def compute_f0(self, audio, p_len=None, thred=0.03):
        """
        Compute F0 using hybrid RMVPE+FCPE
        """
        results = []
        
        # Try RMVPE
        try:
            self._load_rmvpe()
            f0_rmvpe = self.rmvpe.infer_from_audio(audio, thred=thred)
            results.append(f0_rmvpe)
        except Exception as e:
            print(f"  RMVPE failed: {e}")
        
        # Try FCPE
        try:
            self._load_fcpe()
            f0_fcpe = self.fcpe.compute_f0(audio, p_len=len(results[0]) if results else p_len)
            results.append(f0_fcpe)
        except Exception as e:
            print(f"  FCPE failed: {e}")
        
        if not results:
            raise RuntimeError("Both RMVPE and FCPE failed")
        
        if len(results) == 1:
            return results[0]
        
        # Align lengths
        min_len = min(len(r) for r in results)
        results = [r[:min_len] for r in results]
        
        # Take median (more robust than mean)
        f0 = np.nanmedian(results, axis=0)
        
        return f0
