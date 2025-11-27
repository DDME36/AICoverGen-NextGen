"""
AICoverGen NextGen - Pitch Extraction
FCPE (Fast Context-aware Pitch Estimation) implementation
"""

import numpy as np
import torch
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try to import torchfcpe (official FCPE implementation)
TORCHFCPE_AVAILABLE = False
try:
    from torchfcpe import spawn_bundled_infer_model
    TORCHFCPE_AVAILABLE = True
except ImportError:
    pass


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
        self.use_torchfcpe = False
        
        # Try torchfcpe first (official implementation)
        if TORCHFCPE_AVAILABLE:
            try:
                self.model = spawn_bundled_infer_model(device=device)
                self.use_torchfcpe = True
                print("✓ FCPE loaded (torchfcpe)")
                return
            except Exception as e:
                print(f"⚠️  torchfcpe failed: {e}")
        
        # Fallback message
        print("⚠️  FCPE not available, will use librosa pyin fallback")
    
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
        if self.use_torchfcpe and self.model is not None:
            return self._compute_torchfcpe(audio, p_len, f0_min, f0_max)
        else:
            return self._fallback_f0(audio, p_len, f0_min, f0_max)
    
    def _compute_torchfcpe(self, audio, p_len, f0_min, f0_max):
        """Compute F0 using torchfcpe"""
        try:
            # Prepare audio tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(-1).to(self.device)
            
            # Run inference
            with torch.no_grad():
                f0 = self.model.infer(
                    audio_tensor,
                    sr=self.sampling_rate,
                    decoder_mode="local_argmax",
                    threshold=0.006
                )
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
            print(f"⚠️  torchfcpe inference failed: {e}")
            return self._fallback_f0(audio, p_len, f0_min, f0_max)
    
    def _fallback_f0(self, audio, p_len, f0_min, f0_max):
        """Fallback pitch detection using librosa pyin"""
        import librosa
        
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
        except Exception as e:
            print(f"⚠️  Fallback f0 failed: {e}")
            return np.zeros(p_len if p_len else len(audio) // self.hop_length)


class HybridPitchExtractor:
    """
    Hybrid pitch extraction combining RMVPE and FCPE
    Takes median of both for more robust results
    """
    
    def __init__(self, device="cuda", is_half=True):
        self.device = device
        self.is_half = is_half
        self.rmvpe = None
        self.fcpe = None
    
    def _load_rmvpe(self):
        if self.rmvpe is None:
            from rmvpe import RMVPE
            rmvpe_path = os.path.join(BASE_DIR, 'rvc_models', 'rmvpe.pt')
            self.rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)
    
    def _load_fcpe(self):
        if self.fcpe is None:
            self.fcpe = FCPE(device=self.device)
    
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
            target_len = len(results[0]) if results else p_len
            f0_fcpe = self.fcpe.compute_f0(audio, p_len=target_len)
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
