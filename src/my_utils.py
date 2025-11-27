import numpy as np
import os


def load_audio(file, sr):
    """Load audio file and resample to target sample rate"""
    # Convert Path to string if needed
    file = str(file)
    file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
    
    # Check if file exists
    if not os.path.exists(file):
        raise RuntimeError(f"Audio file not found: {file}")
    
    # Try librosa first (more reliable with various formats)
    try:
        import librosa
        audio, _ = librosa.load(file, sr=sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e1:
        pass
    
    # Fallback to soundfile
    try:
        import soundfile as sf
        audio, orig_sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return audio.astype(np.float32)
    except Exception as e2:
        pass
    
    # Last resort: ffmpeg
    try:
        import ffmpeg
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.float32).flatten()
    except Exception as e3:
        raise RuntimeError(f"Failed to load audio with all methods. File: {file}")
