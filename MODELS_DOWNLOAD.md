# 📥 Models Download Guide

## 🚀 Quick Setup (Recommended)

```bash
python setup_models.py
```

This will download all required models automatically.

---

## 📦 Required Models

### 1. Hubert Base (RVC Encoder)
- **File**: `hubert_base.pt`
- **Size**: ~181 MB
- **Location**: `rvc_models/`
- **Download**: [Hugging Face](https://huggingface.co/IAHispano/Applio/resolve/main/Resources/hubert_base.pt)

### 2. RMVPE (Pitch Detection)
- **File**: `rmvpe.pt`
- **Size**: ~181 MB
- **Location**: `rvc_models/`
- **Download**: [Hugging Face](https://huggingface.co/IAHispano/Applio/resolve/main/Resources/rmvpe.pt)

---

## 📦 Auto-Download Models

These models are downloaded automatically by `audio-separator` on first use:

### Mel-RoFormer (Vocal Separation)
- **File**: `vocals_mel_band_roformer.ckpt`
- **Size**: ~200 MB
- **Quality**: SDR 12.6 (Best)
- **Note**: Auto-downloads to `~/.cache/audio-separator/`

---

## 📦 Optional Models

### FCPE (Fast Pitch Detection)
- **File**: `fcpe.pt`
- **Size**: ~41 MB
- **Location**: `rvc_models/`
- **Download**: [Hugging Face](https://huggingface.co/IAHispano/Applio/resolve/main/Resources/fcpe.pt)

---

## 📂 Folder Structure

```
AICoverGen-NextGen/
├── rvc_models/
│   ├── hubert_base.pt      ← Required
│   ├── rmvpe.pt            ← Required
│   ├── fcpe.pt             ← Optional
│   └── YOUR_MODEL/         ← Your RVC models
│       ├── model.pth
│       └── model.index
├── separation_models/      ← Auto-created
└── song_output/            ← Output files
```

---

## 🎤 Adding RVC Voice Models

1. Download RVC v2 model (.pth + .index)
2. Create folder in `rvc_models/` with model name
3. Put .pth and .index files inside
4. Refresh models in WebUI

Example:
```
rvc_models/
└── MyVoice/
    ├── MyVoice.pth
    └── MyVoice.index
```

---

## ❓ FAQ

**Q: Do I need all models?**
A: Only `hubert_base.pt` and `rmvpe.pt` are required. Mel-RoFormer auto-downloads.

**Q: Where does Mel-RoFormer download to?**
A: `~/.cache/audio-separator/` (managed by audio-separator library)

**Q: How much space do I need?**
A: ~500 MB for base models + ~200 MB for Mel-RoFormer

---

## 🔗 Links

- [audio-separator models](https://github.com/nomadkaraoke/python-audio-separator)
- [RVC models](https://huggingface.co/lj1995/VoiceConversionWebUI)
- [Applio resources](https://huggingface.co/IAHispano/Applio)
