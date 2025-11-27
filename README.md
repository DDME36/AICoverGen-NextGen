# 🎵 AICoverGen NextGen

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDME36/AICoverGen-NextGen/blob/main/AICoverGen_NextGen_Colab.ipynb)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Create AI Cover songs with any voice, easily in Google Colab**

[🚀 Quick Start](#-quick-start) • [✨ Features](#-features) • [📖 Usage](#-usage) • [🎯 Tips](#-tips)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 **BS-RoFormer** | SOTA vocal separation (SDR 12.97) - Best quality vocal extraction |
| 🔇 **UVR-DeEcho-DeReverb** | Remove echo and reverb from vocals for cleaner output |
| 🧠 **ContentVec** | Better feature extraction for RVC v2 |
| 🎵 **RVC v2** | High quality voice conversion |
| 🎚️ **RMVPE** | Accurate pitch detection |
| 🎛️ **Auto-Mixing** | Intelligent compression, EQ matching, and gain staging |
| 🌐 **YouTube Support** | Paste YouTube link directly |
| 📁 **File Upload** | Upload your own audio files |

---

## 🚀 Quick Start

### Google Colab (Recommended)

1. Click **Open in Colab** above
2. Run Cell 1 (Install) - wait ~5 minutes
3. Run Cell 2 (WebUI)
4. Click the Gradio link that appears
5. Start creating AI Covers!

### Local Installation

```bash
# Clone repository
git clone https://github.com/DDME36/AICoverGen-NextGen.git
cd AICoverGen-NextGen

# Install dependencies (Python 3.10 required)
pip install -r requirements.txt

# Download models
python setup_models.py

# Run WebUI
python src/webui.py --share
```

---

## 📖 Usage

### 1️⃣ Download Voice Model

Go to **Download model** tab and paste model link from:
- [HuggingFace](https://huggingface.co/models?search=rvc)
- [Pixeldrain](https://pixeldrain.com/)

### 2️⃣ Generate AI Cover

1. Go to **Generate** tab
2. Select Voice Model
3. Paste YouTube link or upload file
4. Adjust Pitch (if needed)
5. Click **Generate**

### 3️⃣ Pitch Settings

| Conversion | Pitch Change |
|------------|--------------|
| Male → Female | +1 to +2 |
| Female → Male | -1 to -2 |
| Same voice | 0 |

---

## 🎯 Tips

- **Index Rate 0.5-0.7** = Smoothest voice
- **Protect 0.33** = Prevent voice cracking
- **RMS Mix 0.25** = Natural loudness
- **RMVPE** = Best pitch detection

---

## 🔧 Models Used

| Model | Purpose | Quality |
|-------|---------|---------|
| BS-RoFormer | Vocal Separation | SDR 12.97 |
| UVR-DeEcho-DeReverb | Remove Echo/Reverb | High quality |
| ContentVec | Feature Extraction | High quality |
| RMVPE | Pitch Detection | 98%+ accuracy |
| RVC v2 | Voice Conversion | High quality |

---

## 🔄 Pipeline

```
YouTube/Audio File
       ↓
1. Vocal Separation (BS-RoFormer)
       ↓
2. DeReverb/DeEcho (UVR-DeEcho-DeReverb)
       ↓
3. Voice Conversion (RVC v2)
       ↓
4. Auto-Mixing (Compression + EQ + Gain Staging)
       ↓
   Final AI Cover
```

## 📁 Project Structure

```
src/
├── config.py          # Configuration & paths
├── downloader.py      # YouTube download
├── separator.py       # Vocal separation
├── voice_converter.py # RVC voice conversion
├── mixer.py           # Audio mixing & effects
├── pipeline.py        # Main pipeline orchestrator
├── webui.py           # Gradio UI
├── main.py            # Entry point
├── rvc.py             # RVC core
└── mdx.py             # MDX-Net fallback
```

---

## 📝 Credits

- [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [audio-separator](https://github.com/karaokenerds/python-audio-separator)
- [RVC WebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)
- [UVR Models](https://huggingface.co/seanghay/uvr_models)

---

## 📄 License

MIT License - Free to use, but do not use for illegal purposes.

---

<div align="center">

**Made with ❤️ by DDME36**

</div>
