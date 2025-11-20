# 🎙️ CSM Voice Cloning & Conversational Speech Generator

A Python implementation for generating natural-sounding, context-aware conversations using [Sesame AI's CSM-1B](https://huggingface.co/sesame/csm-1b) (Conversational Speech Model). This project enables **voice cloning** and **semantic speech synthesis** for creating realistic podcast-style dialogues.

---

## 🌟 Features

- ✅ **Voice Cloning**: Clone any voice with just 30-60 seconds of reference audio
- ✅ **Semantic Understanding**: Generates prosody based on conversational context (not just reading text)
- ✅ **Multi-Turn Conversations**: Maintains natural flow across multiple speakers
- ✅ **Context-Aware Prosody**: Adapts tone, rhythm, and emotion based on dialogue history
- ✅ **GPU Accelerated**: Optimized for CUDA (also supports CPU/MPS)

---

## 🎯 What Makes CSM Different?

| Feature | Traditional TTS | CSM (This Project) |
|---------|----------------|-------------------|
| **Prosody** | Rule-based, robotic | Learned from conversational data |
| **Context** | None | Uses conversation history |
| **Emotion** | Manual tags required | Inferred from semantics |
| **Voice Cloning** | Requires fine-tuning | Few-shot learning (30s audio) |

**Example:**
```
Same text: "Yeah, I agree."

Context 1 (excited): → Enthusiastic, upbeat delivery
Context 2 (sad): → Empathetic, softer delivery
```

---

## 🚀 Quick Start

### 1️⃣ Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/csm-voice-cloning.git
cd csm-voice-cloning

# Clone CSM repository
git clone https://github.com/SesameAILabs/csm.git

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
torch==2.9.1
torchaudio==2.9.1
torchtune==0.6.1
torchao==0.14.1
transformers==4.49.0
huggingface_hub==0.28.1
tokenizers==0.21.0
moshi==0.2.2
sentencepiece
sounddevice
librosa
scipy
```

### 2️⃣ Setup HuggingFace Authentication
```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # Get token from https://huggingface.co/settings/tokens
```

### 3️⃣ Run the Generator
```bash
python conversational_podcast_generator.py
```

**Output:** `full_conversation.wav` (multi-turn podcast conversation)

---

## 🎤 Voice Cloning Guide

### Step 1: Record Your Voice

Record 30-60 seconds of **natural speech** (not reading):
```bash
✅ Clear audio (no background noise)
✅ Conversational tone (include "um", "uh", "like")
✅ Varied intonation (not monotone)
✅ Sample rate: 24kHz recommended
```

**Example Tools:**
- Audacity (free)
- GarageBand (Mac)
- Voice Memos app

### Step 2: Prepare Your Prompt
```python
MY_VOICE = {
    "text": "Exact transcript of what you said in the recording...",
    "audio": "path/to/my_voice_sample.wav"
}

# Create prompt
my_prompt = prepare_prompt(
    text=MY_VOICE["text"],
    speaker=0,
    audio_path=MY_VOICE["audio"],
    sample_rate=generator.sample_rate
)
```

### Step 3: Generate with Your Voice
```python
audio = generator.generate(
    text="New text to synthesize in your voice",
    speaker=0,
    context=[my_prompt],  # Your voice reference
)

torchaudio.save("my_cloned_voice.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

---

## 📖 Code Structure
```
csm-voice-cloning/
│
├── conversational_podcast_generator.py  # Main script
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── csm/                                 # Cloned from Sesame AI
│   ├── generator.py                     # Model loader
│   ├── models.py                        # Architecture
│   └── ...
│
├── prompts/                             # Voice reference files
│   ├── conversational_a.wav
│   ├── conversational_b.wav
│   └── your_voice.wav                   # Add your recording here
│
└── outputs/                             # Generated audio
    ├── full_conversation.wav
    └── my_cloned_voice.wav
```

---

## 🧠 How It Works

### Architecture Flow
```
User Input Text
       ↓
[Text Encoder] (Transformer)
       ↓
[Semantic Tokens] (Captures meaning, not just words)
       ↓
[Acoustic Model] (Predicts mel-spectrograms + context)
       ↓
[Neural Codec] (Converts to audio waveform)
       ↓
Output Audio (24kHz WAV)
```

### Context-Aware Generation
```python
# First utterance: Uses voice prompt
context = [prompt_a]

# Later utterances: Uses previous speech by same speaker
context = [speaker0_segments[-1]]
```

**Why?** This maintains:
- Voice consistency
- Conversational rhythm
- Natural prosody evolution

---

## 📊 Performance

| Hardware | Generation Speed | Memory Usage |
|----------|-----------------|--------------|
| NVIDIA A100 (80GB) | ~2s per utterance | ~4GB VRAM |
| NVIDIA RTX 3090 | ~4s per utterance | ~4GB VRAM |
| CPU (16 cores) | ~15s per utterance | ~6GB RAM |

---

## 🔧 Configuration

### Key Parameters
```python
generator.generate(
    text="...",
    speaker=0,                    # 0 or 1 (speaker ID)
    context=[],                   # List of previous Segments
    max_audio_length_ms=None,     # Optional length limit
    temperature=1.0,              # Sampling randomness (default: 1.0)
)
```

### Advanced Options
```python
# Adjust prosody variation
temperature=0.8  # More deterministic
temperature=1.2  # More expressive

# Limit generation length
max_audio_length_ms=10_000  # 10 seconds max
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
```python
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

Run this between generations to free GPU memory.

### Issue: Audio Quality Issues
```bash
# Check sample rate
print(generator.sample_rate)  # Should be 24000

# Verify input audio
torchaudio.info("your_voice.wav")
```

### Issue: Dependency Conflicts
```bash
# Use exact versions
pip install torch==2.9.1 torchaudio==2.9.1 --force-reinstall
```

---

## 📚 Examples

### Example 1: Simple Voice Clone
```python
# Load model
generator = load_csm_1b(device="cuda")

# Your voice prompt
my_voice = prepare_prompt(
    text="Hey there! This is my natural speaking voice.",
    speaker=0,
    audio_path="my_voice.wav",
    sample_rate=generator.sample_rate
)

# Generate
audio = generator.generate(
    text="Welcome to my AI-generated podcast!",
    speaker=0,
    context=[my_voice]
)

torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Example 2: Two-Speaker Podcast

See `conversational_podcast_generator.py` for full implementation.

---

## 🔬 Research & Citation

This project uses **CSM-1B** by Sesame AI Labs:
```bibtex
@misc{csm2024,
  title={CSM: Conversational Speech Model},
  author={Sesame AI Labs},
  year={2024},
  url={https://huggingface.co/sesame/csm-1b}
}
---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🔗 Resources

- **CSM-1B Model**: https://huggingface.co/sesame/csm-1b
- **Sesame AI GitHub**: https://github.com/SesameAILabs/csm
- **Demo Audio**: https://drive.google.com/file/d/1xDQIkw5Qb8GlVqT8R7XyvmQxLIV28h6R/view?usp=drive_link

---

## 💬 Contact

- **Author**: ISHIKA JAIN
- **Email**: 17ishika@gmail.com
---

## ⭐ Acknowledgments

- Sesame AI Labs for the CSM-1B model
- HuggingFace for model hosting
- PyTorch community for audio tools

---

**If this project helped you, please consider giving it a ⭐!**
