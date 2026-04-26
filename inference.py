#!/usr/bin/env python3
"""
Inference script for AASIST3 audio deepfake detection.

Usage:
    python inference.py audio1.wav audio2.flac         # predict files
    python inference.py --checkpoint best.pt audio.wav  # use fine-tuned model
"""

import os
import sys
import argparse

import torch
import torchaudio
import soundfile as sf

# Add AASIST3 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST3_DIR = os.path.join(SCRIPT_DIR, "AASIST3")
if os.path.exists(AASIST3_DIR):
    sys.path.insert(0, AASIST3_DIR)

from model import aasist3


def load_audio(filepath, target_sr=16000, max_length=64600):
    """Load and preprocess a single audio file."""
    audio_data, sr = sf.read(filepath)
    audio = torch.from_numpy(audio_data).float()

    # Ensure mono
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.mean(dim=-1, keepdim=True).T

    # Resample
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    # Pre-emphasis
    audio = torchaudio.functional.preemphasis(audio)

    # Pad or truncate
    if audio.shape[1] < max_length:
        audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
    else:
        audio = audio[:, :max_length]

    return audio


def main():
    parser = argparse.ArgumentParser(description="AASIST3 Audio Deepfake Detection")
    parser.add_argument("files", nargs="+", help="Audio file(s) to classify")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (.pt file)")
    parser.add_argument("--w2v_cache", default="weights",
                        help="Cache directory for Wav2Vec2 weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading AASIST3 model...")
    if args.checkpoint:
        # Load fine-tuned model
        model = aasist3(w2v_cache_dir=args.w2v_cache, load_pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        # Load pretrained from HuggingFace
        model = aasist3.from_pretrained("MTUCI/AASIST3", cache_dir=args.w2v_cache)
        print("  Loaded pretrained from HuggingFace")

    model = model.to(device)
    model.eval()

    # Run inference
    print()
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"  {filepath} — FILE NOT FOUND")
            continue

        try:
            audio = load_audio(filepath).to(device)

            with torch.no_grad():
                output = model(audio)
                probs = torch.softmax(output, dim=1)
                # Class 0 = spoof, Class 1 = bonafide
                pred = probs.argmax(dim=1).item()
                confidence = probs.max().item() * 100

                label = "Bonafide ✓" if pred == 1 else "Spoof ✗"
                print(f"  {os.path.basename(filepath):30s} → {label} ({confidence:.1f}%)")

        except Exception as e:
            print(f"  {filepath} — ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
