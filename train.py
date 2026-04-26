#!/usr/bin/env python3
"""
Simplified training script for AASIST3 on ASVspoof5 dataset.
Designed to run on Google Colab with a T4 GPU.

Usage:
    python train.py                          # train from scratch
    python train.py --resume checkpoint.pt   # resume training
    python train.py --finetune               # fine-tune from pretrained weights
"""

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add AASIST3 to path so we can import the model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST3_DIR = os.path.join(SCRIPT_DIR, "AASIST3")
if os.path.exists(AASIST3_DIR):
    sys.path.insert(0, AASIST3_DIR)

from model import aasist3


# ============================================================
# Dataset
# ============================================================

class ASVspoof5Dataset(Dataset):
    """
    Loads ASVspoof5 audio from TSV protocol files.
    
    TSV columns (space-separated):
    SPEAKER_ID FLAC_FILE GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
    
    KEY column (index 8) = "bonafide" or "spoof" -> label 1 or 0
    """

    def __init__(self, root_dir, meta_path, max_length=64600, sample_rate=16000, max_samples=None):
        """
        Args:
            root_dir:    Path to folder containing .flac files (e.g., asvspoof5/flac_T/)
            meta_path:   Path to TSV protocol file (e.g., ASVspoof5.train.tsv)
            max_length:  Max audio length in samples (64600 = ~4s at 16kHz)
            sample_rate: Target sample rate
            max_samples: Limit number of samples (None = use all)
        """
        self.root_dir = root_dir
        self.sr = sample_rate
        self.max_length = max_length

        # Parse TSV file
        self.samples = []
        with open(meta_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                flac_file = parts[1] + ".flac"
                key = parts[8]  # "bonafide" or "spoof"
                label = 1 if key == "bonafide" else 0  # 1=bonafide, 0=spoof
                audio_path = os.path.join(root_dir, flac_file)
                self.samples.append((audio_path, label))

        # Filter to only existing files
        existing = [(p, l) for p, l in self.samples if os.path.exists(p)]
        if len(existing) < len(self.samples):
            print(f"  Warning: {len(self.samples) - len(existing)} files not found, using {len(existing)} available files")
        self.samples = existing

        # Limit samples if requested
        if max_samples and max_samples < len(self.samples):
            # Balanced sampling
            bonafide = [(p, l) for p, l in self.samples if l == 1]
            spoof = [(p, l) for p, l in self.samples if l == 0]
            n_each = max_samples // 2
            self.samples = bonafide[:n_each] + spoof[:n_each]
            np.random.shuffle(self.samples)

        # Count labels
        n_bonafide = sum(1 for _, l in self.samples if l == 1)
        n_spoof = sum(1 for _, l in self.samples if l == 0)
        print(f"  Loaded {len(self.samples)} samples: {n_bonafide} bonafide, {n_spoof} spoof")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]

        try:
            audio_data, sr = sf.read(audio_path)
            audio = torch.from_numpy(audio_data).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a silent audio sample on error
            return torch.zeros(1, self.max_length), label

        # Ensure mono
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.mean(dim=-1, keepdim=True).T

        # Resample if needed
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)

        # Pre-emphasis
        audio = torchaudio.functional.preemphasis(audio)

        # Pad or truncate to max_length
        if audio.shape[1] < self.max_length:
            audio = torch.nn.functional.pad(audio, (0, self.max_length - audio.shape[1]))
        else:
            # Random crop during training
            start = np.random.randint(0, audio.shape[1] - self.max_length + 1)
            audio = audio[:, start:start + self.max_length]

        return audio, label


# ============================================================
# Metrics
# ============================================================

def compute_eer(scores, labels):
    """Compute Equal Error Rate (EER)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer * 100  # return as percentage


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, grad_accum_steps=1):
    model.train()
    total_loss = 0
    n_batches = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (audio, labels) in enumerate(dataloader):
        audio = audio.to(device)
        labels = labels.to(device)

        outputs = model(audio)
        loss = loss_fn(outputs, labels) / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        # Accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches
        if (i + 1) % 50 == 0:
            avg_loss = total_loss / n_batches
            acc = correct / total * 100
            print(f"    Batch {i+1}/{len(dataloader)} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    return total_loss / max(n_batches, 1), correct / max(total, 1) * 100


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []

    for audio, labels in dataloader:
        audio = audio.to(device)
        outputs = model(audio)
        probs = torch.softmax(outputs, dim=1)
        # Score = probability of bonafide (class 1)
        scores = probs[:, 1].cpu().numpy()
        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Accuracy
    preds = (all_scores > 0.5).astype(int)
    accuracy = (preds == all_labels).mean() * 100

    # EER
    try:
        eer = compute_eer(all_scores, all_labels)
    except Exception:
        eer = float("nan")

    return accuracy, eer


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train AASIST3 on ASVspoof5")
    parser.add_argument("--train_dir", default="asvspoof5/flac_T",
                        help="Path to training FLAC files")
    parser.add_argument("--train_meta", default="asvspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv",
                        help="Path to training TSV protocol file")
    parser.add_argument("--dev_dir", default="asvspoof5/flac_D",
                        help="Path to dev FLAC files")
    parser.add_argument("--dev_meta", default="asvspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
                        help="Path to dev TSV protocol file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (4-8 for T4 GPU)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for quick testing, e.g. 1000)")
    parser.add_argument("--max_dev_samples", type=int, default=5000,
                        help="Limit dev samples for faster validation")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers (2 for Colab)")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune from pretrained AASIST3 weights (from HuggingFace)")
    parser.add_argument("--w2v_cache", default="weights",
                        help="Cache directory for Wav2Vec2 weights")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create checkpoint dir
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets
    print("\nLoading training data...")
    train_dataset = ASVspoof5Dataset(
        root_dir=args.train_dir,
        meta_path=args.train_meta,
        max_samples=args.max_train_samples,
    )

    print("\nLoading dev data...")
    dev_dataset = ASVspoof5Dataset(
        root_dir=args.dev_dir,
        meta_path=args.dev_meta,
        max_samples=args.max_dev_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load model
    print("\nLoading model...")
    if args.finetune:
        print("  Fine-tuning from pretrained AASIST3 weights...")
        model = aasist3.from_pretrained("MTUCI/AASIST3", cache_dir=args.w2v_cache)
    else:
        print("  Training from scratch (will download Wav2Vec2 backbone)...")
        model = aasist3(w2v_cache_dir=args.w2v_cache)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    best_eer = float("inf")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        avg_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, 
            grad_accum_steps=args.grad_accum
        )

        # Evaluate
        dev_acc, dev_eer = evaluate(model, dev_loader, device)

        elapsed = time.time() - start_time
        print(f"  Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Dev Acc: {dev_acc:.1f}% | Dev EER: {dev_eer:.2f}% | Time: {elapsed:.0f}s")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "dev_eer": dev_eer,
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "latest.pt"))

        # Save best
        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  ★ New best EER: {best_eer:.2f}%")

        print()

    print(f"Training complete! Best EER: {best_eer:.2f}%")
    print(f"Checkpoints saved in: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
