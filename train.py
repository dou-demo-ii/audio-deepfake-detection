#!/usr/bin/env python3
"""
AASIST3 training on ASVspoof5 — combined script.

Augmentation: every sampled file is served 3×:
  original FLAC  |  MP3 @ 128 kbps  |  MP3 @ 256 kbps

Usage:
    python train.py
    python train.py --resume checkpoints/latest.pt
    python train.py --max_train_samples 50000

Recommended Colab usage:
    python train.py \
      --finetune \
      --max_train_samples 1000 \
      --max_dev_samples 500 \
      --epochs 1 \
      --batch_size 1 \
      --grad_accum 16 \
      --lr 1e-5 \
      --no_fp16 \
      --save_model_only \
      --no_epoch_snapshots \
      --checkpoint_dir /content/checkpoints
"""

import argparse
import io
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


# ── locate AASIST3 source ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST3_DIR = os.path.join(SCRIPT_DIR, "AASIST3")

if os.path.exists(AASIST3_DIR):
    sys.path.insert(0, AASIST3_DIR)

from model import aasist3


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset
# ═══════════════════════════════════════════════════════════════════════════ #


class ASVspoof5Dataset(Dataset):
    """
    Loads ASVspoof5 audio from its TSV protocol file.

    TSV columns:
      SPEAKER_ID FLAC_FILE GENDER CODEC CODEC_Q CODEC_SEED
      ATTACK_TAG ATTACK_LABEL KEY TMP

    KEY index 8:
      bonafide -> label 1
      spoof    -> label 0
    """

    def __init__(
        self,
        root_dir: str,
        meta_path: str,
        max_length: int = 64600,
        sample_rate: int = 16000,
    ):
        self.root_dir = root_dir
        self.sr = sample_rate
        self.max_length = max_length

        self.samples: list[tuple[str, int]] = []

        with open(meta_path) as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 9:
                    continue

                flac_file = parts[1] + ".flac"
                key = parts[8]

                label = 1 if key == "bonafide" else 0
                audio_path = os.path.join(root_dir, flac_file)

                self.samples.append((audio_path, label))

        before = len(self.samples)
        self.samples = [(p, l) for p, l in self.samples if os.path.exists(p)]
        dropped = before - len(self.samples)

        if dropped:
            print(f"  ⚠  {dropped} files not found on disk — skipped.")

        n_bon = sum(label for _, label in self.samples)
        n_spo = len(self.samples) - n_bon

        print(
            f"  {len(self.samples):,} samples  "
            f"({n_bon:,} bonafide / {n_spo:,} spoof)"
        )

    def __len__(self):
        return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        """
        Load FLAC into mono float32 tensor [1, T], resampled to self.sr.
        """
        audio_np, sr = sf.read(path)
        audio = torch.from_numpy(audio_np).float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.mean(dim=-1, keepdim=True).T

        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)

        return audio

    def _pad_or_crop(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Make audio length fixed to self.max_length.
        """
        total_samples = audio.shape[1]

        if total_samples < self.max_length:
            audio = torch.nn.functional.pad(audio, (0, self.max_length - total_samples))
        elif total_samples > self.max_length:
            start = random.randint(0, total_samples - self.max_length)
            audio = audio[:, start : start + self.max_length]

        return audio

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        try:
            audio = self._load(path)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            audio = torch.zeros(1, self.max_length)

        audio = torchaudio.functional.preemphasis(audio)
        audio = self._pad_or_crop(audio)

        return audio.squeeze(0), label


# ═══════════════════════════════════════════════════════════════════════════ #
#  MP3 augmentation wrapper
# ═══════════════════════════════════════════════════════════════════════════ #


class MP3AugmentedDataset(Dataset):
    """
    Wraps a Dataset returning (waveform, label), then round-trips audio
    through MP3 encode/decode at a given bitrate.
    """

    def __init__(self, dataset: Dataset, bitrate: str, sample_rate: int = 16000):
        self.dataset = dataset
        self.sample_rate = sample_rate

        if isinstance(bitrate, str) and bitrate.endswith("k"):
            self.bitrate_int = int(bitrate.replace("k", "")) * 1000
        else:
            self.bitrate_int = int(bitrate)

    def __len__(self):
        return len(self.dataset)

    def _mp3_roundtrip(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [T] or [1, T]
        returns same shape convention as input.
        """
        squeezed = waveform.ndim == 1

        if squeezed:
            waveform = waveform.unsqueeze(0)

        buffer = io.BytesIO()

        try:
            torchaudio.save(
                buffer,
                waveform,
                self.sample_rate,
                format="mp3",
                compression=self.bitrate_int,
            )
            buffer.seek(0)
            decoded, _ = torchaudio.load(buffer, format="mp3")
        except Exception:
            decoded = waveform.clone()

        target_len = waveform.shape[1]

        if decoded.shape[1] < target_len:
            decoded = torch.nn.functional.pad(decoded, (0, target_len - decoded.shape[1]))
        else:
            decoded = decoded[:, :target_len]

        return decoded.squeeze(0) if squeezed else decoded

    def __getitem__(self, idx: int):
        waveform, label = self.dataset[idx]
        waveform = self._mp3_roundtrip(waveform)
        return waveform, label


# ═══════════════════════════════════════════════════════════════════════════ #
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════ #


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    return float((fpr[idx] + fnr[idx]) / 2 * 100)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Train / eval loops
# ═══════════════════════════════════════════════════════════════════════════ #


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    device,
    grad_accum_steps: int = 1,
    scaler=None,
):
    model.train()

    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, (audio, labels) in enumerate(dataloader):
        audio = audio.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(audio)
                loss = loss_fn(outputs, labels) / grad_accum_steps

            scaler.scale(loss).backward()
        else:
            outputs = model(audio)
            loss = loss_fn(outputs, labels) / grad_accum_steps
            loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 50 == 0:
            print(
                f"    Batch {i + 1:>5}/{len(dataloader)} │ "
                f"loss {total_loss / n_batches:.4f} │ "
                f"acc {correct / total * 100:.1f}%"
            )

    return total_loss / max(n_batches, 1), correct / max(total, 1) * 100


@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None):
    model.eval()

    all_scores = []
    all_labels = []

    for audio, labels in dataloader:
        audio = audio.to(device, non_blocking=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(audio)
        else:
            outputs = model(audio)

        probs = torch.softmax(outputs, dim=1)

        # class 1 = bonafide
        scores = probs[:, 1].cpu().numpy()

        all_scores.extend(scores)
        all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    preds = (all_scores > 0.5).astype(int)
    accuracy = (preds == all_labels).mean() * 100

    try:
        eer = compute_eer(all_scores, all_labels)
    except Exception:
        eer = float("nan")

    return accuracy, eer


# ═══════════════════════════════════════════════════════════════════════════ #
#  Args
# ═══════════════════════════════════════════════════════════════════════════ #


def parse_args():
    parser = argparse.ArgumentParser(description="Train AASIST3 on ASVspoof5")

    # paths
    parser.add_argument("--train_dir", default="asvspoof5/flac_T")
    parser.add_argument(
        "--train_meta",
        default="asvspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv",
    )
    parser.add_argument("--dev_dir", default="asvspoof5/flac_D")
    parser.add_argument(
        "--dev_meta",
        default="asvspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv",
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--w2v_cache", default="weights")

    # data
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=100_000,
        help=(
            "Random subset size drawn from ASVspoof5 train. "
            "Each sample is served 3x: FLAC, MP3-128k, MP3-256k."
        ),
    )
    parser.add_argument("--max_dev_samples", type=int, default=5_000)
    parser.add_argument("--subset_seed", type=int, default=42)

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    # misc
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to full checkpoint latest.pt to resume training.",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Load pretrained AASIST3 weights from Hugging Face before training.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use automatic mixed precision AMP.",
    )
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")

    # checkpoint control
    parser.add_argument(
        "--save_model_only",
        action="store_true",
        help="Save best.pt as model.state_dict() only, without optimizer state.",
    )
    parser.add_argument(
        "--no_epoch_snapshots",
        action="store_true",
        help="Disable epoch_000.pt, epoch_001.pt snapshots.",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset builder
# ═══════════════════════════════════════════════════════════════════════════ #


def build_train_dataset(args) -> ConcatDataset:
    base = ASVspoof5Dataset(
        root_dir=args.train_dir,
        meta_path=args.train_meta,
    )

    n = min(args.max_train_samples, len(base))

    rng = random.Random(args.subset_seed)
    indices = rng.sample(range(len(base)), n)

    subset = Subset(base, indices)

    print(f"\n  Subset: {n:,} of {len(base):,} samples drawn (seed={args.subset_seed})")
    print(f"  3 views → {3 * n:,} training items per epoch\n")

    original = subset
    mp3_128 = MP3AugmentedDataset(subset, bitrate="128k")
    mp3_256 = MP3AugmentedDataset(subset, bitrate="256k")

    return ConcatDataset([original, mp3_128, mp3_256])


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main
# ═══════════════════════════════════════════════════════════════════════════ #


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice : {device}")

    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"AMP    : {'ON' if use_amp else 'OFF'}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # datasets
    print("── Training data ─────────────────────────────────────────────")
    train_dataset = build_train_dataset(args)

    print("── Dev data ──────────────────────────────────────────────────")
    dev_base = ASVspoof5Dataset(
        root_dir=args.dev_dir,
        meta_path=args.dev_meta,
    )

    if args.max_dev_samples and args.max_dev_samples < len(dev_base):
        rng_dev = random.Random(args.subset_seed)
        dev_idx = rng_dev.sample(range(len(dev_base)), args.max_dev_samples)
        dev_dataset = Subset(dev_base, dev_idx)
        print(f"  Dev subset: {args.max_dev_samples:,} of {len(dev_base):,}")
    else:
        dev_dataset = dev_base

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # model
    print("\n── Model ─────────────────────────────────────────────────────")

    if args.finetune:
        print("  Fine-tuning from pretrained AASIST3 weights (HuggingFace)…")
        model = aasist3.from_pretrained("MTUCI/AASIST3", cache_dir=args.w2v_cache)
    else:
        print("  Initialising from scratch (Wav2Vec2 backbone will be downloaded)…")
        model = aasist3(w2v_cache_dir=args.w2v_cache)

    model = model.to(device)

    if args.finetune:
        if hasattr(model, "frontend"):
            for param in model.frontend.parameters():
                param.requires_grad = False
        elif hasattr(model, "wav2vec2"):
            for param in model.wav2vec2.parameters():
                param.requires_grad = False

        print("Frozen Wav2Vec2/XLSR backbone to preserve weights & save VRAM.")

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  {n_total:,} total params  |  {n_trainable:,} trainable\n")

    # optimizer only trains unfrozen parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=1e-7,
        weight_decay=0,
    )

    loss_fn = nn.CrossEntropyLoss()

    # resume
    start_epoch = 0
    best_eer = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume} …")

        ckpt = torch.load(args.resume, map_location=device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_eer = ckpt.get("best_eer", float("inf"))

            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            print(
                f"  Resumed at epoch {start_epoch} "
                f"(best EER so far: {best_eer:.2f}%)\n"
            )
        else:
            raise ValueError(
                "Resume requires a full checkpoint with model_state_dict and "
                "optimizer_state_dict. Do not resume from model-only best.pt."
            )

    # training info
    eff_batch = args.batch_size * args.grad_accum

    print("═" * 62)
    print(f"  Epochs           : {args.epochs}")
    print(
        f"  Batch / eff.     : {args.batch_size} × {args.grad_accum} "
        f"grad-accum = {eff_batch}"
    )
    print(f"  LR               : {args.lr}")
    print(f"  Train items/epoch: {len(train_dataset):,}")
    print("  MP3 augmentation : 128 kbps + 256 kbps (in-memory)")
    print("═" * 62 + "\n")

    # train
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        print(f"Epoch {epoch + 1}/{args.epochs}")

        avg_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.grad_accum,
            scaler=scaler,
        )

        dev_acc, dev_eer = evaluate(
            model=model,
            dataloader=dev_loader,
            device=device,
            scaler=scaler,
        )

        elapsed = time.time() - t0

        print(
            f"  loss {avg_loss:.4f} │ train acc {train_acc:.1f}% │ "
            f"dev acc {dev_acc:.1f}% │ dev EER {dev_eer:.2f}% │ {elapsed:.0f}s"
        )

        # checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "dev_eer": dev_eer,
            "best_eer": best_eer,
        }

        if scaler:
            ckpt["scaler_state_dict"] = scaler.state_dict()

        # latest.pt = full checkpoint for resume
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))

        # best.pt = best model for inference / Streamlit
        if dev_eer < best_eer:
            best_eer = dev_eer

            best_path = os.path.join(args.checkpoint_dir, "best.pt")

            if args.save_model_only:
                torch.save(model.state_dict(), best_path)
            else:
                torch.save(ckpt, best_path)

            print(f"  ★ New best EER: {best_eer:.2f}%  → saved best.pt")

        # optional per-epoch checkpoint
        if not args.no_epoch_snapshots:
            snapshot_path = os.path.join(
                args.checkpoint_dir,
                f"epoch_{epoch:03d}.pt",
            )
            torch.save(ckpt, snapshot_path)

        print()

    print(f"Done. Best EER: {best_eer:.2f}%")
    print(f"Checkpoints in: {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()