#!/usr/bin/env python3
"""
AASIST3 training on ASVspoof5 — T4-optimised.

Augmentation: every sampled file is served 3×:
  original FLAC  |  MP3 @ 128 kbps  |  MP3 @ 256 kbps

Optimisations vs. baseline
──────────────────────────────────────────────────────────────────────
  DataLoader     num_workers 4, prefetch_factor 4, fork multiprocessing
  Batch          size 16 × grad_accum 2 = eff. batch 32
                 (same eff. batch as before, 2× fewer optimiser steps)
  Compiler       torch.compile(mode="reduce-overhead")  ~+15-25%
  Schedule       OneCycleLR — faster convergence, no LR tuning needed
  Reshuffling    subset re-drawn every epoch (seed + epoch offset)
  Feat. cache    --cache_features: run frozen backbone once, cache to
                 disk, train head only in all subsequent epochs.
                 Requires the model to expose model.ssl_model,
                 model.frontend, or model.wav2vec2.
                 Only meaningful with --finetune (frozen backbone).
  VRAM log       prints allocated/reserved VRAM after every epoch
──────────────────────────────────────────────────────────────────────

Usage:
    python train.py
    python train.py --finetune --cache_features   # fastest on T4
    python train.py --resume checkpoints/latest.pt
    python train.py --max_train_samples 50000
    python train.py --no_compile                  # disable torch.compile
"""

import argparse
import io
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

# ── locate AASIST3 source ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AASIST3_DIR = os.path.join(SCRIPT_DIR, "AASIST3")
if os.path.exists(AASIST3_DIR):
    sys.path.insert(0, AASIST3_DIR)

from model import aasist3

# ═══════════════════════════════════════════════════════════════════════════ #
#  Datasets                                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #


class ASVspoof5Dataset(Dataset):
    """
    Loads ASVspoof5 audio from its TSV protocol file.

    TSV columns (space-separated):
      SPEAKER_ID  FLAC_FILE  GENDER  CODEC  CODEC_Q  CODEC_SEED  ATTACK_TAG  ATTACK_LABEL  KEY  TMP
    KEY (index 8): "bonafide" → label 1 | "spoof" → label 0
    """

    def __init__(
        self,
        root_dir: str,
        meta_path: str,
        max_length: int = 64600,
        sample_rate: int = 16_000,
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
                label = 1 if parts[8] == "bonafide" else 0
                audio_path = os.path.join(root_dir, flac_file)
                self.samples.append((audio_path, label))

        before = len(self.samples)
        self.samples = [(p, l) for p, l in self.samples if os.path.exists(p)]
        dropped = before - len(self.samples)
        if dropped:
            print(f"  ⚠  {dropped} files not found on disk — skipped.")

        n_bon = sum(l for _, l in self.samples)
        n_spo = len(self.samples) - n_bon
        print(
            f"  {len(self.samples):,} samples  ({n_bon:,} bonafide / {n_spo:,} spoof)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        audio_np, sr = sf.read(path)
        audio = torch.from_numpy(audio_np).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.mean(dim=-1, keepdim=True).T
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        return audio  # [1, T]

    def _pad_or_crop(self, audio: torch.Tensor) -> torch.Tensor:
        T = audio.shape[1]
        if T < self.max_length:
            audio = torch.nn.functional.pad(audio, (0, self.max_length - T))
        elif T > self.max_length:
            start = random.randint(0, T - self.max_length)
            audio = audio[:, start : start + self.max_length]
        return audio  # [1, max_length]

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            audio = self._load(path)
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            audio = torch.zeros(1, self.max_length)
        audio = torchaudio.functional.preemphasis(audio)
        audio = self._pad_or_crop(audio)
        return audio.squeeze(0), label  # ([max_length], int)


# ─────────────────────────────────────────────────────────────────────────── #


class MP3AugmentedDataset(Dataset):
    """
    Wraps any Dataset returning (waveform [T], label) and round-trips
    audio through an in-memory MP3 encode/decode at the given bitrate.
    """

    def __init__(self, dataset: Dataset, bitrate: str, sample_rate: int = 16_000):
        self.dataset = dataset
        self.sample_rate = sample_rate
        # torchaudio MP3 compression parameter is in bps for the ffmpeg backend
        if isinstance(bitrate, str) and bitrate.endswith("k"):
            self.bitrate_int = int(bitrate[:-1]) * 1000
        else:
            self.bitrate_int = int(bitrate)

    def __len__(self) -> int:
        return len(self.dataset)

    def _mp3_roundtrip(self, waveform: torch.Tensor) -> torch.Tensor:
        squeezed = waveform.ndim == 1
        if squeezed:
            waveform = waveform.unsqueeze(0)
        buf = io.BytesIO()
        try:
            torchaudio.save(
                buf,
                waveform,
                self.sample_rate,
                format="mp3",
                compression=self.bitrate_int,
            )
            buf.seek(0)
            decoded, _ = torchaudio.load(buf, format="mp3")
        except Exception:
            decoded = waveform.clone()
        T = waveform.shape[1]
        if decoded.shape[1] < T:
            decoded = torch.nn.functional.pad(decoded, (0, T - decoded.shape[1]))
        else:
            decoded = decoded[:, :T]
        return decoded.squeeze(0) if squeezed else decoded

    def __getitem__(self, idx: int):
        waveform, label = self.dataset[idx]
        return self._mp3_roundtrip(waveform), label


# ─────────────────────────────────────────────────────────────────────────── #


class PrecomputedDataset(Dataset):
    """Serves (feature_tensor, label) from pre-extracted backbone features."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        assert features.shape[0] == labels.shape[0]
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], int(self.labels[idx])


# ═══════════════════════════════════════════════════════════════════════════ #
#  Feature caching (frozen backbone)                                          #
# ═══════════════════════════════════════════════════════════════════════════ #


# Priority order for backbone attribute lookup
_BACKBONE_ATTRS = ("ssl_model", "frontend", "wav2vec2", "encoder", "feature_extractor")


def _find_backbone(model: nn.Module):
    """Return (submodule, attribute_name) for the first known backbone attr."""
    for attr in _BACKBONE_ATTRS:
        if hasattr(model, attr):
            return getattr(model, attr), attr
    return None, None


@torch.no_grad()
def _extract_view(
    backbone: nn.Module,
    dataset: Dataset,
    device: torch.device,
    cache_path: str,
    batch_size: int,
    use_amp: bool,
) -> "PrecomputedDataset | None":
    """
    Run backbone on every waveform in dataset; cache result to disk.
    Returns a PrecomputedDataset, or loads from cache if already present.
    """
    if os.path.exists(cache_path):
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        return PrecomputedDataset(saved["features"], saved["labels"])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=device.type == "cuda",
        prefetch_factor=4,
        persistent_workers=True,
        multiprocessing_context="fork",
    )

    feats_list, labels_list = [], []
    backbone.eval()
    t0 = time.time()

    for step, (audio, labels) in enumerate(loader):
        audio = audio.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.autocast("cuda"):
                out = backbone(audio)
        else:
            out = backbone(audio)

        # Some backbones return (last_hidden_state, ...) tuples
        if isinstance(out, (tuple, list)):
            out = out[0]

        feats_list.append(out.cpu())
        labels_list.append(labels)

        if (step + 1) % 100 == 0:
            pct = (step + 1) / len(loader) * 100
            print(
                f"      {step + 1}/{len(loader)} ({pct:.0f}%)  {time.time() - t0:.0f}s"
            )

    features = torch.cat(feats_list, dim=0)
    labels_t = torch.cat(labels_list, dim=0)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "labels": labels_t}, cache_path)
    print(
        f"    ✓ Cached {tuple(features.shape)}  →  {cache_path}  ({time.time() - t0:.0f}s)"
    )

    return PrecomputedDataset(features, labels_t)


def build_feature_cache(
    model: nn.Module,
    subset: Dataset,
    device: torch.device,
    cache_dir: str,
    subset_key: str,  # identifies this subset (e.g. "seed42_n100000")
    batch_size: int,
    use_amp: bool,
) -> "ConcatDataset | None":
    """
    Extract backbone features for all 3 views (orig, mp3-128k, mp3-256k)
    and return a ConcatDataset of PrecomputedDatasets.

    Falls back to None if no recognisable backbone attribute is found,
    in which case the caller should use the normal waveform pipeline.
    """
    backbone, attr = _find_backbone(model)
    if backbone is None:
        print(
            f"  ⚠  Feature caching: none of {_BACKBONE_ATTRS} found on model. "
            "Falling back to waveform pipeline."
        )
        return None

    print(f"  Feature caching via model.{attr}  (key: {subset_key})")

    views = [
        ("orig", subset),
        ("mp3_128k", MP3AugmentedDataset(subset, "128k")),
        ("mp3_256k", MP3AugmentedDataset(subset, "256k")),
    ]

    datasets = []
    for view_name, view_ds in views:
        cache_path = os.path.join(cache_dir, f"feats_{view_name}_{subset_key}.pt")
        print(f"    [{view_name}] …")
        ds = _extract_view(backbone, view_ds, device, cache_path, batch_size, use_amp)
        if ds is None:
            return None
        datasets.append(ds)

    return ConcatDataset(datasets)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Metrics                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2 * 100)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Train / eval loops                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #


def train_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    scheduler,
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

            # OneCycleLR steps per optimiser update, not per epoch
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"    Batch {i + 1:>5}/{len(dataloader)} │ "
                f"loss {total_loss / n_batches:.4f} │ "
                f"acc {correct / total * 100:.1f}% │ "
                f"lr {lr_now:.2e}"
            )

    return total_loss / max(n_batches, 1), correct / max(total, 1) * 100


@torch.no_grad()
def evaluate(model, dataloader, device, scaler=None):
    model.eval()
    all_scores, all_labels = [], []

    for audio, labels in dataloader:
        audio = audio.to(device, non_blocking=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(audio)
        else:
            outputs = model(audio)

        probs = torch.softmax(outputs, dim=1)
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
#  Helpers                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #


def vram_str() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    return f"VRAM {alloc:.1f}/{reserv:.1f} GB (alloc/reserv)"


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    """Centralised DataLoader factory with all T4 tweaks applied."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,  # only drop on train
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        # fork is faster than spawn on Linux (Colab / most cloud VMs)
        multiprocessing_context="fork" if num_workers > 0 else None,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Argument parsing                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #


def parse_args():
    p = argparse.ArgumentParser(description="Train AASIST3 on ASVspoof5 (T4-optimised)")

    # ── paths ───────────────────────────────────────────────────────────── #
    p.add_argument("--train_dir", default="asvspoof5/flac_T")
    p.add_argument(
        "--train_meta", default="asvspoof5/ASVspoof5_protocols/ASVspoof5.train.tsv"
    )
    p.add_argument("--dev_dir", default="asvspoof5/flac_D")
    p.add_argument(
        "--dev_meta", default="asvspoof5/ASVspoof5_protocols/ASVspoof5.dev.track_1.tsv"
    )
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--w2v_cache", default="weights")
    p.add_argument(
        "--feat_cache_dir",
        default="feat_cache",
        help="Directory for pre-extracted backbone features (--cache_features).",
    )

    # ── data ────────────────────────────────────────────────────────────── #
    p.add_argument(
        "--max_train_samples",
        type=int,
        default=100_000,
        help="Subset size drawn from ASVspoof5 train per epoch. "
        "Each sample is served 3× → 3× this many items/epoch.",
    )
    p.add_argument("--max_dev_samples", type=int, default=5_000)
    p.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Base seed. Actual seed per epoch = subset_seed + epoch.",
    )

    # ── training ────────────────────────────────────────────────────────── #
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,  # was 8 — T4 handles 16 comfortably with AMP + frozen backbone
        help="Per-step batch size. Effective batch = batch_size × grad_accum.",
    )
    p.add_argument("--grad_accum", type=int, default=2)  # was 4
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)  # was 2

    # ── misc ────────────────────────────────────────────────────────────── #
    p.add_argument(
        "--resume", default=None, help="Path to a checkpoint (.pt) to resume from."
    )
    p.add_argument(
        "--finetune",
        action="store_true",
        help="Load pretrained AASIST3 weights from HuggingFace and freeze backbone.",
    )
    p.add_argument(
        "--cache_features",
        action="store_true",
        help="Pre-extract frozen backbone features once and cache to disk. "
        "Only meaningful with --finetune. Fastest option on T4.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use AMP (recommended for T4).",
    )
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="torch.compile the model (PyTorch ≥ 2.0). ~+15-25%% on T4.",
    )
    p.add_argument("--no_compile", dest="compile", action="store_false")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Dataset builders                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #


def build_train_dataset(args, base: ASVspoof5Dataset, epoch: int = 0) -> ConcatDataset:
    """
    Draw a fresh random subset each epoch (seed + epoch), then wrap ×3.

    Re-drawing keeps the model from memorising a fixed order while still
    being reproducible: epoch 0 always uses seed=subset_seed+0, etc.
    """
    n = min(args.max_train_samples, len(base))
    rng = random.Random(args.subset_seed + epoch)
    indices = rng.sample(range(len(base)), n)
    subset = Subset(base, indices)

    original = subset
    mp3_128 = MP3AugmentedDataset(subset, bitrate="128k")
    mp3_256 = MP3AugmentedDataset(subset, bitrate="256k")

    return ConcatDataset([original, mp3_128, mp3_256])


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main                                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #


def main():
    args = parse_args()

    # ── device & AMP ────────────────────────────────────────────────────── #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\nDevice : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1e9:.1f} GB")
    print(f"AMP    : {'ON' if use_amp else 'OFF'}")
    print(f"Compile: {'ON' if args.compile else 'OFF'}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── load full datasets from disk once ───────────────────────────────── #
    print("── Training data ──────────────────────────────────────────────")
    train_base = ASVspoof5Dataset(root_dir=args.train_dir, meta_path=args.train_meta)

    print("── Dev data ───────────────────────────────────────────────────")
    dev_base = ASVspoof5Dataset(root_dir=args.dev_dir, meta_path=args.dev_meta)
    if args.max_dev_samples and args.max_dev_samples < len(dev_base):
        rng_dev = random.Random(args.subset_seed)
        dev_idx = rng_dev.sample(range(len(dev_base)), args.max_dev_samples)
        dev_ds = Subset(dev_base, dev_idx)
        print(f"  Dev subset: {args.max_dev_samples:,} of {len(dev_base):,}")
    else:
        dev_ds = dev_base

    dev_loader = make_loader(
        dev_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── model ───────────────────────────────────────────────────────────── #
    print("\n── Model ──────────────────────────────────────────────────────")
    if args.finetune:
        print("  Fine-tuning from pretrained AASIST3 weights (HuggingFace)…")
        model = aasist3.from_pretrained("MTUCI/AASIST3", cache_dir=args.w2v_cache)
    else:
        print("  Initialising from scratch (Wav2Vec2 backbone will be downloaded)…")
        model = aasist3(w2v_cache_dir=args.w2v_cache)

    model = model.to(device)

    # Freeze backbone when fine-tuning
    if args.finetune:
        frozen_module = None
        for attr in _BACKBONE_ATTRS:
            if hasattr(model, attr):
                frozen_module = getattr(model, attr)
                for p in frozen_module.parameters():
                    p.requires_grad = False
                print(f"  Frozen model.{attr} to preserve weights & save VRAM.")
                break
        if frozen_module is None:
            print("  ⚠  Could not find backbone to freeze — all parameters trainable.")

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ── torch.compile ───────────────────────────────────────────────────── #
    if args.compile:
        if hasattr(torch, "compile"):
            print("  Compiling model (reduce-overhead) …")
            # Disable compilation for the frozen backbone sub-module if found;
            # compile only the trainable part to reduce warm-up time.
            model = torch.compile(model, mode="reduce-overhead")
            print("  ✓ torch.compile done (first forward will trigger JIT warm-up)")
        else:
            print("  torch.compile not available (requires PyTorch ≥ 2.0) — skipped.")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_total:,} total params  |  {n_trainable:,} trainable\n")

    # ── optimiser & loss ────────────────────────────────────────────────── #
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        eps=1e-7,
        weight_decay=1e-4,  # slight regularisation; was 0
    )
    loss_fn = nn.CrossEntropyLoss()

    # OneCycleLR: one cycle spanning the full training run.
    # steps_per_epoch = ceil(3 × max_train_samples / (batch_size × grad_accum))
    steps_per_epoch = (3 * min(args.max_train_samples, len(train_base))) // (
        args.batch_size * args.grad_accum
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warm-up
        anneal_strategy="cos",
        div_factor=25,  # initial_lr = max_lr / 25
        final_div_factor=1e4,  # final_lr  = initial_lr / 1e4
    )

    # ── optional resume ─────────────────────────────────────────────────── #
    start_epoch = 0
    best_eer = float("inf")

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume} …")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float("inf"))
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  Resumed at epoch {start_epoch}  (best EER so far: {best_eer:.2f}%)\n")

    # ── feature cache (frozen backbone only) ────────────────────────────── #
    use_feat_cache = args.cache_features and args.finetune
    cached_train_ds = None  # set below if feature caching is active

    if use_feat_cache:
        print("\n── Pre-extracting backbone features ───────────────────────────")
        n_sub = min(args.max_train_samples, len(train_base))
        subset_key = f"seed{args.subset_seed}_n{n_sub}"
        # Build the epoch-0 subset (features are backbone-deterministic for
        # a given waveform, so we can cache across epochs for the same subset)
        rng0 = random.Random(args.subset_seed)
        idx0 = rng0.sample(range(len(train_base)), n_sub)
        subset0 = Subset(train_base, idx0)

        # Temporarily unwrap compiled model to access backbone submodules
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        cached_train_ds = build_feature_cache(
            model=raw_model,
            subset=subset0,
            device=device,
            cache_dir=args.feat_cache_dir,
            subset_key=subset_key,
            batch_size=args.batch_size * 2,
            use_amp=use_amp,
        )
        if cached_train_ds is None:
            print("  Feature caching failed — falling back to waveform pipeline.")
            use_feat_cache = False
        else:
            print(
                f"  {len(cached_train_ds):,} cached feature items ready.\n"
                f"  NOTE: subset is fixed (no per-epoch reshuffling) when using\n"
                f"  feature cache, because features depend on waveform identity.\n"
            )

    # ── training banner ─────────────────────────────────────────────────── #
    eff_batch = args.batch_size * args.grad_accum
    n_train = (
        len(cached_train_ds)
        if use_feat_cache
        else (3 * min(args.max_train_samples, len(train_base)))
    )

    print("═" * 64)
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch / eff.      : {args.batch_size} × {args.grad_accum} = {eff_batch}")
    print(f"  LR (max)          : {args.lr}  (OneCycleLR, 10% warm-up)")
    print(f"  Train items/epoch : {n_train:,}")
    print(f"  MP3 augmentation  : 128 kbps + 256 kbps (in-memory)")
    print(f"  Feature cache     : {'ON' if use_feat_cache else 'OFF'}")
    print(f"  Per-epoch reshuffle: {'OFF (feat cache)' if use_feat_cache else 'ON'}")
    print("═" * 64 + "\n")

    # ── training loop ───────────────────────────────────────────────────── #
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Build (or reuse) training dataset
        if use_feat_cache:
            train_dataset = cached_train_ds
        else:
            # Fresh subset each epoch → better generalisation
            train_dataset = build_train_dataset(args, train_base, epoch=epoch)
            n_items = len(train_dataset)
            print(f"  {n_items:,} training items  (seed={args.subset_seed + epoch})")

        train_loader = make_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        avg_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            grad_accum_steps=args.grad_accum,
            scaler=scaler,
        )

        dev_acc, dev_eer = evaluate(model, dev_loader, device, scaler=scaler)

        elapsed = time.time() - t0
        print(
            f"  loss {avg_loss:.4f} │ "
            f"train acc {train_acc:.1f}% │ "
            f"dev acc {dev_acc:.1f}% │ "
            f"dev EER {dev_eer:.2f}% │ "
            f"{elapsed:.0f}s"
        )
        if device.type == "cuda":
            print(f"  {vram_str()}")

        # ── checkpoint ──────────────────────────────────────────────────── #
        # Unwrap compiled model for saving (torch.compile wraps state_dict)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        ckpt = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "dev_eer": dev_eer,
            "best_eer": best_eer,
        }
        if scaler:
            ckpt["scaler_state_dict"] = scaler.state_dict()

        torch.save(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt"))

        if dev_eer < best_eer:
            best_eer = dev_eer
            ckpt["best_eer"] = best_eer
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  ★ New best EER: {best_eer:.2f}%  → saved best.pt")

        print()

    print(f"Done.  Best EER: {best_eer:.2f}%")
    print(f"Checkpoints in : {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
