import os

import comet_ml
import hydra
import torch, torch.nn as nn
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
import io
import random
import torchaudio
import numpy as np


from model import aasist3
from datasets import print_fancy
from datasets import ASVspoof2019Dev, ASVspoof5Dev, ASVspoof5Train
from utils import train_one_epoch, compute_scores, compute_antispoofing_metrics


class MP3AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps an existing dataset and re-encodes each audio sample to MP3
    at a given bitrate on-the-fly (in memory), then decodes it back.

    Expects the underlying dataset to return (waveform_tensor, sample_rate, label)
    where waveform_tensor is shape [C, T] or [T].
    Adjust the __getitem__ unpacking below if your dataset returns a different structure.
    """

    def __init__(self, dataset, bitrate: str):
        """
        Args:
            dataset: any PyTorch Dataset whose __getitem__ returns
                     (waveform: Tensor[C,T], sample_rate: int, label)
            bitrate:  e.g. "128k" or "256k"
        """
        self.dataset = dataset
        self.bitrate = bitrate

    def __len__(self):
        return len(self.dataset)

    def _to_mp3_and_back(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Round-trip: tensor → MP3 bytes → tensor."""
        # torchaudio.save / torchaudio.load work with file-like objects
        buf = io.BytesIO()
        # torchaudio expects [C, T]; add channel dim if necessary
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(
            buf,
            waveform,
            sample_rate,
            format="mp3",
            compression=self.bitrate,   # e.g. "128k"
        )
        buf.seek(0)
        decoded, _ = torchaudio.load(buf, format="mp3")
        return decoded

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # --- adapt this unpacking to your dataset's actual return signature ---
        waveform, sample_rate, label = item[0], item[1], item[2]
        extra = item[3:]          # any additional fields (e.g. utterance id)
        # ----------------------------------------------------------------------
        waveform = self._to_mp3_and_back(waveform, sample_rate)
        return (waveform, sample_rate, label) + tuple(extra)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config):
    print_fancy(str(OmegaConf.to_container(config)))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config["find_unused_parameters"])

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="comet_ml",
        gradient_accumulation_steps=config.get("gradient_accumulation_steps")
    )

    print_fancy("Accelerator loaded")

    # ------------------------------------------------------------------ #
    #  Training data: ASVspoof5 only, 100 000-sample random subset,       #
    #  served as: original FLAC  +  MP3 128 kbps  +  MP3 256 kbps        #
    # ------------------------------------------------------------------ #
    full_asvspoof5train = ASVspoof5Train(
        root_dir=config['data']["asvspoof5_train"]["root_dir"],
        meta_path=config['data']["asvspoof5_train"]["meta_path"],
    )

    # Reproducible random subset of 100 000 items
    subset_size = config.get("train_subset_size", 100_000)
    subset_size = min(subset_size, len(full_asvspoof5train))
    rng = random.Random(config.get("subset_seed", 42))
    subset_indices = rng.sample(range(len(full_asvspoof5train)), subset_size)
    asvspoof5_subset = Subset(full_asvspoof5train, subset_indices)

    accelerator.print(
        f"ASVspoof5 subset: {subset_size} samples "
        f"out of {len(full_asvspoof5train)} total"
    )

    # Three views of the same subset
    asvspoof5_original  = asvspoof5_subset                          # raw FLAC
    asvspoof5_mp3_128   = MP3AugmentedDataset(asvspoof5_subset, bitrate="128k")
    asvspoof5_mp3_256   = MP3AugmentedDataset(asvspoof5_subset, bitrate="256k")

    train_dataset = ConcatDataset([
        asvspoof5_original,
        asvspoof5_mp3_128,
        asvspoof5_mp3_256,
    ])

    accelerator.print(
        f"Total training items (orig + 128k + 256k): {len(train_dataset)}"
    )
    print_fancy("Train dataset ready")

    # ------------------------------------------------------------------ #
    #  Validation data (unchanged)                                         #
    # ------------------------------------------------------------------ #
    asvspoof5dev = ASVspoof5Dev(
        root_dir=config['data']["asvspoof5_dev"]["root_dir"],
        meta_path=config['data']['asvspoof5_dev']['meta_path']
    )

    asvspoof19dev = ASVspoof2019Dev(
        root_dir=config['data']['asvspoof2019_dev']["root_dir"],
        meta_path=config['data']['asvspoof2019_dev']['meta_path']
    )

    print_fancy('Validation datasets loaded')

    # ------------------------------------------------------------------ #
    #  DataLoaders                                                         #
    # ------------------------------------------------------------------ #
    train_dl = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        # pin_memory speeds up CPU→GPU transfers; adjust if OOM
        pin_memory=True,
    )

    asv19_dl = DataLoader(
        asvspoof19dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
    )

    asv5_dl = DataLoader(
        asvspoof5dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
    )

    print_fancy('DataLoaders initialised')

    # ------------------------------------------------------------------ #
    #  Model / optimiser / loss                                            #
    # ------------------------------------------------------------------ #
    loss_fn  = nn.CrossEntropyLoss()
    model    = aasist3()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=1e-7,
        weight_decay=0,
    )

    train_dl, asv19_dl, asv5_dl, loss_fn, model, optimizer = accelerator.prepare(
        train_dl, asv19_dl, asv5_dl, loss_fn, model, optimizer
    )

    print_fancy("Important entities created")

    # ------------------------------------------------------------------ #
    #  Comet ML                                                            #
    # ------------------------------------------------------------------ #
    accelerator.init_trackers(
        project_name=config.get("comet_project_name", "default"),
        config=OmegaConf.to_container(config),
        init_kwargs={
            "comet": {
                "api_key":         os.environ.get("COMET_API_KEY", None),
                "workspace":       config.get("comet_workspace", None),
                "project_name":    config.get("comet_project_name", "default"),
                "experiment_name": config.get("comet_run_name", "default"),
                "auto_output_logging": "simple",
            }
        },
    )
    print_fancy("Comet experiment initialised through Accelerator")

    # ------------------------------------------------------------------ #
    #  Optional checkpoint restore                                         #
    # ------------------------------------------------------------------ #
    resume_epoch = 0
    if config.get("resume_from_checkpoint"):
        checkpoint_path = config.get("resume_from_checkpoint")
        if os.path.exists(checkpoint_path):
            weights_before = {
                n: p.clone().detach() for n, p in model.named_parameters()
            }
            accelerator.print(f"Restoring checkpoint from {checkpoint_path}")
            accelerator.load_state(checkpoint_path)

            weights_changed = any(
                not torch.equal(weights_before[n], p)
                for n, p in model.named_parameters()
            )
            if weights_changed:
                accelerator.print("✅ Model weights successfully loaded from checkpoint")
            else:
                accelerator.print("⚠️  Warning: Model weights did not change after loading checkpoint")

    print_fancy("Model restored.")

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    for epoch in tqdm(range(resume_epoch, config.get("num_epochs"))):
        current_loss = train_one_epoch(
            model, train_dl, loss_fn, optimizer, accelerator,
            max_batches=config.get("max_train_batches"),
        )
        accelerator.log({"avg_loss_per_epoch": current_loss})

        # --- ASVspoof 2019 dev ---
        asv19_scores, asv19_labels = compute_scores(
            asv19_dl, model, accelerator, max_batches=config.get("max_val_batches")
        )
        asv19dcf, asv19_eer, asv19_cllr = compute_antispoofing_metrics(asv19_scores, asv19_labels)
        accelerator.log(
            {"asv19_dev_dcf": asv19dcf, "asv19_dev_eer": asv19_eer, "asv19_dev_cllr": asv19_cllr},
            step=epoch,
        )
        print_fancy(f"asv19  eer={asv19_eer:.4f}  dcf={asv19dcf:.4f}")

        # --- ASVspoof 5 dev ---
        asv5_scores, asv5_labels = compute_scores(
            asv5_dl, model, accelerator, max_batches=config.get("max_val_batches")
        )
        asv5dcf, asv5_eer, asv5_cllr = compute_antispoofing_metrics(asv5_scores, asv5_labels)
        accelerator.log(
            {"asv5_dev_dcf": asv5dcf, "asv5_dev_eer": asv5_eer, "asv5_dev_cllr": asv5_cllr},
            step=epoch,
        )
        print_fancy(f"asv5   eer={asv5_eer:.4f}  dcf={asv5dcf:.4f}")

        # --- checkpoint ---
        checkpoint_name = f"{config.get('comet_run_name')}_epoch_{epoch}"
        checkpoint_path = os.path.join(config.get("checkpoint_base_path"), checkpoint_name)
        accelerator.save_state(checkpoint_path)

    accelerator.end_training()


if __name__ == "__main__":
    main()
