import os

import hydra
import torch, torch.nn as nn
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from model import aasist3
from datasets import print_fancy
from datasets import ASVspoof2019Eval, ASVspoof2021DF, ASVspoof2021LA, ASVspoof5Eval
from utils import compute_scores, compute_antispoofing_metrics


@hydra.main(config_path="configs", config_name="test", version_base="1.1")
def main(config):
    print_fancy(str(OmegaConf.to_container(config)))
    accelerator = Accelerator()

    print_fancy("Accelerator loaded")
    asvspoof2019 = ASVspoof2019Eval(
        root_dir=config['data']['asvspoof2019_eval']['root_dir'],
        meta_path=config['data']['asvspoof2019_eval']['meta_path']
    )
    asvspoof2021_df = ASVspoof2021DF(
        root_dir=config['data']['asvspoof2021_df']['root_dir'],
        meta_path=config['data']['asvspoof2021_df']['meta_path']
    )
    asvspoof2021_la = ASVspoof2021LA(
        root_dir=config['data']['asvspoof2021_la']['root_dir'],
        meta_path=config['data']['asvspoof2021_la']['meta_path']
    )
    asvspoof5 = ASVspoof5Eval(
        root_dir=config['data']['asvspoof5_test']['root_dir'],
        meta_path=config['data']['asvspoof5_test']['meta_path']
    )

    print_fancy("test datasets loaded")

    asv19_dl = DataLoader(
        asvspoof2019,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=config['shuffle']
    )

    asv5_dl = DataLoader(
        asvspoof5,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=config['shuffle']
    )

    asv21_df_dl = DataLoader(
        asvspoof2021_df,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=config['shuffle']
    )

    asv21_la_dl = DataLoader(
        asvspoof2021_la,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=config['shuffle']
    )

    print_fancy('dataloaders initialised')

    model = aasist3.from_pretrained("MTUCI/AASIST3", cache_dir="/data/home/borodin_sam/another_workspace/AASIST3/weights")
    model_weights_before = {name: param.clone().detach() for name, param in model.named_parameters()}
    checkpoint_path = config.get("weights_path")
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)

    weights_changed = False
    for name, param in model.named_parameters():
        if not torch.equal(model_weights_before[name], param):
            weights_changed = True
            accelerator.print(f"Weights changed for parameter: {name}")
            break

    if weights_changed:
        accelerator.print("✅ Model weights successfully loaded from checkpoint")
    else:
        accelerator.print("⚠️ Warning: Model weights did not change after loading checkpoint")
        print_fancy("Model restorated.")

    asv19_dl, asv5_dl, asv21_df_dl, asv21_la_dl, model = accelerator.prepare(
        asv19_dl,
        asv5_dl,
        asv21_df_dl,
        asv21_la_dl,
        model
    )

    print_fancy("Important entities created")

    asv19_scores, asv19_labels = compute_scores(asv19_dl, model, accelerator, max_batches=config.get("max_val_batches"))
    asv19dcf, asv19_eer, asv19_cllr = compute_antispoofing_metrics(asv19_scores, asv19_labels)
    print_fancy(f"asv19 eer{asv19_eer},\n asv19 dcf {asv19dcf}")

    asv5_scores, asv5_labels = compute_scores(asv5_dl, model, accelerator, max_batches=config.get("max_val_batches"))
    asv5dcf, asv5_eer, asv5_cllr = compute_antispoofing_metrics(asv5_scores, asv5_labels)
    print_fancy(f"asv5 eer{asv5_eer},\n asv5 dcf {asv5dcf}")

    asv21_df_scores, asv21_df_labels = compute_scores(asv21_df_dl, model, accelerator, max_batches=config.get("max_val_batches"))
    asv21_dfdcf, asv21_df_eer, asv21_df_cllr = compute_antispoofing_metrics(asv21_df_scores, asv21_df_labels)
    print_fancy(f"asv21 df eer{asv21_df_eer},\n asv21 df dcf {asv21_dfdcf}")

    asv21_la_scores, asv21_la_labels = compute_scores(asv21_la_dl, model, accelerator, max_batches=config.get("max_val_batches"))
    asv21_ladcf, asv21_la_eer, asv21_la_cllr = compute_antispoofing_metrics(asv21_la_scores, asv21_la_labels)
    print_fancy(f"asv21 la eer{asv21_la_eer},\n asv21 la dcf {asv21_ladcf}")


if __name__ == "__main__":
    main()