import os

import hydra
import torch, torch.nn as nn
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


from model import aasist3
from datasets import print_fancy
from datasets import ASVspoof2019Dev, ASVspoof2019Train, ASVspoof5Dev, MAILABS, MLAAD, ASVspoof5Train
from utils import train_one_epoch, compute_scores, compute_antispoofing_metrics
from safetensors.torch import save_file


PATH_TO_SAVE = "/data/home/borodin_sam/another_workspace/AASIST3/weights/train/FINAL/model.safetensors"


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config):
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_IB_DISABLE'] = '1'
    print_fancy(str(OmegaConf.to_container(config)))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config["find_unused_parameters"])

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
    )

    print_fancy("Accelerator loaded")

    asvspoof19train = ASVspoof2019Train(
        root_dir=config['data']["asvspoof2019_train"]["root_dir"],
        meta_path=config['data']["asvspoof2019_train"]["meta_path"],
    )
    asvspoof24train = ASVspoof5Train(
        root_dir=config['data']["asvspoof5_train"]["root_dir"],
        meta_path=config['data']["asvspoof5_train"]["meta_path"],
    )
    mlaad = MLAAD(
        root_dir=config['data']["mlaad"]["root_dir"],
    )
    mailabs = MAILABS(
        root_dir=config['data']["m_ailabs"]["root_dir"],
    )
    train_dataset = ConcatDataset([asvspoof19train, asvspoof24train, mlaad, mailabs])

    print_fancy("train datasets loaded")

    asvspoof5dev = ASVspoof5Dev(
        root_dir=config['data']["asvspoof5_dev"]["root_dir"],
        meta_path=config['data']['asvspoof5_dev']['meta_path']
    )

    asvspoof19dev = ASVspoof2019Dev(
        root_dir=config['data']['asvspoof2019_dev']["root_dir"],
        meta_path=config['data']['asvspoof2019_dev']['meta_path']
    )

    print_fancy('validation datasets loaded')

    train_dl = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    asv19_dl = DataLoader(
        asvspoof19dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    asv5_dl = DataLoader(
        asvspoof5dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    print_fancy('dataloaders initialised')

    loss_fn = nn.CrossEntropyLoss()

    model = aasist3(load_pretrained=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=1e-7,
        weight_decay=0
    )

    train_dl, asv19_dl, asv5_dl, loss_fn, model, optimizer = accelerator.prepare(
        train_dl,
        asv19_dl,
        asv5_dl,
        loss_fn,
        model,
        optimizer
    )

    print_fancy("Important entities created")

    checkpoint_path = config.get("checkpoint_path", -1)
    resume_epoch = 0
    if config.get("resume_from_checkpoint"):
        checkpoint_path = config.get("resume_from_checkpoint")
        if os.path.exists(checkpoint_path):
            model_weights_before = {name: param.clone().detach() for name, param in model.named_parameters()}
            accelerator.print(f"Restoring checkpoint from {checkpoint_path}")
            accelerator.load_state(checkpoint_path)

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
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.push_to_hub("MTUCI/AASIST3")
    # save_file(unwrapped_model.state_dict(), PATH_TO_SAVE)


if __name__ == "__main__":
    main()