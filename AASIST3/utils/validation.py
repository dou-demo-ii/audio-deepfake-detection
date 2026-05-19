from typing import List, Tuple
import torch
from tqdm import tqdm


def compute_scores(
    dataloader,
    model,
    accelerator,
    max_batches: int = -1
) -> Tuple[List[float], List[int]]:

    model.eval()
    all_scores = []
    all_labels = []
    batch_count = 0

    for inputs, targets in tqdm(dataloader, desc="Model evaluation", leave=False):
        with torch.inference_mode():
            outputs = model(inputs)
            gathered_outputs, gathered_targets = accelerator.gather_for_metrics(
                (outputs.detach(), targets.detach())
            )
            scores = gathered_outputs[:, 1].cpu().numpy().ravel()

        all_scores.extend(scores.tolist())
        all_labels.extend(gathered_targets.tolist())
        batch_count += 1
        if (max_batches != -1) and (batch_count >= max_batches):
            break

    print(f"Evaluated {len(all_scores)} samples, {len(all_labels)} labels")
    return all_scores, all_labels