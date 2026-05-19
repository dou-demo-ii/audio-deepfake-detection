from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, accelerator, max_batches=-1):
    """
    Trains the model for one epoch.
    Args:
        model: The neural network to train.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        accelerator: Accelerator (from HuggingFace Accelerate).
        max_batches: Maximum number of batches to process.
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = batch

        with accelerator.accumulate(model):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        accelerator.log({"loss": loss.item()})
        num_batches += 1
        if (max_batches != -1) and (num_batches > max_batches):
            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss