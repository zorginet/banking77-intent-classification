"""A module for defining model architectures and their training functions."""

import copy
from timeit import default_timer as timer

import torch
from sklearn.metrics import f1_score


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Runs a single training epoch.

    Args:
        model: The model to train.
        dataloader: The data loader for the training dataset.
        optimizer: The optimiser for updating weights.
        criterion: The loss function.
        device: The computing device (CPU/GPU).

    Returns:
        A tuple (average loss per epoch, macro F1-score).
    """
    # Set the model to training mode (enables dropout and batch normalization)
    model.train()
    running_loss = 0.0
    epoch_preds = []
    epoch_targets = []

    # Update model weights for each batch
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        epoch_targets.extend(yb.cpu().numpy())

    # Calculate average metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    macro_f1 = f1_score(epoch_targets, epoch_preds, average="macro")
    return epoch_loss, macro_f1


def evaluate_epoch(model, dataloader, criterion, device):
    """Evaluates the model on the validation set.

    Args:
        model: The model to evaluate.
        dataloader: The data loader for the validation dataset.
        criterion: The loss function.
        device: The computing device (CPU/GPU).

    Returns:
        A tuple (average loss per epoch, macro F1-score).
    """
    # Set the model to evaluation mode without gradients
    model.eval()
    running_loss = 0.0
    epoch_preds = []
    epoch_targets = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            running_loss += loss.item()
            epoch_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            epoch_targets.extend(yb.cpu().numpy())
    # Calculate average metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    macro_f1 = f1_score(epoch_targets, epoch_preds, average="macro")
    return epoch_loss, macro_f1


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=30,
    patience=5,
):
    """Trains the model with saving the best weights and early stopping.

    Args:
        model: The model to train.
        train_dl: DataLoader for the training dataset.
        val_dl: DataLoader for the validation dataset.
        optimizer: The optimiser for updating weights.
        criterion: The loss function.
        scheduler: The learning rate scheduler.
        device: The computing device (CPU/GPU).
        num_epochs: Number of training epochs.
        patience: Number of epochs without improvement before early stopping.

    Returns:
        A tuple (trained model, training losses, validation losses,
        training F1-score, validation F1-score, training time).
    """
    # Record the start time for measuring training duration
    train_time_start = timer()

    # Initialize lists to accumulate metrics per epoch
    train_losses = []
    val_losses = []
    train_macrof1 = []
    val_macrof1 = []

    # Initialize variables for tracking the best model
    best_val_macro_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Iterate over training epochs
    for epoch in range(num_epochs):
        # Perform one training and evaluation cycle
        epoch_train_loss, macro_f1_train = train_epoch(
            model, train_dl, optimizer, criterion, device
        )
        epoch_val_loss, macro_f1_val = evaluate_epoch(model, val_dl, criterion, device)

        # Accumulate metrics for further analysis
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_macrof1.append(macro_f1_train)
        val_macrof1.append(macro_f1_val)

        # Adjust the learning rate based on the validation metric
        scheduler.step(macro_f1_val)

        print(
            f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
            f"Train Macro F1: {macro_f1_train:.4f}, Val Macro F1: {macro_f1_val:.4f}"
        )

        # Check if the validation metric has improved
        if macro_f1_val > best_val_macro_f1:
            best_val_macro_f1 = macro_f1_val
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            # Increment the counter for early stopping
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Finish measuring training time
    train_time_end = timer()
    pure_train_time = round(train_time_end - train_time_start, 1)
    print(f"Training time: {pure_train_time} seconds")

    # Restore the best weights for the final model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_macrof1, val_macrof1, pure_train_time


def train_epoch_bert(model, dataloader, optimizer, device):
    """Performs one training epoch for the BERT model.

    Args:
        model: The BERT model to train.
        dataloader: The data loader for the training dataset.
        optimizer: The optimizer for updating weights.
        device: The computing device (CPU/GPU).

    Returns:
        A tuple (average loss per epoch, macro F1-score).
    """
    # Set the model to training mode (enables dropout and batch normalization)
    model.train()
    running_loss = 0.0
    epoch_preds = []
    epoch_targets = []

    # Update model weights for each batch
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        epoch_targets.extend(batch["labels"].cpu().numpy())

    # Calculate average metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    macro_f1 = f1_score(epoch_targets, epoch_preds, average="macro")
    return epoch_loss, macro_f1


def evaluate_epoch_bert(model, dataloader, device):
    """Evaluates the BERT model on the validation set.

    Args:
        model: The BERT model to evaluate.
        dataloader: The data loader for the validation dataset.
        device: The computing device (CPU/GPU).

    Returns:
        A tuple (average loss per epoch, macro F1-score).
    """
    # Set the model to evaluation mode without gradients
    model.eval()
    running_loss = 0.0
    epoch_preds = []
    epoch_targets = []
    # Disable gradients for faster computations and memory efficiency
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            epoch_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            epoch_targets.extend(batch["labels"].cpu().numpy())
    # Calculate average metrics for the epoch
    epoch_loss = running_loss / len(dataloader)
    macro_f1 = f1_score(epoch_targets, epoch_preds, average="macro")
    return epoch_loss, macro_f1


def train_model_bert(
    model,
    train_dl,
    val_dl,
    optimizer,
    scheduler=None,
    device="mps" if torch.backends.mps.is_available() else "cpu",
    num_epochs=5,
    patience=2,
):
    """Trains the BERT model with early stopping based on validation performance.

    Args:
        model: The BERT model to train.
        train_dl: DataLoader for the training dataset.
        val_dl: DataLoader for the validation dataset.
        optimizer: The optimizer for updating weights.
        scheduler: Learning rate scheduler for adaptive adjustment (optional).
        device: The computing device (CPU/GPU/MPS).
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs without improvement before early stopping.

    Returns:
        A tuple (trained model, training losses, validation losses, training F1, validation F1, training time).
    """
    # Record the start time for training duration calculation
    train_time_start = timer()

    # Store metrics for each epoch for analysis
    train_losses = []
    val_losses = []
    train_macrof1 = []
    val_macrof1 = []

    # Initialize variables for monitoring performance and early stopping
    best_val_macro_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Main training loop over multiple epochs
    for epoch in range(num_epochs):
        # Perform one training and evaluation epoch
        epoch_train_loss, macro_f1_train = train_epoch_bert(
            model, train_dl, optimizer, device
        )
        epoch_val_loss, macro_f1_val = evaluate_epoch_bert(model, val_dl, device)

        # Accumulate metrics for further analysis
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_macrof1.append(macro_f1_train)
        val_macrof1.append(macro_f1_val)

        # Adjust the learning rate based on the validation metric
        if scheduler is not None:
            scheduler.step(macro_f1_val)

        # Print training progress
        print(
            f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
            f"Train Macro F1: {macro_f1_train:.4f}, Val Macro F1: {macro_f1_val:.4f}"
        )

        # Save the best model and control early stopping
        if macro_f1_val > best_val_macro_f1:
            best_val_macro_f1 = macro_f1_val
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Finish training time measurement and print duration
    train_time_end = timer()
    pure_train_time = round(train_time_end - train_time_start, 1)
    print(f"Training time: {pure_train_time} seconds")

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Best model weights loaded with Val Macro F1: {best_val_macro_f1:.4f}")
    return model, train_losses, val_losses, train_macrof1, val_macrof1, pure_train_time
