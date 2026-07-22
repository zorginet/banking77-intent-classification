"""A module for assessing the quality of models."""

import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

# Defining the device for computations: GPU (CUDA), MPS (Apple Silicon), or CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_predictions_for_pytorch(model, X, batch_size=32, device=None):
    """Universal inference for PyTorch models (MLP or BERT).

    Performs predictions based on the provided data, working with both HuggingFace
    style dictionaries (for BERT) and regular tensors (for MLP).

    Args:
        model: Trained PyTorch model in eval mode.
        X: Input data (DataLoader, Tensor, or numpy array).
        batch_size: Batch size for inference. Default is 32.
        device: Device for computations ("cpu", "cuda", "mps"). Default is chosen automatically.

    Returns:
        Tuple (all_preds, all_probs, targets_out):
            - all_preds: Predicted classes.
            - all_probs: Probabilities for each class.
            - targets_out: True labels or None.
    """
    # Set the model to evaluation mode and move it to the appropriate device
    model.eval()
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model.to(device)

    all_preds = []
    all_probs = []
    all_targets = []

    # Check the format of the input data and create a DataLoader if needed
    if isinstance(X, torch.utils.data.DataLoader):
        dataloader = X
    else:
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        else:
            X_tensor = X
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

    # Disable gradients for speed and memory efficiency
    with torch.no_grad():
        for batch in dataloader:
            # Process the batch depending on its format (HuggingFace dict, tuple, tensor)
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

                # Collect true labels if present
                if "labels" in batch:
                    all_targets.extend(batch["labels"].cpu().numpy())
                elif "label" in batch:
                    all_targets.extend(batch["label"].cpu().numpy())

                outputs = model(**inputs)

                # Extract logits from the model output
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    xb, yb = batch[0].to(device), batch[1].to(device)
                    all_targets.extend(yb.cpu().numpy())
                else:
                    xb = batch[0].to(device)
                logits = model(xb)

            else:
                xb = batch.to(device)
                logits = model(xb)

            # Convert logits to probabilities and class predictions
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_probs.extend(probs)

    # Form the output arrays
    targets_out = np.array(all_targets) if len(all_targets) > 0 else None
    return np.array(all_preds), np.array(all_probs), targets_out


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, device=None):
    """Calculates and prints model performance metrics on three datasets.

    Supports both classical models (Sklearn/CatBoost) and neural networks (PyTorch).

    Args:
        model: Trained model.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        device: Device for PyTorch model computations. Default is chosen automatically.
    """
    # Get predictions depending on the type of model
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)
    else:
        # For PyTorch models, perform inference on the specified device
        train_preds, train_probs, y_train_dl = get_predictions_for_pytorch(
            model, X_train, device=device
        )
        val_preds, val_probs, y_val_dl = get_predictions_for_pytorch(
            model, X_val, device=device
        )
        test_preds, test_probs, y_test_dl = get_predictions_for_pytorch(
            model, X_test, device=device
        )

        # Replace labels with those extracted from the model batches (if they were there)
        if y_train_dl is not None:
            y_train = y_train_dl
        if y_val_dl is not None:
            y_val = y_val_dl
        if y_test_dl is not None:
            y_test = y_test_dl

    # Calculate and print ROC-AUC to evaluate the quality of probabilities
    print(
        "ROC_AUC Train:",
        round(
            roc_auc_score(y_train, train_probs, multi_class="ovr", average="weighted"),
            4,
        ),
    )
    print(
        "ROC_AUC Val:",
        round(
            roc_auc_score(y_val, val_probs, multi_class="ovr", average="weighted"), 4
        ),
    )
    print(
        "ROC_AUC Test:",
        round(
            roc_auc_score(y_test, test_probs, multi_class="ovr", average="weighted"), 4
        ),
    )
    print("---")
    # Calculate Macro F1 for uniform evaluation of all classes
    print("Macro F1 Train:", round(f1_score(y_train, train_preds, average="macro"), 4))
    print("Macro F1 Val:", round(f1_score(y_val, val_preds, average="macro"), 4))
    print("Macro F1 Test:", round(f1_score(y_test, test_preds, average="macro"), 4))
    print("---")
    # Calculate Weighted F1 taking class frequency into account
    print(
        "Weighted F1 Train:",
        round(f1_score(y_train, train_preds, average="weighted"), 4),
    )
    print("Weighted F1 Val:", round(f1_score(y_val, val_preds, average="weighted"), 4))
    print(
        "Weighted F1 Test:", round(f1_score(y_test, test_preds, average="weighted"), 4)
    )


def get_metrics(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name,
    training_time,
    hyperparams=None,
    optimizer=None,
    device=None,
):
    """Collects key metrics of a trained model for comparison.

    Automatically extracts predictions and model parameters, regardless of its type
    (Sklearn/CatBoost or PyTorch), and calculates F1-Macro on all datasets.

    Args:
        model: Trained model.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        model_name: Model name for the registry.
        training_time: Training time in seconds.
        hyperparams: String with hyperparameters (if None, extracted automatically).
        optimizer: Optimizer for PyTorch models.
        device: Device for PyTorch model computations.

    Returns:
        dict: Dictionary with model metrics.
    """
    # Determine the device for computations if not provided
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    # Get predictions depending on the type of model
    if hasattr(model, "predict"):
        # For classical models (Sklearn, CatBoost)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)

        # Form a description of the parameters or use the provided one
        if hyperparams is None:
            try:
                raw_params = str(model.get_params())
                rounded_params = re.sub(
                    r"\b\d+\.\d+\b", lambda m: f"{float(m.group()):.3f}", raw_params
                )
                params = rounded_params[:100] + "..."
            except Exception:
                params = "N/A"
        else:
            params = hyperparams
    else:
        # For PyTorch models, perform inference on the specified device
        train_preds, _, y_train_dl = get_predictions_for_pytorch(
            model, X_train, device=device
        )
        val_preds, _, y_val_dl = get_predictions_for_pytorch(
            model, X_val, device=device
        )
        test_preds, _, y_test_dl = get_predictions_for_pytorch(
            model, X_test, device=device
        )

        # Replace labels with those extracted from the model batches (if they were there)
        if y_train_dl is not None:
            y_train = y_train_dl
        if y_val_dl is not None:
            y_val = y_val_dl
        if y_test_dl is not None:
            y_test = y_test_dl

        # Form a description of the model configuration and optimizer
        if hyperparams is None:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if total_params >= 1e6:
                params_str = f"{total_params / 1e6:.1f}M params"
            else:
                params_str = f"{total_params / 1e3:.1f}k params"

            model_class = model.__class__.__name__

            if optimizer is not None:
                opt_name = optimizer.__class__.__name__
                lr = optimizer.param_groups[0].get("lr", "N/A")
                wd = optimizer.param_groups[0].get("weight_decay", 0.0)
                params = f"{model_class} ({params_str}), {opt_name}: lr={lr}, wd={wd}"
            else:
                params = f"{model_class} ({params_str})"
        else:
            params = hyperparams

    # Calculate F1-Macro for all datasets
    train_f1 = round(f1_score(y_train, train_preds, average="macro"), 4)
    val_f1 = round(f1_score(y_val, val_preds, average="macro"), 4)
    test_f1 = round(f1_score(y_test, test_preds, average="macro"), 4)

    # Return a dictionary of metrics for the registry
    return {
        "Model": model_name,
        "Hyperparams": params,
        "Train F1 (Macro)": train_f1,
        "Val F1 (Macro)": val_f1,
        "Test F1 (Macro)": test_f1,
        "Time (sec)": round(training_time, 1),
    }


def save_metrics_to_registry(metrics_dict, filepath="../data/metrics_registry.csv"):
    """Saves model metrics to a shared CSV file.

    Overwrites the model entry if it already exists in the registry to ensure
    up-to-date information and avoid duplicate results.

    Args:
        metrics_dict: Dictionary with model metrics to save.
        filepath: Path to the CSV file of the metrics registry. Default is "../data/metrics_registry.csv".
    """
    # Ensure the directory for saving the file exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Prepare the new row of metrics as a DataFrame
    new_df = pd.DataFrame([metrics_dict])

    # Remove the old version of the model (if it exists) and add the new one
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = df[df["Model"] != metrics_dict["Model"]]
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    # Save the updated registry to the file
    df.to_csv(filepath, index=False)
    print(f" The metrics for '{metrics_dict['Model']}' have been saved to {filepath}")
