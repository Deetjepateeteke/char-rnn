import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from .model import CharRNN


def save_checkpoint(model, optimizer, epoch, loss, accuracy, char2idx, idx2char, config, path):
    """ Save the model's state with its metadata during training. """
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "config": config.model_dump()
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_model(model_path: str) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
    """
    Load a model with its metadata to continue training or make predictions.

    Args:
        model_path  (Path): The location the model should be loaded from.

    Returns:
        tuple[nn.Module, torch.optim.Optimizer, dict]: A tuple containing:
            - The loaded model
            - The model's optimizer
            - The model's metadata
    """
    checkpoint = torch.load(model_path, weights_only=False)

    char2idx, idx2char = checkpoint["char2idx"], checkpoint["idx2char"]
    config = checkpoint["config"]

    model = CharRNN(len(char2idx), config["hidden_size"], config["num_layers"], config["dropout"])
    model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr"])
    optimizer.load_state_dict(checkpoint["optim_state"])

    metadata = {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "accuracy": checkpoint["accuracy"],
        "char2idx": char2idx,
        "idx2char": idx2char,
        "config": config
    }

    print(f"Loaded model from '{model_path}'")

    return model, optimizer, metadata


def plot_losses(loss_history, title: str = "Loss History"):
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
