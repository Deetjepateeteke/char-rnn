import torch
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(model, optimizer, epoch, loss, char2idx, idx2char, config, path):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "config": config
    }, path)
    print(f"Checkpoint saved -> {path}")


def plot_losses(loss_history, xlabel: str, title: str = "Loss History"):
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.show()
