import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from config import TrainConfig
from src.dataset import CharDataset
from src.model import CharRNN
from src.utils import plot_losses, save_checkpoint


def train(config: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # --- Data loading ---
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = " ".join(f.readlines()).lower()

    dataset = CharDataset(text, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset):,} sequences")
    print(f"Batches/epoch: {len(dataloader):,}")

    # --- Initialize model ---
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    # --- Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    accuracy_fn = Accuracy(task="multiclass", num_classes=model.vocab_size)

    # --- Training loop ---
    loss_history = []
    acc_history = []

    for epoch in range(config.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0

        hidden = model.init_hidden(batch_size=config.batch_size, device=device)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        last_update = float("-inf")
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # logits: (batch_size, seq_len, vocab_size)
            logits, hidden = model(inputs, hidden)
            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            hidden = hidden.detach()

            loss = criterion(logits.permute(0, 2, 1), targets)

            # --- Backward pass ---
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_fn(preds, targets)

            # Update metrics every 0.3s
            if pbar.format_dict["elapsed"] - last_update >= 0.3:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_acc = epoch_acc / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "accuracy": f"{avg_acc * 100:.2f}%"})
                last_update = pbar.format_dict["elapsed"]

        # --- End of epoch ---
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_acc = epoch_acc / len(dataloader)
        loss_history.append(avg_epoch_loss)
        acc_history.append(avg_epoch_acc)

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_epoch_loss,
                accuracy=avg_epoch_acc,
                char2idx=dataset.char2idx,
                idx2char=dataset.idx2char,
                config=config,
                path=config.checkpoint_dir / f"epoch_{epoch + 1:03}.pt"
            )

    # --- Final save and plot ---
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=avg_epoch_loss,
        accuracy=avg_epoch_acc,
        char2idx=dataset.char2idx,
        idx2char=dataset.idx2char,
        config=config,
        path=config.checkpoint_dir / "final.pt"
    )

    plot_losses(loss_history)
    print("--- Training complete ---")
