import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from config import TrainConfig
from .dataset import CharDataSet
from .model import CharRNN
from .utils import plot_losses, save_checkpoint


def train(config: TrainConfig):
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # ---------------------------------------------------------------
    # 2. Data
    # ---------------------------------------------------------------
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = " ".join(f.readlines()).lower()

    dataset = CharDataSet(text, config.seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset):,} sequences")
    print(f"Batches/epoch: {len(dataloader):,}")

    # ---------------------------------------------------------------
    # 3. Model
    # ---------------------------------------------------------------
    model = CharRNN(
        vocab_size=dataset.vocab_size,
        hidden_size=config.hidden_size,
        dropout=config.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    # --------------------------------------------------------------
    # 4. Loss and optimizer
    # ---------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=config.lr)

    # ---------------------------------------------------------------
    # 5. Training loop
    # ---------------------------------------------------------------
    epoch_loss_history = []
    batch_loss_history = []

    for epoch in range(config.epochs):
        model.train()
        hidden = model.init_hidden(batch_size=config.batch_size, device=device)

        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, hidden = model(inputs, hidden)
            hidden = hidden.detach()
            preds = torch.softmax(logits, dim=1).argmax(dim=1)

            loss = criterion(logits, targets)

            # --- Backward pass ---
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            batch_loss_history.append(loss.item())


            # Log every N batches
            if (batch_idx + 1) % config.log_every == 0:
                avg = epoch_loss / (batch_idx + 1)
                tqdm.write(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx+1:}/{len(dataloader):>5} | "
                    f"Loss {avg:.4f}"
                )
                plot_losses(batch_loss_history, xlabel="Batches")

        # --- End of epoch ---
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_loss_history.append(avg_epoch_loss)
        tqdm.write(
            f"\nEpoch {epoch+1} complete - "
            f"Loss {avg_epoch_loss:.4f}"
        )

        plot_losses(batch_loss_history, title="Batch Loss History", xlabel="Batches")

        if (epoch + 1) % config.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss,
                char2idx=dataset.char2idx,
                idx2char=dataset.idx2char,
                config=config,
                path=config.checkpoint_dir / f"epoch_{epoch:03}.pt"
            )

    # --- Final save and plot ---
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=loss,
        char2idx=dataset.char2idx,
        idx2char=dataset.idx2char,
        config=config,
        path=config.checkpoint_dir / "final.pt"
    )

    plot_losses(epoch_loss_history, xlabel="Epochs")
    print("--- Training complete ---")
