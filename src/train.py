import datetime

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm.auto import tqdm

from config import TrainConfig
from src.dataset import CharDataset
from src.model import CharRNN
from src.utils.checkpointing import save_checkpoint


def train(config: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter(f"runs/{timestamp}/train")
    val_writer = SummaryWriter(f"runs/{timestamp}/validation")

    # --- Data loading ---
    with open(config.data_path, "r", encoding="utf-8") as f:
        text = " ".join(f.readlines()).lower()

    # 80/20 train-test-split
    split_idx = int(len(text) * 0.8)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = CharDataset(train_text, config.seq_len)
    val_dataset = CharDataset(val_text, config.seq_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Dataset size: {len(train_dataset):,} sequences")
    print(f"Batches/epoch: {len(train_dataloader):,}")

    # --- Initialize model ---
    model = CharRNN(
        vocab_size=train_dataset.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        device=device
    ).to(device)

    dummy_input = torch.zeros(config.batch_size, config.seq_len, dtype=torch.int64).to(device)
    dummy_hidden = model.init_hidden(batch_size=config.batch_size, device=device)
    train_writer.add_graph(model, (dummy_input, dummy_hidden))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # --- Loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)

    accuracy_fn = Accuracy(task="multiclass", num_classes=model.vocab_size).to(device)

    scaler = GradScaler(device)

    # --- Training loop ---
    for epoch in range(config.epochs):
        model.train()

        train_loss = 0.0
        train_acc = 0.0

        hidden = model.init_hidden(config.batch_size, device=device)

        last_update = float("-inf")
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device):
            # logits: (batch_size, seq_len, vocab_size)
                logits, hidden = model(inputs, hidden)
                loss = criterion(logits.permute(0, 2, 1), targets)

            preds = torch.softmax(logits, dim=2).argmax(dim=2)
            hidden = hidden.detach()

            # --- Backward pass ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc += accuracy_fn(preds, targets).item()

            # Update metrics every 0.3s
            if pbar.format_dict["elapsed"] - last_update >= 0.3:
                loss = train_loss / (batch_idx + 1)
                acc = train_acc / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{loss:.4f}", "accuracy": f"{acc * 100:.2f}%"})
                last_update = pbar.format_dict["elapsed"]

        # Validation
        model.eval()
        with torch.inference_mode():
            val_loss = 0.0
            val_acc = 0.0

            hidden = model.init_hidden(config.batch_size, device=device)

            last_update = float("-inf")
            val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f"Validating")
            for batch_idx, (inputs, targets) in val_pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits, hidden = model(inputs, hidden)
                preds = torch.softmax(logits, dim=2).argmax(dim=2)

                val_loss += criterion(logits.permute(0, 2, 1), targets).item()
                val_acc += accuracy_fn(preds, targets).item()

                # Update metrics every 0.3s
                if pbar.format_dict["elapsed"] - last_update >= 0.3:
                    loss = val_loss / (batch_idx + 1)
                    acc = val_acc / (batch_idx + 1)
                    pbar.set_postfix({"loss": f"{loss:.4f}", "accuracy": f"{acc * 100:.2f}%"})
                    last_update = pbar.format_dict["elapsed"]

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        # Update lr scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        train_writer.add_scalar("graph/learning_rate", current_lr, global_step=epoch)

        train_writer.add_scalar("loss", train_loss, global_step=epoch)
        train_writer.add_scalar("accuracy", train_acc, global_step=epoch)
        val_writer.add_scalar("loss", val_loss, global_step=epoch)
        val_writer.add_scalar("accuracy", val_acc, global_step=epoch)

        # Build weight and gradient histograms
        for name, param in model.named_parameters():
            train_writer.add_histogram(f"weights/{name}", param, global_step=epoch)

            if param.grad is not None:
                train_writer.add_histogram(f"gradients/{name}", param.grad, global_step=epoch)

        train_writer.flush()

        if (epoch + 1) % config.save_every == 0 and epoch + 1 != config.epochs:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                accuracy=val_acc,
                char2idx=train_dataset.char2idx,
                idx2char=train_dataset.idx2char,
                config=config,
                path=config.checkpoint_dir / f"epoch_{epoch + 1:03}.pt"
            )

    save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                accuracy=val_acc,
                char2idx=train_dataset.char2idx,
                idx2char=train_dataset.idx2char,
                config=config,
                path=config.checkpoint_dir / f"final.pt"
            )

    train_writer.close()
    print("--- Training complete ---")
