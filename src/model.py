import torch
import torch.nn as nn
import torch.nn.functional as F


class CharRNN(nn.Module):
    """
    Character-level text predictor using a vanilla Elman RNN.

    Architecture:
        One-hot Tensors -> RNN -> Dropout -> Linear

    Args:
        vocab_size  (int): Number of unique characters in the dataset.
        hidden_size (int): The amount of hidden units in the RNN layer.
        num_layers  (int): Number of stacked RNN layers.
        dropout     (float): Dropout rate applied between RNN layers
                             and before the Linear fc. Ignored if num_layers=1.
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int,
                 device: torch.device,
                 dropout: float = 0.0,):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # --- Layers ---
        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            device=device
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, x, hidden=None) -> tuple[torch.Tensor, torch.Tensor]:
        """ Define a forward step in the model, which returns logits and a new hidden state. """
        # x: (batch_size, seq_len) -> raw integer indices
        # x_onehot: (batch_size, seq_len, vocab_size) -> one-hot encodings
        x_onehot = F.one_hot(x, num_classes=self.vocab_size).float()

        output, hidden = self.rnn(x_onehot, hidden)
        output = self.dropout(output)
        logits = self.fc(output)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """ Initialize the hidden state to zeros. """
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
