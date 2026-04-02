import torch
from torch import nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dropout: float=0.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size= hidden_size

        self.rnn = nn.RNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh"
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x, hidden)

        output = self.dropout(output)

        logits = self.fc(output)

        return logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """ Initialize the hidden state to zeros. """
        return torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float32, device=device)
