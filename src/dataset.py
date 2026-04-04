import torch
from torch.utils.data import Dataset


def build_vocab(text: str) -> tuple[dict, dict]:
    """
    Build character-to-index and index-to-character mappings.

    Args:
        text (str): The text the vocabulary is build from.

    Returns:
        tuple[dict, dict]: character-to-idx and idx-to-character mappings
    """
    chars = sorted(set(text))

    char2idx = {char: idx for idx, char in enumerate(chars)}
    idx2char = {idx: char for idx, char in enumerate(chars)}
    return char2idx, idx2char


def encode(text: str, char2idx: dict) -> list[int]:
    """ Convert a string to an encoded list of indices. """
    return [char2idx[char] for char in text]


def decode(indices: list[int], idx2char: dict) -> str:
    """ Convert an encoded list of indices to a string. """
    return "".join([idx2char[idx] for idx in indices])


class CharDataset(Dataset):
    """
    Sliding window dataset over an encoded character sequence.

    Each sample is a (input, target) pair, where:
        - input is a sequence of length seq_length
        - target is the same sequence shifted right by one character
    """
    def __init__(self, text, seq_length: int):
        self.seq_length = seq_length

        self.char2idx, self.idx2char = build_vocab(text)
        self.vocab_size = len(self.char2idx)

        self.data = torch.tensor(encode(text, self.char2idx), dtype=torch.long)

    def __len__(self) -> int:
        """ Return the amount of batches, not the amount of chars. """
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ Return the idx'th sample in the dataset. """
        x = self.data[idx: idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return x, y
