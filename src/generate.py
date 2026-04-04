from typing import Optional

import torch
from tqdm.auto import tqdm

from src.dataset import decode, encode
from src.utils import load_model


def generate(model_path: str,
             length: int = 100,
             temperature: float = 0.7,
             prompt: Optional[str] = None) -> str:
    """
    Generate text autoregressively from a trained char-rnn.

    The generation process has two phases:
        1. **Priming** - 'Warming up' the model's hidden state.
        2. **Sampling** - The model generates `length` new characters.

    Args:
        - model_path    (str): Path to the saved checkpoint.
        - length        (int): The amount of characters the model must generate.
        - temperature   (float): Controls sampling randomness.
        - prompt        (Optional[str]): An optional starting point for
            the model to start generating.

    Returns:
        str: The decoded output string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inferencing on: {device}")

    # --- Load checkpoint ---
    model, _, metadata = load_model(model_path)
    model = model.to(device)

    char2idx = metadata["char2idx"]
    idx2char = metadata["idx2char"]

    # Encode the given prompt.
    if prompt is not None:
        token_ids = encode(prompt, char2idx)
    else:
        # Start generating from a random starting point when no prompt is set.
        token_ids = torch.randint(0, model.vocab_size, size=(1,)).tolist()

    # --- Prime hidden state ---
    hidden = model.init_hidden(batch_size=1, device=device)

    model.eval()
    with torch.inference_mode():
        for x in token_ids[:-1]:
            x = torch.tensor(x).reshape(1, 1).to(device)
            _, hidden = model(x, hidden)

    # --- Autoregressive generation ---
    current_idx = torch.tensor(token_ids[-1]).reshape(1, 1)

    with torch.inference_mode():
        for _ in tqdm(range(length), desc="Generating"):
            logits, hidden = model(current_idx.to(device), hidden)

            probs = torch.softmax(logits / temperature, dim=-1)
            current_idx = torch.multinomial(probs.squeeze(), num_samples=1).unsqueeze(0)

            token_ids.append(current_idx.item())

    output = decode(token_ids, idx2char)
    print(output)
    return output
