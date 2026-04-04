from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class TrainConfig(BaseModel):
    """
    Configuration for training char-rnn.
    Handles data paths, model architecture and training hyperparameters.
    """
    # --- Data ---
    data_path: str = Field("data/shakespeare.txt")
    seq_len: int = Field(50, gt=0)

    # --- Model Architecture ---
    hidden_size: int = Field(16, gt=0)
    num_layers: int = Field(1, gt=0)
    dropout: float = Field(0.3, ge=0.0, le=1.0)

    # --- Training ---
    epochs: int = Field(10, gt=0)
    batch_size: int = Field(32, gt=0)
    lr: float = Field(1e-3, gt=0.0)

    # --- Checkpointing ---
    save_every: int = Field(1, gt=0)
    checkpoint_dir: str = Field("checkpoints")

    @field_validator("data_path", mode="plain")
    @classmethod
    def file_must_exist(cls, v: str) -> Path:
        """ Ensures the given path exists. """
        if not Path(v).exists():
            raise ValueError(f"data_path '{v}' does not exist")
        return Path(v)

    @field_validator("checkpoint_dir", mode="plain")
    @classmethod
    def dir_must_exist(cls, v: str) -> Path:
        """ Automatically creates the checkpoint folder if it doesn't exist yet. """
        dir_path = Path(v)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
