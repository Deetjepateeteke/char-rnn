import argparse
import json

from config import TrainConfig
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])
    args = parser.parse_args()

    with open(args.config) as f:
        raw = json.load(f)

    # Pydantic validates config data here.
    config = TrainConfig(**raw)

    if args.mode == "train":
        train(config)

if __name__ == "__main__":
    main()
