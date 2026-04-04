import argparse
import json

from config import TrainConfig
from src.train import train
from src.generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "generate"])

    # Only required if mode == "train"
    parser.add_argument("--config", type=str, default="config.json")

    # Only applicable if mode == "generate"
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("-l", type=int, default=100)
    parser.add_argument("-t", type=float, default=0.7)
    parser.add_argument("-p", type=str, required=False)
    args = parser.parse_args()

    if args.mode == "train":
        with open(args.config) as f:
            raw = json.load(f)

        # Pydantic validates config data here.
        config = TrainConfig(**raw)
        train(config)

    elif args.mode == "generate":
        generate(
            model_path=args.model,
            length=args.l,
            temperature=args.t,
            prompt=args.p
        )


if __name__ == "__main__":
    main()
