import argparse

import tokenizers
import torch

import train

START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"


def generate(
    model,
    text,
    tokens=None,
    temperature=0.7,
    device=None,
    tokenizer="data/tokenizer.json",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)

    if isinstance(tokenizer, str):
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer)

    input_ids = torch.tensor([tokenizer.encode(text or START_TOKEN).ids], device=device)

    count = 0
    for token_ids in model.generate(input_ids, temperature=temperature, top_k=50):
        decoded = tokenizer.decode(token_ids[0].tolist())
        count += 1

        if decoded == END_TOKEN:
            return
        if tokens is not None and count > tokens:
            return

        yield decoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text using a model created with train.py",
        epilog=f"If no text is provided, the model will generate a brand new document using {START_TOKEN} by default",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="path to the model checkpoint",
        metavar="PATH",
    )
    parser.add_argument(
        "-n",
        "--tokens",
        type=int,
        default=None,
        help=f"number of tokens to generate (default: until {END_TOKEN} is emitted)",
        metavar="COUNT",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="temperature for the generation",
        metavar="TEMP",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="set the device on which the model should run (default: the best available)",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="alternative path to a tokenizer.json file (default: use the path indicated in the model configuration)",
        metavar="PATH",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help=f"starting text (default: {START_TOKEN})",
        metavar="TEXT",
    )
    args = parser.parse_args()

    run = train.Run.from_file(args.model)
    tokenizer_path = args.tokenizer or run.config.tokenizer

    text = " ".join(args.text)
    print(text, end="", flush=True)

    for token in generate(
        run.model,
        text,
        tokens=args.tokens,
        temperature=args.temperature,
        device=args.device,
        tokenizer=tokenizer_path,
    ):
        print(token, end="", flush=True)
