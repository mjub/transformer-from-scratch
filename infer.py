import argparse
import sys

import torch

from src import aux, train

START_TOKEN = "< start of text >"
END_TOKEN = "< end of text >"


def generate(
    model,
    tokenizer,
    text=None,
    tokens=None,
    temperature=0.7,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)

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
        description="Generate text from a trained Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python infer.py -m checkpoint.pt "A functor is"
  python infer.py -m checkpoint.pt -n 100 -t 0.8
  python infer.py -m checkpoint.pt --set max_position_embeddings=512
""",
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
        help="number of tokens to generate (default: until end token)",
        metavar="COUNT",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (default: 0.7)",
        metavar="TEMP",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device to run on (default: cuda if available)",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write output to file instead of stdout",
        metavar="PATH",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="override config values with key=value",
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="starting text for generation",
        metavar="TEXT",
    )
    args = parser.parse_args()

    run = train.Run.from_file(args.model)
    try:
        aux.apply_overrides(run.config, **dict(s.split("=", 1) for s in args.set))
    except aux.ConfigError as e:
        parser.error(str(e))

    text = " ".join(args.text)
    output = open(args.output, "w") if args.output else sys.stdout

    output.write(text)
    for token in generate(
        run.model,
        run.tokenizer,
        text=text or None,
        tokens=args.tokens,
        temperature=args.temperature,
        device=args.device,
    ):
        output.write(token)
        output.flush()

    output.write("\n")

    if args.output:
        output.close()
