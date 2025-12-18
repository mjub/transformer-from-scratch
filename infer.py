import argparse
import sys

import torch

from src import aux, train


class GenerativeRun(train.Run):
    @torch.no_grad()
    def generate(self, text, temperature=0.8, top_k=50, device="cpu"):
        self.model.to(device)
        self.model.eval()

        START_TOKEN = self.tokenizer.encode("<|startoftext|>").ids[0]
        END_TOKEN = self.tokenizer.encode("<|endoftext|>").ids[0]

        input_ids = torch.tensor(
            [self.tokenizer.encode(text).ids if text else [START_TOKEN]],
            device=device,
        )

        while True:
            # Keep no more than max_position_embeddings tokens
            if input_ids.size(1) > self.config.max_position_embeddings:
                input_ids = input_ids[:, -self.config.max_position_embeddings :]

            next_token = self.model.next_token(
                input_ids, temperature=temperature, top_k=top_k
            )
            if next_token[0] == END_TOKEN:
                return

            yield self.tokenizer.decode(next_token[0].tolist())

            input_ids = torch.cat([input_ids, next_token], dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from a trained Transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python infer.py -r runs/nlab-gpt-large.pt "A functor is"
  python infer.py -r runs/nlab-gpt-large.pt -n 100 -t 0.8
  python infer.py -r runs/nlab-gpt-large.pt --set max_position_embeddings=512
""",
    )
    parser.add_argument(
        "-r",
        "--run",
        required=True,
        help="path to a run checkpoint",
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
        default=0.8,
        help="sampling temperature (default: 0.8)",
        metavar="TEMP",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=50,
        help="top_k value (default: 50)",
        metavar="K",
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
        "text",
        nargs="*",
        help="starting text for generation",
        metavar="TEXT",
    )
    args = parser.parse_args()

    run = GenerativeRun.from_file(args.run)

    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = " ".join(args.text)
    output = open(args.output, "w") if args.output else sys.stdout

    # We write the text given as input
    output.write(text)
    output.flush()

    try:
        for n, token in enumerate(
            run.generate(
                text, temperature=args.temperature, top_k=args.top_k, device=device
            )
        ):
            if args.tokens and n >= args.tokens:
                break
            output.write(token)
            output.flush()
    except KeyboardInterrupt:
        pass

    output.write("\n")

    if args.output:
        output.close()
