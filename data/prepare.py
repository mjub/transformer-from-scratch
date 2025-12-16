import argparse
import html
import itertools
import logging as log
import os
import pathlib
import random
import re
import sys

import torch
import tqdm

sys.path.append("src")
import aux

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import tokenizers

NORMALIZE_PATTERNS = list(
    map(
        lambda x: (re.compile(x[0]), x[1]),
        [
            # Strip special anchors
            (r"\[\[![^\]]+\]\]", r""),
            # Strip internal links
            (r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]", r"\1"),
            # Strip bold and italics
            # Note: we can't strip _ without risking to break math environments
            (r"\*(\**)([^\W\*](?:[^\*]|(?<=\\)\*)*)(?<!\\)\*(?(1)\1|(?!\*))", r"\2"),
            # Strip references
            (r"\{#[^\}]+}", ""),
            # Strip multiple newlines
            (r"\n{3,}", r"\n\n"),
            # Strip multiple spaces
            (r" {2,}", r" "),
        ],
    )
)

SPLIT_PATTERN = "|".join(
    [
        # Split at LaTeX commands
        r"\\[a-zA-Z][a-zA-Z0-9]*",
        # Split at escaped characters
        r"\\[^a-zA-Z]",
        # Split at math environments
        r"\$+|\\\[|\\\]",
        # Split at delimiters
        r"[\(\)\[\]\{\}]",
        # Split at Markdown operators
        r"[#*`]+",
        # Split at numbers
        r"\d+",
        # Split at words
        r"\w+",
        # Split at spaces
        r"\s+",
        # Split at punctuation
        r"[^\s\w]",
    ]
)


def normalize(document):
    document = html.unescape(document)
    for pattern, repl in NORMALIZE_PATTERNS:
        document = pattern.sub(repl, document)
    return document


def prepare_data(config, source_dir):
    pages = list(pathlib.Path(source_dir).rglob("*.md"))

    random.seed(config.seed)
    random.shuffle(pages)

    documents = []

    log.info(f"📚 Loading {len(pages):,} pages...")
    for page in tqdm.tqdm(pages):
        try:
            with open(page.parent / "name", "r", encoding="utf-8") as fd:
                title = fd.read().strip()
            with open(page, "r", encoding="utf-8") as fd:
                body = fd.read().strip()
            documents.append(normalize(f"# {title}\n\n{body}"))
        except Exception as e:
            log.warning(f"⚠️  Skipping {page}: {e.__class__.name__}: {e}")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(SPLIT_PATTERN, behavior="isolated"),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    log.info("🔤 Training BPE tokenizer...")
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=config.vocab_size, special_tokens=config.special_tokens
    )
    tokenizer.train_from_iterator(documents, trainer=trainer, length=len(documents))
    tokenizer.save(config.tokenizer_path)
    log.info(f"✅ Tokenizer trained and saved to {config.tokenizer_path}")
    log.info(f"    → Vocab size: {tokenizer.get_vocab_size():,}")

    log.info("⚙️  Encoding the data now...")
    data = torch.tensor(
        list(
            itertools.chain.from_iterable(
                (
                    x.ids
                    for x in tokenizer.encode_batch_fast(
                        [
                            f"<|startoftext|>{document}<|endoftext|>"
                            for document in documents
                        ]
                    )
                )
            )
        )
    )
    log.info(f"📦 Encoded {len(data):,} tokens")

    n = int(len(data) * 0.9)
    torch.save(data[:n].clone(), config.train_data)
    torch.save(data[n:].clone(), config.val_data)

    log.info(f"✅ Done! Train: {n:,} tokens | Val: {len(data) - n:,} tokens")
    log.info(f"    → Training data saved at: {config.train_data}")
    log.info(f"    → Validation data saved at: {config.val_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="path to the configuration JSON file",
        metavar="PATH",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        default="nlab-content",
        help="path to the source directory (default: nlab-content)",
        metavar="PATH",
    )
    args = parser.parse_args()

    log.basicConfig(
        level=log.INFO,
        format="\033[2m%(asctime)s\033[0m \033[1m\033[36m%(levelname)s\033[0m \033[1m[%(name)s]\033[0m %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            log.FileHandler("data/prepare.log"),
            log.StreamHandler(),
        ],
        force=True,
    )

    log.info(f"🚀 Starting data preparation with config: {args.config}")
    prepare_data(aux.load_config(args.config, validate=False), args.source_dir)
