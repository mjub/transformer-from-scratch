import html
import itertools
import json
import os
import pathlib
import random
import re
import types

import torch
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import tokenizers

SOURCE_DIR = "nlab-content"

NORMALIZE_PATTERNS = list(
    map(
        lambda x: (re.compile(x[0]), x[1]),
        [
            (r"\[\[![^\]]+\]\]", r""),
            (r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]", r"\1"),
            (r"\{#[^\}]+}", ""),
            (r"\n{3,}", r"\n\n"),
            (r" {2,}", r" "),
            (r"\t", r"    "),
        ],
    )
)

SPLIT_PATTERN = "|".join(
    [
        r"\\[a-zA-Z][a-zA-Z0-9]*",
        r"\\[^a-zA-Z]",
        r"\$+",
        r"[\(\)\[\]\{\}]",
        r"[#*`]+",
        r"\d+",
        r"\w+",
        r"\s+",
        r"[^\s\w]",
    ]
)


def normalize(document):
    document = html.unescape(document)
    for pattern, repl in NORMALIZE_PATTERNS:
        document = pattern.sub(repl, document)
    return document


def prepare_data(config):
    pages = list(pathlib.Path(SOURCE_DIR).rglob("*.md"))

    random.seed(config.seed)
    random.shuffle(pages)

    documents = []

    print(f"📚 Loading {len(pages):,} pages...")
    for page in tqdm.tqdm(pages):
        try:
            with open(page.parent / "name", "r", encoding="utf-8") as fd:
                title = fd.read().strip()
            with open(page, "r", encoding="utf-8") as fd:
                body = fd.read().strip()
            documents.append(normalize(f"# {title}\n\n{body}"))
        except Exception as e:
            print(f"⚠️ Skipping {page}: {e}")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(SPLIT_PATTERN, behavior="isolated"),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    print("🔤 Training BPE tokenizer...")
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=config.vocab_size, special_tokens=config.special_tokens
    )
    tokenizer.train_from_iterator(documents, trainer=trainer, length=len(documents))
    tokenizer.save("data/tokenizer.json")
    print(f"🎯 Tokenizer trained, vocab size: {tokenizer.get_vocab_size():,}")

    print("⚙️ Encoding corpus...")
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
    print(f"📦 Encoded {len(data):,} tokens")

    n = int(len(data) * 0.9)
    torch.save(data[:n].clone(), "data/train_data.bin")
    torch.save(data[n:].clone(), "data/val_data.bin")
    print(f"✅ Done! Train: {n:,} tokens | Val: {len(data) - n:,} tokens")


if __name__ == "__main__":
    with open("config.json") as fd:
        config = json.load(fd, object_hook=lambda d: types.SimpleNamespace(**d))
    prepare_data(config)
