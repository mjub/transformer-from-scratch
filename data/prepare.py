import html
import itertools
import json
import os
import pathlib
import random
import types

import torch
import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import tokenizers

SOURCE_DIR = "nlab-content"

SPLIT_PATTERN = "|".join(
    [
        r"\\[a-zA-Z][a-zA-Z0-9]*",
        r"\\[^a-zA-Z]",
        r"[\(\)\[\]\{\}]",
    ]
)


def normalize(document):
    return html.unescape(document)


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
            document = normalize(f"# {title}\n\n{body}")
            documents.append(f"<|startoftext|>\n{document}\n<|endoftext|>\n")
        except Exception as e:
            print(f"  ⚠️  Skipping {page}: {e}")

    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=config.vocab_size, special_tokens=config.special_tokens
    )

    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Split(SPLIT_PATTERN, behavior="isolated"),
            tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )

    print("🔤 Training BPE tokenizer...")
    tokenizer.train_from_iterator(documents, trainer=trainer, length=len(documents))
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.save("data/tokenizer.json")
    print(f"🎯 Tokenizer trained, vocab size: {tokenizer.get_vocab_size():,}")

    print("⚙️  Encoding corpus...")
    ids = (x.ids for x in tokenizer.encode_batch_fast(documents))
    data = torch.tensor(list(itertools.chain.from_iterable(ids)))
    print(f"📦 Encoded {len(data):,} tokens")

    n = int(len(data) * 0.9)
    torch.save(data[:n].clone(), "data/train_data.bin")
    torch.save(data[n:].clone(), "data/val_data.bin")
    print(f"✅ Done! Train: {n:,} tokens | Val: {len(data) - n:,} tokens")


if __name__ == "__main__":
    with open("config.json") as fd:
        config = json.load(fd, object_hook=lambda d: types.SimpleNamespace(**d))
    prepare_data(config)
