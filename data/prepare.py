import html
import random
import pathlib

import tokenizers
import torch
import tqdm


SOURCE_DIR = "nlab-content"

TRAIN_DATA = "data/train_data.bin"
VAL_DATA = "data/val_data.bin"

def prepare_data():
    pages = pathlib.Path(SOURCE_DIR).rglob("*.md")
    print(f"Found {len(pages):,} pages to merge.")
    random.shuffle(pages)
