import html
import random
import re
from pathlib import Path

from tqdm import tqdm

SOURCE_DIR = Path("nlab-content")
OUTPUT_FILE = Path("data/input.md")

FILE_PATTERN = "*.md"


SPECIAL = re.compile(r"\[\[![^\]]+\]\]")
INTERNAL_LINKS = re.compile(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]")
EXTERNAL_LINKS = re.compile(r"\[([^\]]+)\]\(https?://\S+\)")


def build_huge_file():
    articles = list(SOURCE_DIR.rglob(FILE_PATTERN))

    # Shuffle the articles
    random.seed(303)
    random.shuffle(articles)

    print(f"Found {len(articles)} articles to merge.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for fpath in tqdm(articles, desc="Merging"):
            try:
                # Read the article title
                with open(fpath.parent / "name", "r", encoding="utf-8") as infile:
                    text = f"# {infile.read()}\n\n"
                # Read the article body
                with open(fpath, "r", encoding="utf-8") as infile:
                    text = text + infile.read()
                # We perform some light data cleaning
                # TODO Remove more markup
                text = html.unescape(text)
                text = SPECIAL.sub(r"", text)
                text = INTERNAL_LINKS.sub(r"\1", text)
                text = EXTERNAL_LINKS.sub(r"\1", text)
                outfile.write(f"<|startoftext|>\n{text}\n<|endoftext|>\n")
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    print(
        f"Build complete. Artifact size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    build_huge_file()
