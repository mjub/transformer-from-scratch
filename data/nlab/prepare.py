from pathlib import Path

from tqdm import tqdm


SOURCE_DIR = Path("nlab-content")
OUTPUT_FILE = Path("data/nlab/input.md")
FILE_PATTERN = "*.md"


def build_huge_file():
    files = list(SOURCE_DIR.rglob(FILE_PATTERN))
    print(f"Found {len(files)} files to merge.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for fpath in tqdm(files, desc="Merging"):
            try:
                with open(fpath, "r", encoding="utf-8") as infile:
                    for chunk in iter(lambda: infile.read(4096), ""):
                        outfile.write(chunk)

                    outfile.write("\n<|endoftext|>\n")
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    print(
        f"Build complete. Artifact size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    build_huge_file()
