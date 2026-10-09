"""
fetching_test.py
----------------
Downloads the 15 LoL test images (low-light + ground-truth) from HuggingFace
and saves them into:
    test_images/low/   ← dark input images
    test_images/high/  ← reference (ground-truth) images

Run once before starting app.py:
    python fetching_test.py
"""

import os
import zipfile
from pathlib import Path
import shutil

TEST_DIR = Path("test_images")
LOW_DIR  = TEST_DIR / "low"
HIGH_DIR = TEST_DIR / "high"


def fetch_test_images():
    if LOW_DIR.exists() and len(list(LOW_DIR.glob("*.*"))) >= 10:
        print(f"[OK] Test images already present in '{TEST_DIR}/'")
        return

    print("[>>] Downloading LoL Dataset from HuggingFace ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "Run:  pip install huggingface_hub"
        )

    zip_path = hf_hub_download(
        repo_id="geekyrakshit/LoL-Dataset",
        filename="lol_dataset.zip",
        repo_type="dataset",
    )
    print(f"[>>] Extracting from {zip_path} ...")

    extract_root = Path("_lol_tmp")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # Locate the test split (low / high dirs anywhere under the extracted tree)
    import glob
    low_dirs  = [Path(p) for p in glob.glob(str(extract_root / "**" / "low"),  recursive=True)]
    high_dirs = [Path(p) for p in glob.glob(str(extract_root / "**" / "high"), recursive=True)]

    # Prefer dirs that live inside an "eval" / "test" / "val" parent
    # Use substring match so "eval15" correctly matches "eval"
    def pick(dirs):
        test_dirs = [
            d for d in dirs
            if any(kw in part.lower() for part in d.parts for kw in ("eval", "test", "val"))
        ]
        return test_dirs[0] if test_dirs else (dirs[0] if dirs else None)

    src_low  = pick(low_dirs)
    src_high = pick(high_dirs)

    if src_low is None or src_high is None:
        raise RuntimeError(
            "Could not locate low/high directories in the downloaded dataset. "
            f"Extracted to: {extract_root}"
        )

    print(f"[>>] Using low  folder: {src_low}")
    print(f"[>>] Using high folder: {src_high}")

    LOW_DIR.mkdir(parents=True, exist_ok=True)
    HIGH_DIR.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    MAX_IMAGES = 20   # safety cap — eval15 has exactly 15

    copied = 0
    for f in sorted(src_low.iterdir()):
        if f.suffix.lower() in image_exts and copied < MAX_IMAGES:
            shutil.copy2(f, LOW_DIR / f.name)
            copied += 1

    for f in sorted(src_high.iterdir()):
        if f.suffix.lower() in image_exts:
            high_dest = HIGH_DIR / f.name
            # Only copy if matching low image exists
            if (LOW_DIR / f.name).exists():
                shutil.copy2(f, high_dest)

    print(f"[OK] Saved {copied} test image(s) to '{TEST_DIR}/'")

    # Clean up temporary extraction
    shutil.rmtree(extract_root, ignore_errors=True)


if __name__ == "__main__":
    fetch_test_images()
    print("\nTest images are ready. Run:  streamlit run app.py")