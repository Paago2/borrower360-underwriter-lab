#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ==========================================
# IMPORT CHECKS
# ==========================================
try:
    from huggingface_hub import snapshot_download
    import kagglehub
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Run: uv pip install huggingface-hub kagglehub")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path.cwd()
RAW_DIR = BASE_DIR / "data" / "00_raw"
EXTRACTED_DIR = BASE_DIR / "data" / "01_extracted"

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# Folder for your manual drag-and-drop files
MANUAL_DROP = RAW_DIR / "manual_drop"
MANUAL_DROP.mkdir(parents=True, exist_ok=True)

# Kaggle datasets (require ~/.kaggle/kaggle.json)
KAGGLE_DATASETS = {
    # "owner/dataset": "local_folder_name"
    "urbikn/sroie-datasetv2": "sroie_receipts",
    # Add more here as needed
    # "chaitalithakkar/synthetic-kyc-and-transaction-risk-dataset": "synthetic_kyc",
}

# Hugging Face dataset repos (download actual files into your folder)
HF_DATASETS = {
    # "repo_id": "local_folder_name"
    "RIPS-Goog-23/RVL-CDIP": "rvl_cdip_documents",
    "naver-clova-ix/synthdog-en": "synthdog_en",
    # Add more here as needed
    # "ArneBinder/xfund": "xfund_forms",
}

# ==========================================
# HELPERS
# ==========================================
def copy_anything(src_path: str | Path, dest_dir: str | Path) -> None:
    """
    Copies either a directory OR a single file into dest_dir.
    This is safe because kagglehub may return either a folder or a zip file.
    """
    src = Path(src_path)
    dest = Path(dest_dir)

    dest.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"⚠️  Source not found: {src}")
        return

    if src.is_dir():
        print(f"   ↳ Copying folder contents -> {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        print(f"   ↳ Copying file -> {dest / src.name}")
        shutil.copy2(src, dest / src.name)


def is_wsl() -> bool:
    try:
        return "wsl" in os.uname().release.lower()
    except Exception:
        return False


def print_wsl_path(path_obj: Path) -> None:
    """
    If running on WSL, prints the path in Windows-friendly formats.
    Avoids unicode escape issues and f-string backslash issues.
    """
    if not is_wsl():
        return

    try:
        s_path = str(path_obj.resolve())

        # /mnt/c/... -> C:\...
        if s_path.startswith("/mnt/c/"):
            win_path = "C:\\" + s_path[len("/mnt/c/"):].replace("/", "\\")
            print(f"🪟 Windows Path: {win_path}")
            return

        # /home/... -> \\wsl.localhost\Ubuntu\home\...
        if s_path.startswith("/home/"):
            tail = s_path.replace("/", "\\")
            network_path = r"\\wsl.localhost\Ubuntu" + tail
            print(f"🪟 Network Path: {network_path}")
            return
    except Exception:
        return


# ==========================================
# DOWNLOADERS
# ==========================================
def download_kaggle() -> None:
    print("\n⬇️  Kaggle downloads...")
    for ds_id, local_name in KAGGLE_DATASETS.items():
        out_dir = RAW_DIR / local_name

        # Clean target folder each run for consistency
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  - Fetching: {ds_id}")
        try:
            cache_path = kagglehub.dataset_download(ds_id)
            print(f"   ↳ Kaggle cache path: {cache_path}")
            copy_anything(cache_path, out_dir)
            print(f"    ✅ Ready in: {out_dir}")
        except Exception as e:
            print(f"    ❌ Failed to download {ds_id}: {e}")
            print("       Tip: verify ~/.kaggle/kaggle.json exists and chmod 600 it.")




# ==========================================
# MAIN
# ==========================================
def main() -> None:
    print("=" * 44)
    print("   BORROWER360 DATA DOWNLOADER (LOCAL)")
    print("=" * 44)

    print(f"📁 Project root: {BASE_DIR}")
    print(f"📁 Raw data dir: {RAW_DIR}")
    print(f"📁 Extracted dir: {EXTRACTED_DIR}")

    # Automated downloads
    download_kaggle()
   # download_hf()

    # Manual step reminder
    print("\n" + "=" * 44)
    print("👋 MANUAL STEP (for files you already downloaded)")
    print("=" * 44)
    print("Move/copy anything you already downloaded into:")
    print(f"  ➜ {MANUAL_DROP}")
    print_wsl_path(MANUAL_DROP)

    print("\n✅ Script complete.")


if __name__ == "__main__":
    main()
