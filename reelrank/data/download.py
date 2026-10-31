"""Download KuaiRand-Pure and video caption/category supplements."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path

PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
PURE_MD5 = "0820331067a3784d9691136f772b35a7"
CAPTIONS_URL = "https://zenodo.org/records/18159199/files/kuairand_video_captions.csv"
CATEGORIES_URL = "https://zenodo.org/records/18159199/files/kuairand_video_categories.csv"

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_md5: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_md5 is None or _md5(dest) == expected_md5:
            print(f"skip (exists): {dest.name}")
            return
        print(f"re-download (bad checksum): {dest.name}")
        dest.unlink()
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    if expected_md5 is not None:
        got = _md5(dest)
        if got != expected_md5:
            raise RuntimeError(f"md5 mismatch for {dest.name}: {got} != {expected_md5}")


def download_all(raw_dir: Path = RAW) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "KuaiRand-Pure.tar.gz"
    _download(PURE_URL, archive, PURE_MD5)

    extract_root = raw_dir / "KuaiRand-Pure"
    if not (extract_root / "data").exists():
        print(f"extracting {archive.name}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(raw_dir)
        # Normalize nested layout if needed
        if not extract_root.exists():
            # tar may extract as KuaiRand-Pure/ already
            candidates = [p for p in raw_dir.iterdir() if p.is_dir() and "KuaiRand" in p.name]
            if candidates:
                shutil.move(str(candidates[0]), str(extract_root))

    _download(CAPTIONS_URL, raw_dir / "kuairand_video_captions.csv")
    _download(CATEGORIES_URL, raw_dir / "kuairand_video_categories.csv")
    print(f"ready: {extract_root}")
    return extract_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KuaiRand-Pure + captions")
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    download_all(args.raw_dir)


if __name__ == "__main__":
    main()
