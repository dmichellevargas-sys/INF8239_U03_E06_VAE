"""Descarga reproducible de Fashion-MNIST (commit fijado) con verificación SHA-256.

Uso:  uv run python scripts/download_data.py
"""
from __future__ import annotations

import json
import sys

import requests

from inf8239_u03_gen.config import DATASET_COMMIT, RAW_BASE_URL, RAW_DIR, RAW_FILES, ROOT
from inf8239_u03_gen.data import sha256

MANIFEST = ROOT / "data/raw_manifest.json"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, expected in RAW_FILES.items():
        path = RAW_DIR / name
        url = f"{RAW_BASE_URL}/{name}"
        if not path.exists():
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            path.write_bytes(response.content)
        digest = sha256(path)
        if digest != expected:
            path.unlink()
            print(f"ERROR: SHA-256 de {name} no coincide; archivo eliminado.", file=sys.stderr)
            return 1
        entries.append({"file": f"data/raw/{name}", "url": url, "sha256": digest})
        print(f"OK  {name}  {digest[:16]}…")
    MANIFEST.write_text(json.dumps({
        "dataset": "Fashion-MNIST", "license": "MIT", "commit": DATASET_COMMIT, "files": entries,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
