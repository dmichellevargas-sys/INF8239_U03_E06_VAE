"""Flujo completo: descarga -> entrenamiento y evaluación -> Model Card.

Uso:  uv run python scripts/run_all.py [--epochs 15]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import download_data, train_ae_vae, update_model_card


def main() -> int:
    if download_data.main():
        return 1
    train_ae_vae.main()
    update_model_card.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
