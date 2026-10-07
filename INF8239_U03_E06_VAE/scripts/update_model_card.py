"""Rellena las tablas de MODEL_CARD.md con los resultados de reports/ (última ejecución)."""
from __future__ import annotations

import json
import re

import pandas as pd

from inf8239_u03_gen.config import REPORTS, ROOT

CARD = ROOT / "MODEL_CARD.md"
START, END = "<!-- AUTO:START -->", "<!-- AUTO:END -->"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fmt(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    return df.apply(lambda c: c.map(lambda v: f"{v:.{digits}f}" if isinstance(v, float) else v))


def build() -> str:
    res = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    rec = pd.read_csv(REPORTS / "reconstruction_metrics.csv")
    gen = pd.read_csv(REPORTS / "generation_metrics.csv")
    cost = pd.read_csv(REPORTS / "cost.csv")
    env, split = res["environment"], res["split"]
    gen_cols = ["model", "fd_valid", "fd_test", "precision", "recall", "class_entropy",
                "nn_train_median", "share_closer_than_real_p1", "selected"]
    rec_cols = [c for c in ["model", "bce", "mse", "ssim", "kl", "active_dims"] if c in rec.columns]
    hw = (f"{env['platform']} · {env['processor']} · {env['cpu_count']} hilos · GPU: {env['gpu'] or 'ninguna'} · "
          f"TensorFlow {env['tensorflow']} · Python {env['python']}")
    return "\n\n".join([
        START,
        "_Sección generada por `scripts/update_model_card.py` a partir de `reports/`._",
        (f"**Partición:** entrenamiento {split['train']} · validación {split['valid']} · prueba {split['test']} "
        f"(huella de índices `{split['train_index_sha256'][:16]}…`). **Modelo seleccionado:** "
        f"`{res['selected_model']}`. Exactitud del clasificador de características en prueba: "
        f"{res['classifier_test_accuracy']:.3f}."),
        "### Reconstrucción (prueba)", md(fmt(rec[rec_cols])),
        "### Generación: fidelidad, diversidad y memoria", md(fmt(gen[[c for c in gen_cols if c in gen.columns]])),
        (f"Umbral de copia: distancia RMS {res['copy_threshold_rms']:.4f} (percentil 1 de las distancias de imágenes "
        f"reales de prueba a su vecino de entrenamiento)."),
        "### Costo", f"**Hardware:** {hw}", md(fmt(cost, 2)),
        END,
    ])


def main() -> None:
    text = CARD.read_text(encoding="utf-8")
    auto = build()
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: auto, text, flags=re.DOTALL) \
        if START in text else text + "\n" + auto + "\n"
    CARD.write_text(text, encoding="utf-8")
    print("MODEL_CARD.md actualizado")


if __name__ == "__main__":
    main()
