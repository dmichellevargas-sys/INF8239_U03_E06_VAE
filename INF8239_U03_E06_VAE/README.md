# INF-8239 · Ejercicio 06 · VAE, evaluación y Model Card

Autoencoder (línea base) y VAE sobre **Fashion-MNIST**, con la misma arquitectura y latentes de 2 y 16
dimensiones: reconstrucción, muestreo, interpolación, fidelidad, diversidad, memorización, costo y Model Card.
Proyecto individual de la Unidad 03 (LAB10). Proyecto base del curso: Edwin Ramón José Nolasco.

> Este ejercicio no usa MovieLens. Fashion-MNIST (licencia MIT) se descarga desde su repositorio oficial
> y no se versiona en Git.

## Requisitos
[uv](https://docs.astral.sh/uv/) ≥ 0.5 (instala Python 3.12) e internet la primera vez (≈ 30 MB).
En una CPU común, la ejecución completa tarda entre 5 y 10 minutos; con GPU (WSL2/Linux), bastante menos.

## Ejecución (uv, recomendada)
```bash
uv python install 3.12
uv sync --extra cpu                    # GPU en WSL2/Linux: uv sync --extra gpu
uv run python scripts/check_runtime.py
uv run pytest -q                       # 18 pruebas
uv run python scripts/run_all.py       # descarga -> 4 modelos + evaluación -> Model Card
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/E06_vae_model_card.ipynb
uv run streamlit run app/streamlit_app.py   # explorador del espacio latente (VAE z=2)
```
Prueba rápida: `uv run python scripts/train_ae_vae.py --epochs 2 --limit 3000`.

## Ejecución alternativa (pip)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
$env:PYTHONPATH="src;."             # macOS/Linux: export PYTHONPATH=src:.
python -m pytest -q
python scripts/run_all.py
```

## Estructura
```
MODEL_CARD.md               ficha de los modelos (tablas regeneradas automáticamente)
app/streamlit_app.py        explorador del espacio latente
data/raw_manifest.json      URLs, commit y SHA-256 de Fashion-MNIST
docs/AI_USE_DECLARATION.md  declaración de uso de IA
notebooks/                  notebook ejecutado
reports/                    métricas (CSV/JSON) y figuras
scripts/                    descarga, entrenamiento y evaluación, Model Card, run_all
src/inf8239_u03_gen/        config, data, models, vae, metrics
tests/                      pruebas de datos, modelos, pérdidas y métricas
```

## Decisiones
| Decisión | Razón |
|---|---|
| AE y VAE con la misma arquitectura (el AE base tenía una sola capa con ReLU en el latente) | Comparación justa: la única diferencia es la parte variacional |
| Latentes 2 y 16 para ambos | El proyecto base comparaba un AE z=16 con un VAE z=2 |
| Validación estratificada y `test_step` propio en el VAE | *Early stopping* real del VAE; la base lo entrenaba sin validación |
| AE muestreado desde una gaussiana ajustada a sus códigos | Línea base generativa para el AE |
| Fréchet, precisión/recall k-NN y entropía sobre una CNN propia | Separar fidelidad de diversidad |
| Vecino más cercano en entrenamiento comparado con imágenes reales de prueba | Detectar memorización con un umbral calibrado |
| Selección por Fréchet en validación con filtro de memorización | Regla declarada antes de mirar la prueba |

## Resultados en prueba
| Modelo | BCE ↓ | SSIM ↑ | Fréchet ↓ | Precisión ↑ | Recall ↑ | Muestras "copia" |
|---|---|---|---|---|---|---|
| Real (validación vs. prueba) | — | — | 0.14 | 0.95 | 0.95 | 1 % (por definición) |
| AE z=2 | 252.7 | 0.515 | 73.6 | 0.769 | 0.041 | 2.0 % |
| **AE z=16 (seleccionado)** | **219.8** | **0.694** | **35.1** | 0.696 | **0.543** | 0.05 % |
| VAE z=2 | 256.2 | 0.498 | 97.3 | 0.598 | 0.006 | 3.2 % |
| VAE z=16 | 226.9 | 0.645 | 57.8 | **0.727** | 0.360 | 0.15 % |

Costo: ≈ 470 000 parámetros por modelo, unos 75–80 s de entrenamiento en una CPU de 1 hilo, decodificador de ≈ 0.95 MB.
Detalle y limitaciones en `MODEL_CARD.md`, `reports/` y el notebook.

## Git
```bash
git init
git add pyproject.toml uv.lock requirements.txt .gitignore .python-version
git commit -m "chore: configure reproducible generative environment"
git add src/inf8239_u03_gen/config.py src/inf8239_u03_gen/data.py scripts/download_data.py data/raw_manifest.json tests/test_data.py
git commit -m "feat: pinned Fashion-MNIST download and stratified split"
git add src scripts tests
git commit -m "feat: AE and VAE with shared architecture, generative metrics and memorization test"
git add reports notebooks app
git commit -m "docs: executed notebook, reports and latent explorer"
git add MODEL_CARD.md README.md docs/AI_USE_DECLARATION.md
git commit -m "docs: model card, README and AI use statement"
git tag u03-ejercicio06
```

## Referencias
Ver `MODEL_CARD.md`.
