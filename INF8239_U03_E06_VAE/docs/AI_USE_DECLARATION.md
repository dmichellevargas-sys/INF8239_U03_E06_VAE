# Declaración de uso de inteligencia artificial

## Herramientas utilizadas
- **Claude (Anthropic)**, interfaz web claude.ai, dentro de un proyecto de la asignatura. Octubre de 2026.

## Propósito de cada uso
- Revisar el proyecto base de LAB10 (AE y VAE) frente a la rúbrica.
- Generar el código: descarga verificada, partición, arquitecturas comparables, VAE con validación, métricas de reconstrucción, fidelidad, diversidad y memorización, figuras, pruebas y Model Card automática.
- Redactar README, Model Card, notebook e informe.

## Prompts o decisiones relevantes
1. Se compartieron los archivos del proyecto base de modelos generativos y, después del Ejercicio 05, el enunciado y la rúbrica del Ejercicio 06: *"al finalizar seguimos con este Ejercicio 06 · VAE, evaluación y Model Card…"*.
2. Decisiones revisadas por la estudiante: misma arquitectura para AE y VAE; latentes 2 y 16 para ambos; AE muestreado con una gaussiana ajustada; métricas sobre una CNN entrenada solo con entrenamiento; umbral de memorización calibrado con imágenes reales de prueba; selección por distancia de Fréchet en validación.

## Elementos verificados por el equipo
> Marque solo lo que haya comprobado personalmente.

- [ ] Ejecuté `uv run pytest -q` y `uv run python scripts/run_all.py`; la Model Card quedó actualizada con mi hardware.
- [ ] Revisé visualmente las figuras de muestras, interpolación y memorización y estoy de acuerdo con su interpretación.
- [ ] Comprobé que la prueba solo se usa para el reporte final.
- [ ] Verifiqué las referencias (Kingma y Welling, Heusel et al., Kynkäänniemi et al., Xiao et al., Mitchell et al.).
- [ ] Puedo explicar cada archivo de `src/` y `scripts/`.

## Cambios realizados sobre resultados generados
| Problema detectado | Corrección |
|---|---|
| El VAE base no tenía `test_step`, por lo que no podía validarse ni usar *early stopping*. | Se añadió `test_step` y validación estratificada. |
| El método `_losses` del VAE chocaba con un atributo interno de Keras (`TrackedList`). | Se renombró a `_compute_losses`. |
| `scipy.linalg.sqrtm` ya no acepta el argumento `disp`. | Se adaptó la llamada y se estabilizaron las covarianzas con 1e-6·I. |
| El cálculo inicial de precisión/recall podía requerir más de 1 GB de memoria. | Se calculó por bloques con productos matriciales. |
| La regla de selección eligió el AE z=16, aunque visualmente sus muestras tienen más artefactos que las del VAE z=16. | No se cambió la regla a posteriori; la discrepancia se documenta como limitación en la Model Card y en el informe. |

## Responsabilidad asumida
La estudiante responde por el código, los resultados, las referencias y las conclusiones, y declara poder explicar todo lo entregado.
