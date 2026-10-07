# Model Card · Autoencoder y VAE (Fashion-MNIST)

## Modelos y versiones
- Versión 1.0 (octubre de 2026). Cuatro modelos con la **misma arquitectura densa** 784 → 256 → 128 → z → 128 → 256 → 784 (ReLU; salida sigmoide):
  - `ae_z2`, `ae_z16`: autoencoders deterministas (línea base), latente lineal de 2 y 16 dimensiones, pérdida BCE sumada por imagen.
  - `vae_z2`, `vae_z16`: VAE (Kingma y Welling, 2014), codificador con media y log-varianza, truco de reparametrización, pérdida BCE + KL (β = 1).
- Entrenamiento: Adam, lote 256, hasta 15 épocas con *early stopping* sobre la pérdida de validación (paciencia 3), semilla 42 y operaciones deterministas.
- Muestreo: VAE desde N(0, I); AE desde una gaussiana de covarianza completa ajustada a sus códigos de entrenamiento.
- Archivos: `models/<modelo>_encoder.keras`, `models/<modelo>_decoder.keras`; `models/decoder.keras` (VAE z=2) alimenta la app.
- Responsable: [nombre de la estudiante], INF-8239, UASD. Proyecto base: Edwin Ramón José Nolasco.

## Uso previsto y usuarios
Docencia: comparar reconstrucción, muestreo, interpolación, fidelidad, diversidad, memorización y costo de un AE y un VAE pequeños. Usuarios: estudiantes y docentes del curso.

## Usos fuera de alcance
- Generar imágenes para catálogos, publicidad o entrenamiento de otros modelos sin revisión humana.
- Afirmar privacidad o anonimización por el hecho de ser datos sintéticos.
- Detección de anomalías o control de calidad en producción.
- Cualquier imagen distinta de prendas en escala de grises de 28 × 28 sobre fondo uniforme.

## Dataset, licencia y particiones
Fashion-MNIST (Xiao et al., 2017; licencia MIT), descargado desde https://github.com/zalandoresearch/fashion-mnist en el commit `b2617bb6` con verificación SHA-256 (`data/raw_manifest.json`). Validación = 10 % estratificado del entrenamiento oficial (semilla 42); la prueba oficial solo se usa para el reporte. Las etiquetas solo se usan para entrenar el clasificador de características y para analizar resultados por clase.

## Arquitectura y espacio latente
El AE y el VAE comparten codificador y decodificador; solo cambia la cabeza latente y la pérdida. Con z = 2 el espacio latente puede visualizarse (`reports/fig_latent_space.png`). Con z = 16 el VAE usa 10 de 16 dimensiones activas (KL > 0.01): el resto colapsa al prior.

## Métricas de reconstrucción y generación
- **Reconstrucción** (prueba): BCE por imagen, MSE por píxel, SSIM y concordancia de clase según el clasificador.
- **Fidelidad y diversidad:** características de la penúltima capa de una CNN entrenada solo con el conjunto de entrenamiento (`classifier_test_accuracy` en `reports/results.json`). Distancia de Fréchet (fórmula del FID de Heusel et al., 2017, con este extractor propio; no es comparable con FID publicados), precisión y recall k-NN con k = 5 (Kynkäänniemi et al., 2019) y entropía de las clases predichas. La fila *real* (validación frente a prueba) da el valor de referencia alcanzable.

## Diversidad y vecinos cercanos
Para 2 000 muestras se calcula la distancia RMS por píxel a la imagen de entrenamiento más cercana y se compara con la de 2 000 imágenes reales de prueba. Un modelo se considera sospechoso de memorizar si más del 2 % de sus muestras está más cerca del entrenamiento que el percentil 1 de las imágenes reales. Limitación: en espacio de píxeles, las muestras borrosas quedan "cerca" de muchas imágenes sin ser copias.

## Criterio de selección
Menor distancia de Fréchet en validación entre los modelos que no superan el límite de memorización.

## Resultados, costo y hardware
<!-- AUTO:START -->

_Sección generada por `scripts/update_model_card.py` a partir de `reports/`._

**Partición:** entrenamiento 54000 · validación 6000 · prueba 10000 (huella de índices `f21e3d1771bfb062…`). **Modelo seleccionado:** `ae_z16`. Exactitud del clasificador de características en prueba: 0.887.

### Reconstrucción (prueba)

| model | bce | mse | ssim | kl | active_dims |
|---|---|---|---|---|---|
| ae_z2 | 252.745 | 0.027 | 0.515 | nan | nan |
| ae_z16 | 219.811 | 0.012 | 0.694 | nan | nan |
| vae_z2 | 256.243 | 0.028 | 0.498 | 6.295 | 2.000 |
| vae_z16 | 226.870 | 0.015 | 0.645 | 13.395 | 10.000 |

### Generación: fidelidad, diversidad y memoria

| model | fd_valid | fd_test | precision | recall | class_entropy | nn_train_median | share_closer_than_real_p1 | selected |
|---|---|---|---|---|---|---|---|---|
| real (validación vs. prueba) | 0.141 | nan | 0.946 | 0.950 | 0.996 | 0.124 | 0.010 | False |
| ae_z2 | 73.377 | 73.562 | 0.769 | 0.041 | 0.917 | 0.095 | 0.019 | False |
| ae_z16 | 34.945 | 35.051 | 0.696 | 0.543 | 0.942 | 0.154 | 0.001 | True |
| vae_z2 | 97.114 | 97.306 | 0.598 | 0.006 | 0.885 | 0.093 | 0.032 | False |
| vae_z16 | 57.891 | 57.772 | 0.727 | 0.360 | 0.952 | 0.119 | 0.002 | False |

Umbral de copia: distancia RMS 0.0585 (percentil 1 de las distancias de imágenes reales de prueba a su vecino de entrenamiento).

### Costo

**Hardware:** Linux-6.18.44-fc-v77-x86_64-with-glibc2.39 · x86_64 · 1 hilos · GPU: ninguna · TensorFlow 2.21.0 · Python 3.12.13

| model | latent_dim | parameters | decoder_parameters | decoder_kb | epochs_run | train_seconds | seconds_per_epoch |
|---|---|---|---|---|---|---|---|
| ae_z2 | 2 | 469010 | 234896 | 940.27 | 15 | 81.55 | 5.44 |
| ae_z16 | 16 | 472608 | 236688 | 947.28 | 15 | 76.68 | 5.11 |
| vae_z2 | 2 | 469268 | 234896 | 940.29 | 15 | 76.68 | 5.11 |
| vae_z16 | 16 | 474672 | 236688 | 947.29 | 15 | 74.79 | 4.99 |

<!-- AUTO:END -->

## Limitaciones, riesgos y supervisión
- **Convergencia:** las cuatro curvas de validación seguían bajando en la época 15; con más épocas las cifras pueden cambiar.
- **La distancia de Fréchet mezcla fidelidad y diversidad:** el AE z=16 obtiene la menor distancia por su mayor recall, aunque sus muestras presentan artefactos visibles; el VAE z=16 tiene mayor precisión en validación y muestras más limpias. Un criterio basado solo en precisión elegiría el VAE.
- **Muestras borrosas:** los modelos de latente 2 producen imágenes promedio de baja diversidad (recall < 0.05).
- **Memorización en píxeles:** la prueba no detecta copias parciales ni rotadas.
- **Extractor de características propio:** las métricas dependen de una CNN de 3 épocas (≈ 0.89 de exactitud).
- **Supervisión:** revisar visualmente cada lote generado, publicar siempre junto con la prueba de vecinos cercanos y no usar fuera del aula.

## Referencias
- Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. En *Advances in Neural Information Processing Systems 30*.
- Kingma, D. P., & Welling, M. (2014). *Auto-encoding variational Bayes* [Conferencia ICLR 2014]. arXiv. https://arxiv.org/abs/1312.6114
- Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019). Improved precision and recall metric for assessing generative models. En *Advances in Neural Information Processing Systems 32*.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. En *Proceedings of the Conference on Fairness, Accountability, and Transparency* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms* (arXiv:1708.07747). arXiv.
