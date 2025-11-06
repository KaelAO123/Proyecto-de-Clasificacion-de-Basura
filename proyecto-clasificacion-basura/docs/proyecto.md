# ECO AI — Clasificación de Basura con IA

1. Introducción

La gestión ineficiente de residuos y la falta de separación correcta impactan en el reciclaje y el medio ambiente. Este proyecto propone un clasificador de basura que identifica en tiempo real seis categorías de residuos: cardboard, glass, metal, paper, plastic y trash.

- Qué voy a hacer (Python): construir un pipeline completo en Python para entrenar y usar un modelo de visión por computadora, con una app de cámara para inferencia en tiempo real.
- Cómo (Redes Neuronales): usar transferencia de aprendizaje (MobileNetV2 por defecto) con Keras/TensorFlow, aumentación de datos, entrenamiento en dos etapas y evaluación con métricas estándar.
- ¿Por qué?: facilitar la separación correcta de residuos, mejorar tasas de reciclaje y educar con recomendaciones visuales y contenedores sugeridos.
- ¿Para qué?: asistencia educativa, prototipos de punto limpio, campañas en campus/municipios, y base técnica para integrar sistemas de contenedores inteligentes.

2. Planteamiento del problema

La separación de residuos suele ser manual, inconsistente y sin retroalimentación inmediata. No existen mecanismos sencillos de observación y guía en tiempo real para el ciudadano promedio en el punto de descarte. Esto reduce la calidad del material reciclable y aumenta costos de clasificación posterior.

3. Objetivos

- General: Clasificar residuos en tiempo real a partir de imágenes, mostrando clase, confianza y recomendaciones de reciclaje.
- Específicos:
  - Preparar y dividir un dataset balanceado (train/val/test) con aumentación.
  - Entrenar un modelo CNN con transferencia de aprendizaje y fine-tuning.
  - Lograr métricas competitivas (p. ej., >80% accuracy en test como punto de partida).
  - Generar reportes reproducibles de evaluación y análisis de errores.
  - Proveer una interfaz de cámara fácil de usar (PC y guía para iPhone/Safari).

4. Justificación

4.1. Justificación Social

Una mejor clasificación en origen mejora las tasas de reciclaje, reduce desechos enviados a rellenos y promueve educación ambiental. La interfaz propone contenedor y consejos, ayudando a cambiar hábitos y decisiones.

4.2. Justificación Técnica

- Software: Python 3.10+, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn, Albumentations, Matplotlib/Seaborn.
- Hardware: CPU (mínimo) y preferible GPU para entrenamiento; cámara web para inferencia; almacenamiento para dataset y modelos.
- RRHH: 1–2 desarrolladores/estudiantes con conocimientos en ML/Visión por Computadora y manejo de datos.
- Infraestructura opcional: hosting para publicar reportes o una demo web ligera.

4.3. Justificación Económica (referencial)

- Hosting: 3600 Bs./año (opcional, para demos o reportes).
- Servicio de Internet: 1800 Bs./año (si no se dispone de infraestructura institucional).
- Entrenamiento con GPU en la nube (opcional): desde 2000 Bs./mes según horas y proveedor.

5. Diseño del Sistema Inteligente

Flujo principal:

1) Datos: `data/raw/` → limpieza y división → `data/processed/{train,val,test}`.
2) Generadores: normalización 1/255; augmentación en train (rotación, shifts, flips, zoom, brillo, etc.).
3) Modelo: backbone pre-entrenado (MobileNetV2) + cabeza densa personalizada; clasificación softmax.
4) Entrenamiento: dos etapas (cabeza → fine-tuning del backbone) con callbacks (EarlyStopping, Checkpoint, ReduceLROnPlateau, TensorBoard, CSVLogger).
5) Evaluación: métricas (accuracy, precision, recall, AUC), matriz de confusión, reporte por clase, análisis de errores.
6) Inferencia: cámara del PC con overlay de clase/confianza y recomendaciones; opción de guía por iPhone/Safari.

Diseño de la red neuronal:

- Backbone: MobileNetV2 (o EfficientNetB0, ResNet50 opcionales) con `include_top=False`.
- Cabeza: GlobalAveragePooling2D → Dense(512, ReLU) + BN + Dropout → Dense(256, ReLU) + BN + Dropout → Dense(128, ReLU) + Dropout → Dense(num_classes, Softmax).
- Pérdida: categorical_crossentropy. Optimizador: Adam. Métricas: accuracy, precision, recall, AUC.

5.1 Arquitectura detallada (capas, neuronas y tamaños)

- Entrada: imágenes RGB reescaladas a `224×224×3` y normalizadas a $[0,1]$.
- Extracción de características (congelada al inicio): `MobileNetV2` preentrenada en ImageNet, `include_top=False`.
- Cabeza de clasificación (según `src/model_builder.py`):
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(512, activation=ReLU) → BatchNorm → Dropout(0.4)
  - Dense(256, activation=ReLU) → BatchNorm → Dropout(0.3)
  - Dense(128, activation=ReLU) → Dropout(0.2)
  - Dense(6, activation=Softmax)

Número de neuronas en capas densas: 512 → 256 → 128 → 6 (salida). El conteo exacto de parámetros se imprime con `model.summary()` y se guarda en `training_info.json` (campo `total_parameters`).

5.2 Salida y función de pérdida (Softmax + Cross-Entropy)

- Softmax por clase $k$ con logits $z_k$:

$$
\mathrm{softmax}(z)_k = p_k = \frac{e^{z_k}}{\sum_{j=1}^{C} e^{z_j}}
$$

- Pérdida de entropía cruzada categórica (one-hot $y$):

$$
\mathcal{L} = -\sum_{k=1}^{C} y_k\, \log(p_k)
$$

5.3 Optimización (Adam), tasas de aprendizaje y fine-tuning

- Etapa 1 (cabeza densa, backbone congelado): Adam con $\alpha=0.001$.
- Etapa 2 (fine-tuning, backbone liberado): Adam con $\alpha=10^{-6}$ (proveniente de `fine_tune_learning_rate=1e-5` dividido por 10 en el código).
- Actualización Adam (con corrección de sesgo):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\\
	heta_{t+1} &= \theta_t - \alpha\, \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
\end{aligned}
$$

- Ajuste dinámico de LR: `ReduceLROnPlateau` reduce por factor 0.2 si `val_loss` no mejora por 5 épocas.
- `EarlyStopping` (paciencia=10, restaura mejores pesos por `val_loss`).
- `ModelCheckpoint` guarda `best_model.h5` según `val_accuracy` máxima.

5.4 Métricas (definiciones formales)

- Exactitud: $\displaystyle \mathrm{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$.
- Precisión: $\displaystyle \mathrm{Precision} = \frac{TP}{TP+FP}$.
- Recall: $\displaystyle \mathrm{Recall} = \frac{TP}{TP+FN}$.
- F1-Score: $\displaystyle F1 = 2\cdot \frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$.
- AUC: Área bajo la curva ROC; resume el trade-off TPR-FPR en todos los umbrales.

5.5 Pesos de clase (desbalance)

Se calculan con `sklearn.utils.compute_class_weight` sobre etiquetas del generador de train. Intuitivamente:

$$
w_c \propto \frac{N}{C\cdot N_c}
$$

donde $N$ es el número total de ejemplos, $C$ el número de clases y $N_c$ los ejemplos de la clase $c$. Estos pesos se pasan a `model.fit(..., class_weight=...)`.

5.6 Aumentación y normalización de datos

- Normalización: reescalado $x \leftarrow x/255$ en todos los splits.
- Aumentación (train): rotación, desplazamientos, flips, zoom, brillo/contraste, `channel_shift`, `HueSaturationValue`, `GaussianBlur`, `GaussNoise`, `CoarseDropout`, `RandomGamma` (ver `src/data_generators.py` y `src/data_augmentation.py`).
- Splits: 70% train, 15% val, 15% test (con `splitfolders` y fallback manual).

5.7 Callbacks y artefactos

- `TensorBoard`: logs en `models/trained_models/<run>/logs`.
- `CSVLogger`: `training_log.csv` por época.
- Artefactos: `best_model.h5`, `final_model.h5`, `*_history.json`, `*_metadata.json`, `training_info.json`, `training_history.png`.

6. Código de Python

Estructura de módulos (ver `docs/` para detalle):

- Preprocesado: `src/data_preprocessing.py`
- Generadores y pesos de clase: `src/data_generators.py`
- Aumentación avanzada/balanceo: `src/data_augmentation.py`
- Construcción/compilación del modelo: `src/model_builder.py`
- Entrenamiento: `src/model_training.py`
- Evaluación y reportes: `src/model_evaluation.py`
- Inferencia cámara local: `src/camera_classifier.py`
- Guía iPhone/Safari + cámara PC: `src/iphone.py`
- Utilidades de modelos: `src/model_utils.py`

Comandos útiles (PowerShell):

```powershell
# 1) Preprocesar y dividir datos
python proyecto-clasificacion-basura\src\data_preprocessing.py

# 2) Entrenar
python proyecto-clasificacion-basura\src\model_training.py

# 3) Evaluar (con modelo ya entrenado)
python proyecto-clasificacion-basura\src\model_evaluation.py

# 4) Inferencia con cámara del equipo
python proyecto-clasificacion-basura\src\camera_classifier.py

# 5) Guía iPhone (Safari) + cámara del PC
python proyecto-clasificacion-basura\src\iphone.py
```

6.1 ¿Cómo se generaron las gráficas?

- Curvas de entrenamiento (`training_history.png`): generadas en `ModelTrainer.plot_training_history(history)` con Matplotlib a partir del historial combinado de las dos etapas (keys: `accuracy`, `val_accuracy`, `loss`, `val_loss`, `precision`, `val_precision`, `recall`, `val_recall`).
- Matriz de confusión (`confusion_matrix.png`): `ModelEvaluator.plot_confusion_matrix(...)` usando `seaborn.heatmap` con etiquetas de clase.
- Distribución y accuracy por clase (`class_distribution.png`): `ModelEvaluator.plot_class_distribution(...)` comparando conteos reales vs. predichos y cálculo de $\mathrm{Accuracy}_c$ por clase.
- Misclasificaciones (`misclassifications.png`): `ModelEvaluator.analyze_misclassifications(...)` toma ejemplos erróneos y los visualiza con títulos Real/Pred.
- Ejemplos de aumentación (`reports/figures/data_augmentation_examples.png`): `DataGenerator.visualize_augmentations(...)` muestra un batch transformado del generador de train.

6.2 Parámetros y tamaños

- Tamaño de entrada: 224×224×3.
- Número de clases: 6 (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
- Neuronas densas: 512 → 256 → 128 → 6.
- Total de parámetros: se reporta automáticamente en consola (`model.summary()`) y se guarda en `training_info.json` → `total_parameters`.

7. Conclusiones y Recomendaciones

Este proyecto demuestra la viabilidad de un clasificador de residuos en tiempo real basado en transferencia de aprendizaje. La arquitectura y el pipeline permiten reproducibilidad (split, augmentación, callbacks, reportes) y facilitan mejoras incrementales.

Recomendaciones:

- Ampliar y balancear el dataset con ejemplos en condiciones reales (iluminación/fondos diversos).
- Explorar backbones alternativos (EfficientNet, ConvNeXt) y técnicas de fine-tuning progresivo.
- Implementar ensembles/tta para aumentar robustez, y cuantización/pruning para despliegues embebidos.
- Integrar un panel web para reportes y telemetría de uso.
- Considerar sensores adicionales (peso, proximidad) para contenedores inteligentes.

8. Detalles adicionales de diseño y fundamentos

8.1 MobileNetV2 (resumen)

MobileNetV2 usa bloques de "inverted residuals" con convoluciones separables en profundidad (depthwise separable), reduciendo parámetros y cómputo. La idea es factorizar una conv 3×3 estándar en una conv depthwise 3×3 seguida de una pointwise 1×1. El costo pasa de $K\cdot K\cdot C_{in}\cdot C_{out}$ a $K\cdot K\cdot C_{in} + C_{in}\cdot C_{out}$.

8.2 Batch Normalization y Dropout

Para una activación $x$ en el batch, BN normaliza y reescala:

$$
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat x + \beta
$$

Dropout desactiva aleatoriamente unidades en entrenamiento con probabilidad $p$ y en inferencia se usa el valor esperado (o se hace "inverted dropout" escalando en train). En este proyecto se usa como regularización en la cabeza densa: 0.3 → 0.4 → 0.3 → 0.2.

8.3 Estrategia de validación, semillas y splits

- Split por `splitfolders.ratio(..., seed=42)` con proporciones 70/15/15; si falla, fallback manual con `np.random.seed(42)`.
- Sin data leakage: test no participa del entrenamiento ni de la selección de hiperparámetros.
- Para comparabilidad: mantener `image_size`, `batch_size`, `learning_rate` y la misma semilla.

8.4 Gestión de artefactos y versiones

- Cada corrida crea `models/trained_models/YYYYMMDD_HHMMSS/`.
- Artefactos: `best_model.h5`, `final_model.h5`, `training_log.csv`, `training_history.png`, `*_history.json`, `*_metadata.json`, `training_info.json`, y `logs/` para TensorBoard.
- Reportes de evaluación en `reports/evaluation/`.

8.5 Consideraciones de rendimiento (FPS y latencia)

- Inferencia en `camera_classifier.py` procesa cada 5 frames para mejorar fluidez y calcula FPS por segundo (impreso por consola). El FPS dependerá de la CPU/GPU, resolución de cámara y tamaño del modelo.
- Para baja latencia: usar `image_size` moderado (224) y evitar escalado innecesario.

9. Limitaciones y riesgos

- Desbalance de clases y sesgo del dataset: se mitiga con aumentación y `class_weight`, pero puede persistir.
- Variaciones de iluminación, fondo y oclusiones degradan el desempeño: cubrir con datos reales y augmentación fotométrica/geométrica.
- Dominio distinto (e.g., residuos locales no presentes en el dataset) reduce generalización; requiere re-entrenamiento o fine-tuning adicional.
- Métricas globales pueden ocultar clases difíciles: revisar `classification_report.md` y `class_distribution.png` por clase.

10. Trabajo futuro

- Fine-tuning progresivo por etapas descongelando bloques del backbone.
- Backbones alternativos (EfficientNet, ConvNeXt) y búsqueda de hiperparámetros.
- Distillation para modelos compactos; cuantización post-training para edge.
- Data-centric AI: curación/etiquetado activo, detección de outliers y corrección de etiquetas.
- Integración IoT (microcontroladores/cámaras embebidas) y un backend para métricas en campo.

11. Reproducibilidad y entorno

- Entorno Python: ver `requirements.txt` (TensorFlow, OpenCV, Albumentations, etc.).
- Plataforma: Windows PowerShell (comandos incluidos). Compatible con Linux/Mac adaptando rutas.
- Versionado de datos/modelos: mantener estructura `data/` y `models/` indicadas.
- Semillas: 42 para splits; configurar también `TF_DETERMINISTIC_OPS` si se busca determinismo estricto (opcional).

12. Anexos

12.1 Hiperparámetros (de `config/parameters.yaml`)

- data:
  - image_size: 224×224
  - batch_size: 32
  - classes: [cardboard, glass, metal, paper, plastic, trash]
  - splits: val=0.15, test=0.15 (train≈0.70)
- training:
  - base_model: MobileNetV2
  - initial_epochs: 15 (cabeza)
  - fine_tune_epochs: 25 (backbone liberado)
  - learning_rate: 1e-3 (cabeza)
  - fine_tune_learning_rate: 1e-5 (dividido por 10 en el código para fine-tuning efectivo ≈ 1e-6)
- augmentation:
  - rotation_range: 20
  - width_shift_range: 0.2
  - height_shift_range: 0.2
  - horizontal_flip: true
  - zoom_range: 0.2

12.2 Estructura de carpetas (resumen)

```
config/parameters.yaml
data/raw/dataset-original/dataset-original/<clase>/...
data/processed/{train,val,test}/<clase>/...
models/trained_models/YYYYMMDD_HHMMSS/{best_model.h5, final_model.h5, logs/, *.json, *.csv}
reports/evaluation/{classification_report.*, confusion_matrix.png, class_distribution.png, misclassifications.png, evaluation_results.json}
```

