# Arquitectura y Flujo del Proyecto

Este proyecto implementa un pipeline de clasificación de basura basado en Deep Learning con TensorFlow/Keras. El flujo principal es:

1. Preparación de datos: `src/data_preprocessing.py`
2. Generación de batches y data augmentation: `src/data_generators.py` y `src/data_augmentation.py`
3. Construcción y compilación del modelo (transfer learning): `src/model_builder.py`
4. Entrenamiento y callbacks (guardado, logs, tensorboard): `src/model_training.py`
5. Evaluación y reportes: `src/model_evaluation.py`
6. Inferencia en tiempo real con cámara: `src/camera_classifier.py` y `src/iphone.py`

## Módulos clave y responsabilidades

- `data_preprocessing.py`: Limpieza de dataset, división train/val/test, verificación de splits.
- `data_generators.py`: Generadores de datos con augmentación, visualización y pesos por clase.
- `data_augmentation.py`: Aumentos avanzados con Albumentations y balanceo de dataset.
- `model_builder.py`: Crea el backbone (MobileNetV2, EfficientNetB0, etc.) y la cabeza densa; compila el modelo.
- `model_training.py`: Entrena en dos etapas (cabeza y fine-tuning), usa callbacks y guarda artefactos.
- `model_evaluation.py`: Métricas, matriz de confusión, errores comunes, reportes CSV/MD/PNG.
- `camera_classifier.py`: Inferencia por cámara con overlay informativo de reciclaje.
- `iphone.py`: Interfaz web para usar la cámara del iPhone (guía visual) + clasificación con la cámara del PC.
- `model_utils.py`: Utilidades para guardar/cargar modelos e historiales.

## Estructura de carpetas

- `config/parameters.yaml`: Hiperparámetros (tamaños, batch, LR, augmentación) y rutas de datos.
- `models/trained_models/YYYYMMDD_HHMMSS/`: Modelos `.h5`, logs de TensorBoard, CSVs e información del entrenamiento.
- `reports/evaluation/`: Resultados de evaluación (matrices, reportes y gráficos).
- `data/raw/` y `data/processed/`: Dataset crudo y dividido.

## Flujo de datos

- Imágenes crudas → `data/raw/dataset-original/dataset-original/<clase>/...`
- División con `splitfolders` o fallback manual → `data/processed/{train,val,test}/<clase>/...`
- Generadores cargan imágenes normalizadas (1/255) al tamaño `image_size`.
- El modelo produce probabilidades softmax por clase; se guarda el mejor checkpoint y el historial.
