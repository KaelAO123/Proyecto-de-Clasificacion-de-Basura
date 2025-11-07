# Documentación del Proyecto: Clasificación de Basura con IA

Bienvenido a la documentación técnica del proyecto de Clasificación de Basura. Aquí encontrarás una guía completa para instalar, entrenar, evaluar y usar el sistema, además de la referencia de cada módulo en `src/`.

## Navegación rápida

- Arquitectura y flujo general: [architecture.md](docs/architecture.md)
- Instalación y preparación del entorno: [setup.md](docs/setup.md)
- Preparación de datos: [setup.md#preparación-de-datos]
- Entrenamiento del modelo: [training.md](docs/training.md)
- Evaluación y reportes: [reports.md](docs/reports.md)
- Inferencia y uso con cámara: [inference.md](docs/inference.md)
- Configuración (YAML): [configuration.md](docs/configuration.md)
- Referencia por módulo (`src/`):
  - [camera_classifier.md](docs/src/camera_classifier.md)
  - [data_augmentation.md](docs/src/data_augmentation.md)
  - [data_generators.md](docs/src/data_generators.md)
  - [data_loader.md](docs/src/data_loader.md)
  - [data_preprocessing.md](docs/src/data_preprocessing.md)
  - [model_builder.md](docs/src/model_builder.md)
  - [model_evaluation.md](docs/src/model_evaluation.md)
  - [model_training.md](docs/src/model_training.md)

## Qué hace este proyecto

- Clasifica imágenes de residuos en seis clases: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Entrena un modelo CNN con transferencia de aprendizaje (por defecto MobileNetV2) usando `TensorFlow/Keras`.
- Genera reportes de evaluación (matriz de confusión, clasificación por clase, métricas) en `reports/evaluation/`.
- Permite inferencia en tiempo real con la cámara del equipo y una interfaz para guiar reciclaje.

## Estructura resumida

- `src/`: código fuente (preprocesamiento, generadores, construcción, entrenamiento, evaluación e inferencia).
- `config/parameters.yaml`: hiperparámetros y rutas.
- `models/trained_models/`: corridas de entrenamiento con modelos y logs.
- `reports/`: gráficos y reportes de evaluación.
- `data/`: datos crudos y procesados (no versionados aquí).
