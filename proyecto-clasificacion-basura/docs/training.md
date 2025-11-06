# Entrenamiento del Modelo

Este proyecto entrena un clasificador con transferencia de aprendizaje (por defecto MobileNetV2) en dos etapas: cabeza densa y fine-tuning del backbone.

## Pasos

1) Asegúrate de tener datos procesados: `data/processed/{train,val,test}`.
2) Lanza el entrenamiento:

```powershell
python proyecto-clasificacion-basura\src\model_training.py
```

## ¿Qué hace `model_training.py`?

- Carga generadores de datos desde `data_generators.py` (con augmentación en train y reescalado en val/test).
- Construye el modelo con `ModelBuilder` (ver `model_builder.py`).
- Compila con Adam, `categorical_crossentropy` y métricas: accuracy, precision, recall, AUC.
- Entrena con callbacks:
  - EarlyStopping (paciencia 10, restaura mejores pesos por `val_loss`).
  - ModelCheckpoint (`best_model.h5` por `val_accuracy`).
  - ReduceLROnPlateau.
  - TensorBoard (logs en `models/trained_models/<run>/logs`).
  - CSVLogger (`training_log.csv`).

## Salidas del entrenamiento

- Carpeta de corrida: `models/trained_models/YYYYMMDD_HHMMSS/`
  - `best_model.h5`: Mejor modelo por `val_accuracy`.
  - `final_model.h5`: Modelo final tras el pipeline (guardado en `save_training_info`).
  - `training_log.csv`: Log por época.
  - `training_history.png`: Curvas (accuracy, loss, precision, recall).
  - `*_history.json`, `*_metadata.json`, `training_info.json`: Metadatos e historial.
  - `logs/`: Para TensorBoard (train/validation).

## TensorBoard (opcional)

Puedes inspeccionar las curvas en TensorBoard apuntando a `models/trained_models/<run>/logs`.
