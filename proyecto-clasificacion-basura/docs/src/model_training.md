# Módulo: `model_training.py`

Entrenamiento en dos etapas (cabeza + fine-tuning) con callbacks y guardado estandarizado.

## Clase: `ModelTrainer`

- __init__(config_path="config/parameters.yaml")
  - Crea `models/trained_models/<timestamp>/` como carpeta de la corrida.

- setup_callbacks() -> list
  - EarlyStopping (`val_loss`), ModelCheckpoint (`val_accuracy`, `best_model.h5`), ReduceLROnPlateau, TensorBoard, CSVLogger.

- save_model_and_history(model, history, model_name="best_model") -> Path
  - Guarda `.h5`, `_history.json` y `_metadata.json` en la carpeta de corrida.

- train_model(model, train_generator, val_generator, class_weights=None) -> (model, history_dict)
  - Etapa 1: entrena la cabeza densa por `initial_epochs`.
  - Etapa 2: descongela backbone (model.layers[0].trainable=True), recompila con LR bajo y continúa.
  - Combina historias y guarda el mejor modelo.

- combine_histories(history1, history2) -> dict
  - Concatena curvas por clave.

- plot_training_history(history)
  - Genera `training_history.png`.

- save_training_info(model, history, train_generator)
  - Guarda `final_model.h5` y `training_info.json` con metadatos (clases, índices, curvas finales, parámetros, etc.).

## Función: `main()`

Orquesta el pipeline: carga generadores (y pesos de clase), construye/compila el modelo (`ModelBuilder`), entrena (`ModelTrainer`), grafica y guarda info. Retorna `(trained_model, history, test_gen)`.
