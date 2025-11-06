# Módulo: `model_utils.py`

Utilidades para guardar/cargar modelos e historiales y obtener el modelo más reciente.

## Funciones

- save_model_and_history(model, history, model_name="best_model") -> Path
  - Guarda `models/trained_models/<model_name>.h5`, `<model_name>_history.json`, `<model_name>_metadata.json`.

- load_model_and_history(model_name="best_model") -> (model, history)
  - Carga el `.h5` y opcionalmente el historial si existe.

- get_latest_model() -> (model, history)
  - Busca el `.h5` más reciente en `models/trained_models/` y lo carga.

Notas: En el pipeline actual, `ModelTrainer` guarda artefactos por corrida en subcarpetas con timestamp; estas utilidades ofrecen un fallback simple por nombre o el más reciente en un plano.
