# Módulo: `camera_classifier.py`

Clasificación en tiempo real con la cámara del equipo. Muestra la clase predicha, confianza, barras de probabilidad por clase y consejos de reciclaje por categoría.

## Clase: `CameraGarbageClassifier`

- __init__(model_path=None, config_path="config/parameters.yaml")
  - Carga `parameters.yaml` y un modelo `.h5` (el más reciente si `model_path` es None).
  - Atributos: `img_size`, `class_names`, `recycling_info`, `category_colors`.

- preprocess_frame(frame) -> np.ndarray
  - Entrada: `frame` (OpenCV, BGR).
  - Procesos: resize → BGR→RGB → normaliza [0,1] → `expand_dims` → (1, H, W, 3).

- predict_frame(frame) -> (predicted_class: str, confidence: float, probs: np.ndarray)
  - Llama a `preprocess_frame` y `model.predict`.

- draw_prediction_info(frame, prediction, confidence, all_probabilities) -> frame
  - Pinta overlays: barra superior con clase y confianza, instrucciones de reciclaje, barras de probabilidad y rectángulo central de enfoque.

- run_camera_classification(camera_index=0)
  - Abre la cámara, dibuja overlays y permite guardar frames.
  - Controles: Q (salir), S (guardar), R (reset stats).

## Uso rápido

```powershell
python proyecto-clasificacion-basura\\src\\camera_classifier.py
```

Requiere un modelo entrenado en `models/trained_models/**/best_model.h5` y configuración en `config/parameters.yaml`.
