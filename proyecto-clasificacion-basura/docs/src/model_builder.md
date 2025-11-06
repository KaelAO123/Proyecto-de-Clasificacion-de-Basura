# Módulo: `model_builder.py`

Construcción y compilación del modelo con transferencia de aprendizaje.

## Clase: `ModelBuilder`

- __init__(config_path="config/parameters.yaml")
  - Define `num_classes` y `image_size` desde la configuración.

- create_base_model(model_name="MobileNetV2") -> tf.keras.Model
  - Soporta: `MobileNetV2`, `EfficientNetB0`, `ResNet50`, `VGG16`, `InceptionV3`.
  - Carga pesos `imagenet`, `include_top=False`, `input_shape=(H, W, 3)`.
  - Congela el backbone (trainable=False) en la fase inicial.

- build_model(base_model_name="MobileNetV2") -> tf.keras.Model
  - Agrega `GlobalAveragePooling2D` + bloques densos con `Dropout`/`BatchNorm` y salida `softmax` (`num_classes`).

- compile_model(model) -> tf.keras.Model
  - Adam con `learning_rate` del YAML; loss `categorical_crossentropy`.
  - Métricas: `accuracy`, `precision`, `recall`, `AUC`.

- model_summary(model)
  - Imprime resumen y cuenta parámetros entrenables/no entrenables.

## Función: `create_ensemble_model(config_path)`

Crea un ensemble promediando salidas de varios backbones (MobileNetV2, EfficientNetB0, ResNet50) y devuelve un `Model` con `Average()` sobre logits/softmax.
