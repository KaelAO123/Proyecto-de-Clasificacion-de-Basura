# Configuración (`config/parameters.yaml`)

Este archivo centraliza hiperparámetros y rutas.

```yaml
data:
  image_size: [224, 224]          # Tamaño (ancho, alto) de entrada del modelo
  batch_size: 32                  # Tamaño de batch para generadores
  validation_split: 0.15          # Ratio de validación (si aplica)
  test_split: 0.15                # Ratio de test (si aplica)
  classes: [cardboard, glass, metal, paper, plastic, trash]
  test_dir: data/processed/test   # Ruta al set de test
  train_dir: data/processed/train # Ruta al set de train
  val_dir: data/processed/val     # Ruta al set de validación

training:
  base_model: MobileNetV2         # Backbone para transfer learning
  initial_epochs: 15              # Épocas para entrenar solo la cabeza densa
  fine_tune_epochs: 25            # Épocas adicionales para fine-tuning
  learning_rate: 0.001            # LR inicial para cabeza densa
  fine_tune_learning_rate: 0.00001# LR base para fine-tuning (se reduce aún más)

augmentation:
  rotation_range: 20
  width_shift_range: 0.2
  height_shift_range: 0.2
  horizontal_flip: true
  zoom_range: 0.2
```

Notas:

- Cambiar `image_size` obliga a reentrenar el modelo.
- Ajustar `batch_size` según memoria de GPU/CPU.
- Las clases deben coincidir con las carpetas del dataset.
