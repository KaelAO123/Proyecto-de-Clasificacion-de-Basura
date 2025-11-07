# Módulo: `data_generators.py`

Generadores de imágenes para train/val/test con `ImageDataGenerator` de Keras, incluyendo augmentación para entrenamiento.

## Clase: `DataGenerator`

- __init__(config_path="config/parameters.yaml")
  - Lee `image_size` y `batch_size`.

- create_data_generators() -> (train_gen, val_gen, test_gen)
  - Train: rescale + rotación, shifts, flip, zoom, brightness, channel shift.
  - Val/Test: solo `rescale`.
  - Lee de `data/processed/{train,val,test}`.

- visualize_augmentations(train_generator, num_images=8)
  - Guarda `reports/figures/data_augmentation_examples.png` con ejemplos de augmentación.

- get_class_weights(train_generator) -> dict
  - Usa `sklearn.utils.compute_class_weight` para calcular pesos por clase.

### Uso rápido

```powershell
python proyecto-clasificacion-basura\\src\\data_generators.py
```
