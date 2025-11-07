# Módulo: `data_augmentation.py`

Aumentos avanzados con Albumentations y balanceo de dataset para clases minoritarias.

## Clase: `AdvancedAugmentor`

- __init__(config_path="config/parameters.yaml")
  - Carga `parameters.yaml` y construye un `A.Compose` con transformaciones (flip, rotate, blur, noise, brightness/contrast, coarse dropout, etc.).

- augment_image(image) -> image
  - Aplica el pipeline de aumentación a una imagen (numpy RGB).

- balance_dataset()
  - Cuenta imágenes por clase en `data/processed/train`.
  - Define un `target_count` y genera imágenes aumentadas para clases con menos muestras.

- augment_class(class_name, current_count, target_count)
  - Aplica aumentos a una clase y guarda imágenes como `aug_*.jpg`.

### Uso básico

```powershell
python proyecto-clasificacion-basura\\src\\data_augmentation.py
```
