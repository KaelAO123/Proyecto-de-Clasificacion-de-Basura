# Módulo: `data_preprocessing.py`

Preprocesamiento y división del dataset. Limpia archivos no imagen, elimina carpetas residuales y genera splits `train/val/test`.

## Clase: `DataPreprocessor`

- __init__(config_path="config/parameters.yaml")
  - Rutas: `raw_data_path = data/raw/dataset-original/dataset-original` y `processed_path = data/processed`.
  - Lee clases desde `config`.

- clean_dataset()
  - Elimina carpetas `__MACOSX` y archivos no imagen en las clases.

- analyze_dataset() -> dict
  - Cuenta imágenes por clase y muestra formas de ejemplo.

- cleanup_previous_splits()
  - Borra `data/processed/{train,val,test,validation}` si existen.

- split_dataset()
  - Intenta usar `splitfolders.ratio(...)`; si falla, hace `_manual_split_dataset`.
  - Ratios por defecto: 0.7/0.15/0.15.

- verify_splits()
  - Verifica conteos por split y por clase.

### Uso completo

```powershell
python proyecto-clasificacion-basura\\src\\data_preprocessing.py
```
