# Módulo: `check_ready.py`

Pequeña utilidad para verificar si el entorno está listo para una evaluación.

## Función: `check_evaluation_ready() -> bool`

Chequea la existencia de:

- `models/trained_models/best_model.h5`
- `data/processed/test`
- `data/processed/val`
- `requirements.txt`

Imprime el estado y devuelve `True/False`.

### Uso

```powershell
python proyecto-clasificacion-basura\\src\\check_ready.py
```
