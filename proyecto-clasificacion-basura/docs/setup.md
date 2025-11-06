# Instalación y Preparación del Entorno

Sigue estos pasos para dejar el proyecto listo en Windows (PowerShell):

## Requisitos

- Python 3.9+ (recomendado 3.10/3.11)
- Pip actualizado (`python -m pip install --upgrade pip`)

Dependencias principales (ver `requirements.txt`): TensorFlow, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Albumentations, Split-Folders, PyYAML, Pillow, etc.

## Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r proyecto-clasificacion-basura\requirements.txt
```

## Preparación de datos

Coloca el dataset original con subcarpetas por clase en:

```
data/raw/dataset-original/dataset-original/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Luego ejecuta el preprocesamiento (limpieza + split):

```powershell
python proyecto-clasificacion-basura\src\data_preprocessing.py
```

Esto generará `data/processed/{train,val,test}/<clase>/...` con los ratios definidos en `config/parameters.yaml`.

## Verificar preparación para evaluación

```powershell
python proyecto-clasificacion-basura\src\check_ready.py
```

Si todo está OK, mostrará que el modelo, datos y requirements existen.
