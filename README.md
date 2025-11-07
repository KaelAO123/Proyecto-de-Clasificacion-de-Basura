# Proyecto de Clasificacion de Basura

Para que el proyecto funcione siga los siguientes pasos:

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu_usuario/proyecto-clasificacion-basura.git
   cd proyecto-clasificacion-basura
   ```

2. Crea un entorno virtual e instala dependencias:

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/Mac
   venv\Scripts\activate      # En Windows

   pip install -r requirements.txt
   ```

## Descarga y organización del dataset

### Instalar Git LFS

Asegúrate de tener instalado **Git LFS** (Large File Storage).
Si no lo tienes, descárgalo desde 👉 [https://git-lfs.com](https://git-lfs.com)

Luego ejecuta:

```bash
git lfs install
```

### Clonar el dataset desde Hugging Face

```bash
git clone https://huggingface.co/datasets/garythung/trashnet
```

### Ejecutar el script organizador

Este script prepara la estructura de carpetas para el dataset:

```bash
python organizador.py
```

### Descomprimir el dataset

Dentro del repositorio clonado encontrarás un archivo `dataset_original.zip`.
Descomprímelo en:

```
data/raw/
```

La estructura final debe verse así:

```
data/raw/dataset-original/dataset-original/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Ejecución del proyecto

Una vez tengas el dataset listo y las dependencias instaladas, puedes ejecutar el pipeline completo:

```bash
# Preprocesamiento de datos
python src/data_preprocessing.py

# Entrenamiento del modelo
python src/model_training.py

# Evaluación y generación de reportes
python src/model_evaluation.py

# Inferencia en tiempo real (cámara)
python src/camera_classifier.py
```

O simplemente ejecutar el flujo completo desde el archivo principal:

```bash
python main.py
```

---

## Resultados esperados

* **Precisión esperada:** ≥ 80%
* **Número de clases:** 6
* **Salidas del modelo:**

  * Clase predicha
  * Nivel de confianza (%)
  * Recomendación de reciclaje

---

## Información académica

**Universidad:** Universidad Mayor de San Andrés (UMSA)
**Facultad:** Ciencias Puras y Naturales
**Carrera:** Informática
**Materia:** Inteligencia Artificial
**Sigla:** INF-372.
**Docente:** Lic. Freddy Miguel Toledo Paz
**Integrantes:**

* Bautista Mollo Denzel Guden
* Reyes Barja Carlos Eduardo
* Rojas Condori Fidel Ángel
  **Gestión:** 2025

## Agradecimientos

Agradecemos al repositorio **TrashNet** por el dataset base y a la comunidad open-source por las herramientas que hacen posible este proyecto educativo.

Si quiere leer mas de la documentacion del proyecto entre a este [README](proyecto-clasificacion-basura/README.md) donde hablamos mas de la estrucutra del trabajo.
