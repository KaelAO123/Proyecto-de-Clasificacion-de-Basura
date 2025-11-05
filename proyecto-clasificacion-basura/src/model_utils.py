# src/model_utils.py
import tensorflow as tf
import json
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_model_and_history(model, history, model_name="best_model"):
    """Guarda el modelo y su historial de entrenamiento"""
    
    models_dir = Path("models/trained_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    model_path = models_dir / f"{model_name}.h5"
    model.save(str(model_path))
    logger.info(f"✅ Modelo guardado: {model_path}")
    
    # Guardar historial
    history_path = models_dir / f"{model_name}_history.json"
    # Convertir numpy arrays a listas para JSON
    history_dict = {}
    for key, values in history.items():
        history_dict[key] = [float(value) for value in values]
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    logger.info(f"✅ Historial guardado: {history_path}")
    
    # Guardar metadatos
    metadata = {
        'model_name': model_name,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape,
        'num_parameters': model.count_params(),
        'classes': ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    }
    
    metadata_path = models_dir / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return model_path

def load_model_and_history(model_name="best_model"):
    """Carga un modelo y su historial guardado"""
    
    models_dir = Path("models/trained_models")
    model_path = models_dir / f"{model_name}.h5"
    history_path = models_dir / f"{model_name}_history.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    # Cargar modelo
    model = tf.keras.models.load_model(str(model_path))
    logger.info(f"✅ Modelo cargado: {model_path}")
    
    # Cargar historial si existe
    history = {}
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        logger.info(f"✅ Historial cargado: {history_path}")
    
    return model, history

def get_latest_model():
    """Obtiene el modelo más reciente entrenado"""
    
    models_dir = Path("models/trained_models")
    model_files = list(models_dir.glob("*.h5"))
    
    if not model_files:
        raise FileNotFoundError("No se encontraron modelos entrenados")
    
    # Ordenar por fecha de modificación (más reciente primero)
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_name = latest_model.stem
    
    return load_model_and_history(model_name)