"""
Módulo del clasificador en tiempo real
"""

import threading
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json

class RealtimeClassifier:
    """Clasificador para inferencia en tiempo real"""
    
    def __init__(self, config_path=None, models_dir=None):
        # Rutas por defecto
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / 'config' / 'parameters.yaml'
        if models_dir is None:
            models_dir = Path(__file__).resolve().parents[2] / 'models' / 'trained_models'
        
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        
        # Cargar configuración
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Encontrar modelo más reciente
        model_path = self._find_latest_model()
        print(model_path)
        if not model_path:
            raise FileNotFoundError('No se encontraron modelos .h5 en el directorio de modelos')
        
        print(f'Cargando modelo: {model_path.name}')
        self.model = tf.keras.models.load_model(str(model_path))
        
        # Configuración
        self.img_size = tuple(self.config['data']['image_size'])
        self.class_names = self.config['data']['classes']
        self.lock = threading.Lock()
        
        print(f'Modelo cargado: {len(self.class_names)} clases')
        print(f'Tamaño de imagen: {self.img_size}')
    
    def _find_latest_model(self):
        """Encontrar el modelo más reciente en el directorio de modelos"""
        model_files = list(self.models_dir.glob('**/*.h5'))
        if not model_files:
            return None
        
        # Preferir best_model.h5 si existe
        best_models = [p for p in model_files if p.name == 'best_model.h5']
        if best_models:
            # Encontrar el best_model más reciente
            return max(best_models, key=lambda p: p.stat().st_mtime)
        else:
            # Usar cualquier modelo .h5
            return max(model_files, key=lambda p: p.stat().st_mtime)
    
    def preprocess_image(self, image_bytes):
        """Preprocesar imagen para el modelo"""
        try:
            # Abrir y convertir imagen
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Redimensionar
            image = image.resize(self.img_size, Image.Resampling.LANCZOS)
            
            # Convertir a array y normalizar
            arr = np.array(image).astype(np.float32) / 255.0
            
            # Expandir dimensión para batch
            arr = np.expand_dims(arr, axis=0)
            
            return arr
            
        except Exception as e:
            raise ValueError(f"Error preprocesando imagen: {e}")
    
    def predict_bytes(self, image_bytes):
        """Realizar predicción sobre bytes de imagen"""
        try:
            # Preprocesar
            x = self.preprocess_image(image_bytes)
            
            # Predecir (con lock para thread safety)
            with self.lock:
                predictions = self.model.predict(x, verbose=0)[0]
            
            # Obtener resultados
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx])
            
            # Crear diccionario de probabilidades
            probabilities = {
                self.class_names[i]: float(predictions[i]) 
                for i in range(len(self.class_names))
            }
            
            return {
                'class_idx': class_idx,
                'class_name': self.class_names[class_idx],
                'confidence': confidence,
                'probabilities': probabilities,
                'success': True
            }
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return {
                'error': str(e),
                'success': False,
                'class_name': 'ERROR',
                'confidence': 0.0
            }
    
    def get_model_info(self):
        """Obtener información del modelo"""
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'classes': self.class_names,
            'image_size': self.img_size
        }