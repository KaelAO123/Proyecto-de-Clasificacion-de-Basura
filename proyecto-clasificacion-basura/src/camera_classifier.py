# src/camera_classifier.py
import cv2
import tensorflow as tf
import numpy as np
import time
from pathlib import Path
import yaml
import sys
import os

# Añadir el directorio padre al path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CameraGarbageClassifier:
    def __init__(self, model_path=None, config_path="config/parameters.yaml"):
        # Cargar configuración
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Cargar modelo
        if model_path is None:
            # Buscar el modelo más reciente
            models_dir = Path("models/trained_models")
            model_files = list(models_dir.glob("**/*.h5"))
            if not model_files:
                raise FileNotFoundError("No se encontraron modelos entrenados")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Cargando modelo: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        print("Modelo cargado exitosamente")
        
        # Configuración
        self.img_size = tuple(self.config['data']['image_size'])
        self.class_names = self.config['data']['classes']
        
        # Información de reciclaje
        self.recycling_info = {
            "cardboard": {
                "category": "RECICLABLE", 
                "container": "Contenedor AZUL",
                "instructions": "Doblar y colocar limpio"
            },
            "glass": {
                "category": "RECICLABLE",
                "container": "Contenedor VERDE", 
                "instructions": "Separar por colores"
            },
            "metal": {
                "category": "RECICLABLE",
                "container": "Contenedor AMARILLO",
                "instructions": "Latas limpias"
            },
            "paper": {
                "category": "RECICLABLE", 
                "container": "Contenedor AZUL",
                "instructions": "Sin manchas de grasa"
            },
            "plastic": {
                "category": "RECICLABLE",
                "container": "Contenedor AMARILLO", 
                "instructions": "Enjuagar y secar"
            },
            "trash": {
                "category": "NO RECICLABLE",
                "container": "Contenedor GRIS/ORGÁNICO",
                "instructions": "Reducir consumo"
            }
        }
        
        # Colores para cada categoría
        self.category_colors = {
            "cardboard": (0, 0, 255),      # Azul
            "glass": (0, 255, 0),          # Verde  
            "metal": (0, 255, 255),        # Amarillo
            "paper": (255, 0, 0),          # Rojo (para contraste)
            "plastic": (0, 165, 255),      # Naranja
            "trash": (128, 128, 128)       # Gris
        }
    
    def preprocess_frame(self, frame):
        """Preprocesa un frame para la predicción"""
        # Redimensionar
        img = cv2.resize(frame, self.img_size)
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalizar
        img = img.astype(np.float32) / 255.0
        # Expandir dimensiones
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict_frame(self, frame):
        """Realiza predicción en un frame"""
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def draw_prediction_info(self, frame, prediction, confidence, all_probabilities):
        """Dibuja información de la predicción en el frame"""
        height, width = frame.shape[:2]
        
        # Información de la clase predicha
        class_info = self.recycling_info[prediction]
        
        # Color basado en la categoría
        color = self.category_colors[prediction]
        
        # Barra superior con información principal
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Texto principal
        main_text = f"{prediction.upper()} - {class_info['category']}"
        confidence_text = f"Confianza: {confidence:.1%}"
        
        cv2.putText(frame, main_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instrucciones de reciclaje
        instructions_y = height - 120
        cv2.rectangle(frame, (0, instructions_y), (width, height), (0, 0, 0), -1)
        
        cv2.putText(frame, f"{class_info['container']}", 
                   (10, instructions_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{class_info['instructions']}", 
                   (10, instructions_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barra de probabilidades en la parte inferior
        bar_height = 20
        bar_width = width // len(self.class_names)
        
        for i, class_name in enumerate(self.class_names):
            prob = all_probabilities[i]
            bar_length = int(prob * bar_width)
            
            # Dibujar barra
            x_start = i * bar_width
            x_end = x_start + bar_length
            y_start = height - bar_height - 120
            y_end = y_start + bar_height
            
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), 
                         self.category_colors[class_name], -1)
            cv2.rectangle(frame, (x_start, y_start), (x_start + bar_width, y_end), 
                         (255, 255, 255), 1)
            
            # Etiqueta de la clase (solo si hay espacio)
            if bar_width > 50:
                cv2.putText(frame, class_name[:3], 
                           (x_start + 2, y_start + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Rectángulo central para enfocar el objeto
        center_x, center_y = width // 2, height // 2
        rect_size = 300
        cv2.rectangle(frame, 
                     (center_x - rect_size // 2, center_y - rect_size // 2),
                     (center_x + rect_size // 2, center_y + rect_size // 2),
                     color, 3)
        
        # Texto "Coloca el objeto aquí"
        cv2.putText(frame, "COLOCA EL OBJETO AQUI", 
                   (center_x - 100, center_y - rect_size // 2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def run_camera_classification(self, camera_index=0):
        """Ejecuta la clasificación en tiempo real con la cámara"""
        
        print("Iniciando cámara...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("No se pudo abrir la cámara")
            return
        
        # Configurar resolución de la cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Cámara lista")
        print("\nCONTROLES:")
        print("   - Presiona 'Q' para salir")
        print("   - Presiona 'S' para guardar frame actual")
        print("   - Presiona 'R' para reiniciar estadísticas")
        print("\nColoca el objeto de basura en el rectángulo central")
        
        # Variables para FPS y estadísticas
        fps_counter = 0
        fps_time = time.time()
        frame_count = 0
        predictions_history = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error leyendo frame de la cámara")
                    break
                
                # Voltear frame horizontalmente (espejo)
                frame = cv2.flip(frame, 1)
                
                # Realizar predicción cada 5 frames para mejor performance
                if frame_count % 5 == 0:
                    predicted_class, confidence, all_probs = self.predict_frame(frame)
                    predictions_history.append(predicted_class)
                    # Mantener solo las últimas 10 predicciones
                    if len(predictions_history) > 10:
                        predictions_history.pop(0)
                
                # Dibujar información en el frame
                frame = self.draw_prediction_info(frame, predicted_class, confidence, all_probs)
                
                # Calcular y mostrar FPS
                frame_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_time = time.time()
                    
                    # Mostrar FPS en consola
                    print(f"FPS: {fps}, Predicción: {predicted_class} ({confidence:.1%})")
                
                # Mostrar frame
                cv2.imshow('Clasificador de Basura - ECO AI', frame)
                
                # Controles de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Guardar frame actual
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}_{predicted_class}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame guardado como: {filename}")
                elif key == ord('r'):
                    # Reiniciar estadísticas
                    predictions_history = []
                    print("Estadísticas reiniciadas")
                
        except KeyboardInterrupt:
            print("\nDeteniendo clasificación...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Cámara liberada")

def main():
    """Función principal"""
    print("INICIANDO CLASIFICADOR DE BASURA EN TIEMPO REAL")
    print("=" * 50)
    
    try:
        # Crear clasificador
        classifier = CameraGarbageClassifier()
        
        # Intentar diferentes índices de cámara
        camera_found = False
        for camera_index in [0, 1, 2]:
            print(f"🔍 Probando cámara índice {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                camera_found = True
                cap.release()
                print(f"Cámara encontrada en índice {camera_index}")
                break
            cap.release()
        
        if not camera_found:
            print("No se encontró ninguna cámara conectada")
            return
        
        # Ejecutar clasificación
        classifier.run_camera_classification(camera_index)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nSoluciones posibles:")
        print("1. Asegúrate de que hay un modelo entrenado")
        print("2. Verifica que la cámara esté conectada")
        print("3. Ejecuta: pip install opencv-python")
        print("4. Prueba con una cámara externa")

if __name__ == "__main__":
    main()