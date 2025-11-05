# src/model_training.py
import sys
import tensorflow as tf
from tensorflow.keras import callbacks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import yaml
import json
from pathlib import Path
# src/model_utils.py
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models_dir = Path("models/trained_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear directorio con timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.models_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_callbacks(self):
        """Configura callbacks para el entrenamiento"""
        
        callbacks_list = [
            # Early Stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model Checkpoint
            callbacks.ModelCheckpoint(
                filepath=str(self.run_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Reduce Learning Rate on Plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir=str(self.run_dir / "logs"),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # CSV Logger
            callbacks.CSVLogger(
                filename=str(self.run_dir / "training_log.csv")
            )
        ]
        
        return callbacks_list
    
    def save_model_and_history(self, model, history, model_name="best_model"):
        """Guarda el modelo y su historial de entrenamiento"""
        
        # Usar self.run_dir en lugar de crear nuevo directorio
        model_path = self.run_dir / f"{model_name}.h5"
        model.save(str(model_path))
        logger.info(f"✅ Modelo guardado: {model_path}")
        
        # Guardar historial
        history_path = self.run_dir / f"{model_name}_history.json"
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
        
        metadata_path = self.run_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return model_path

    def train_model(self, model, train_generator, val_generator, class_weights=None):
        """Entrena el modelo en dos etapas"""
        
        callbacks_list = self.setup_callbacks()
        
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("=" * 60)
        
        # Etapa 1: Entrenar solo capas densas
        print("\n📚 ETAPA 1: Entrenamiento de capas densas")
        print("-" * 50)
        
        history_stage1 = model.fit(
            train_generator,
            epochs=self.config['training']['initial_epochs'],
            validation_data=val_generator,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Etapa 2: Fine-tuning (descongelar capas base)
        print("\n🎯 ETAPA 2: Fine-tuning completo")
        print("-" * 50)
        
        # Descongelar capas base
        model.layers[0].trainable = True
        
        # Recompilar con learning rate más bajo
        fine_tune_lr = self.config['training']['fine_tune_learning_rate']
        model.compile(
            optimizer=tf.keras.optimizers.Adam(fine_tune_lr / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        # Continuar entrenamiento
        history_stage2 = model.fit(
            train_generator,
            epochs=self.config['training']['initial_epochs'] + 
                  self.config['training']['fine_tune_epochs'],
            initial_epoch=history_stage1.epoch[-1] + 1,
            validation_data=val_generator,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Combinar historiales
        combined_history = self.combine_histories(history_stage1, history_stage2)
        
        model_path = self.save_model_and_history(model, combined_history, "best_model")
        
        print(f"💾 Modelo guardado en: {model_path}")
        
        return model, combined_history  # Corregido: usar 'model' en lugar de 'trained_model'

    def combine_histories(self, history1, history2):
        """Combina los historiales de las dos etapas de entrenamiento"""
        combined_history = {}
        for key in history1.history.keys():
            combined_history[key] = history1.history[key] + history2.history[key]
        return combined_history
    
    def plot_training_history(self, history):
        """Genera gráficos del entrenamiento"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Accuracy
        axes[0].plot(history['accuracy'], label='Training Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history['loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Precision
        axes[2].plot(history['precision'], label='Training Precision')
        axes[2].plot(history['val_precision'], label='Validation Precision')
        axes[2].set_title('Model Precision')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True)
        
        # Recall
        axes[3].plot(history['recall'], label='Training Recall')
        axes[3].plot(history['val_recall'], label='Validation Recall')
        axes[3].set_title('Model Recall')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.savefig(str(self.run_dir / "training_history.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_info(self, model, history, train_generator):
        """Guarda información del entrenamiento"""
        
        # Guardar modelo final
        model.save(str(self.run_dir / "final_model.h5"))
        
        # Guardar configuración
        training_info = {
            'timestamp': self.timestamp,
            'model_architecture': 'MobileNetV2 with Custom Head',
            'image_size': self.config['data']['image_size'],
            'batch_size': self.config['data']['batch_size'],
            'initial_epochs': self.config['training']['initial_epochs'],
            'fine_tune_epochs': self.config['training']['fine_tune_epochs'],
            'classes': self.config['data']['classes'],
            'class_indices': train_generator.class_indices,
            'final_training_accuracy': history['accuracy'][-1],
            'final_validation_accuracy': history['val_accuracy'][-1],
            'total_parameters': model.count_params()
        }
        
        with open(str(self.run_dir / "training_info.json"), 'w') as f:
            json.dump(training_info, f, indent=4)
        
        print(f"\n💾 Modelo e información guardados en: {self.run_dir}")

def main():
    from src.data_generators import main as data_generator_main
    from src.model_builder import ModelBuilder
    
    # Cargar datos
    print("📥 Cargando generadores de datos...")
    train_gen, val_gen, test_gen, class_weights = data_generator_main()  # Corregido: solo una llamada
    
    # Construir modelo
    print("🔨 Construyendo modelo...")
    builder = ModelBuilder()
    model = builder.build_model()
    model = builder.compile_model(model)
    builder.model_summary(model)
    
    # Entrenar modelo
    print("🏋️‍♂️ Iniciando entrenamiento...")
    trainer = ModelTrainer()
    trained_model, history = trainer.train_model(
        model, train_gen, val_gen, class_weights
    )
    
    # Visualizar resultados
    trainer.plot_training_history(history)
    trainer.save_training_info(trained_model, history, train_gen)
    
    return trained_model, history, test_gen

if __name__ == "__main__":
    trained_model, history, test_gen = main()