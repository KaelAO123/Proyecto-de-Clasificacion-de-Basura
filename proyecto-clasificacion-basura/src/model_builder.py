# src/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import yaml
from pathlib import Path

class ModelBuilder:
    def __init__(self, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.num_classes = len(self.config['data']['classes'])
        self.image_size = tuple(self.config['data']['image_size'])
        
    def create_base_model(self, model_name="MobileNetV2"):
        """Crea el modelo base con transfer learning"""
        
        base_models = {
            "MobileNetV2": applications.MobileNetV2,
            "EfficientNetB0": applications.EfficientNetB0,
            "ResNet50": applications.ResNet50,
            "VGG16": applications.VGG16,
            "InceptionV3": applications.InceptionV3
        }
        
        if model_name not in base_models:
            raise ValueError(f"Modelo no soportado: {model_name}")
        
        print(f"🔄 Creando modelo base: {model_name}")
        
        # Crear modelo base pre-entrenado
        base_model = base_models[model_name](
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Congelar capas base inicialmente
        base_model.trainable = False
        
        return base_model
    
    def build_model(self, base_model_name="MobileNetV2"):
        """Construye el modelo completo"""
        
        base_model = self.create_base_model(base_model_name)
        
        model = models.Sequential([
            # Capa base pre-entrenada
            base_model,
            
            # Capas de clasificación
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            
            # Capas densas personalizadas
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Capa de salida
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compila el modelo con optimizador y métricas"""
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                'precision',
                'recall',
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("✅ Modelo compilado:")
        print(f"   - Optimizer: Adam (lr={self.config['training']['learning_rate']})")
        print(f"   - Loss: categorical_crossentropy")
        print(f"   - Metrics: accuracy, precision, recall, AUC")
        
        return model
    
    def model_summary(self, model):
        """Muestra resumen del modelo"""
        print("\n📊 Resumen del Modelo:")
        model.summary()
        
        # Contar parámetros entrenables
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
        
        print(f"\n📈 Parámetros del modelo:")
        print(f"   - Entrenables: {trainable_params:,}")
        print(f"   - No entrenables: {non_trainable_params:,}")
        print(f"   - Total: {trainable_params + non_trainable_params:,}")

def create_ensemble_model(config_path="config/parameters.yaml"):
    """Crea un modelo ensemble para mejor performance"""
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    num_classes = len(config['data']['classes'])
    
    # Múltiples modelos base
    base_models = []
    for model_name in ["MobileNetV2", "EfficientNetB0", "ResNet50"]:
        builder = ModelBuilder(config_path)
        base_model = builder.create_base_model(model_name)
        
        # Capas personalizadas para cada modelo base
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        predictions = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        base_models.append(model)
    
    # Combinar outputs
    ensemble_output = layers.Average()([model.output for model in base_models])
    ensemble_model = models.Model(
        inputs=[model.input for model in base_models],
        outputs=ensemble_output
    )
    
    return ensemble_model