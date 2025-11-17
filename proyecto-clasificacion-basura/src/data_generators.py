# src/data_generators.py
import tensorflow as tf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class DataGenerator:
    def __init__(self, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.image_size = tuple(self.config['data']['image_size'])
        self.batch_size = self.config['data']['batch_size']
        
    def create_data_generators(self):
        """Crea generadores de datos para train, validation y test"""
        
        # Data augmentation para training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config['augmentation']['rotation_range'],
            width_shift_range=self.config['augmentation']['width_shift_range'],
            height_shift_range=self.config['augmentation']['height_shift_range'],
            horizontal_flip=self.config['augmentation']['horizontal_flip'],
            zoom_range=self.config['augmentation']['zoom_range'],
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        # Solo rescaling para validation y test
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )
        
        # Generadores
        train_generator = train_datagen.flow_from_directory(
            "data/processed/train",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        val_generator = test_datagen.flow_from_directory(
            "data/processed/val",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            "data/processed/test",
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print("Generadores de datos creados:")
        print(f"   - Training: {train_generator.samples} imágenes")
        print(f"   - Validation: {val_generator.samples} imágenes") 
        print(f"   - Test: {test_generator.samples} imágenes")
        print(f"   - Classes: {list(train_generator.class_indices.keys())}")
        
        return train_generator, val_generator, test_generator
    
    def visualize_augmentations(self, train_generator, num_images=8):
        """Visualiza ejemplos de data augmentation"""
        print("\nVisualizando aumentos de datos...")
        
        # Obtener un batch de imágenes
        x_batch, y_batch = next(train_generator)
        
        # Configurar plot
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i in range(num_images):
            axes[i].imshow(x_batch[i])
            class_idx = np.argmax(y_batch[i])
            class_name = list(train_generator.class_indices.keys())[class_idx]
            axes[i].set_title(f'Class: {class_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('reports/figures/data_augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_class_weights(self, train_generator):
        """Calcula pesos de clases para dataset desbalanceado"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(train_generator.classes)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=train_generator.classes
        )
        
        class_weights_dict = dict(zip(classes, class_weights))
        
        print("Pesos de clases calculados:")
        for class_idx, weight in class_weights_dict.items():
            class_name = list(train_generator.class_indices.keys())[class_idx]
            print(f"   {class_name}: {weight:.2f}")
        
        return class_weights_dict

def main():
    generator = DataGenerator()
    train_gen, val_gen, test_gen = generator.create_data_generators()
    generator.visualize_augmentations(train_gen)
    class_weights = generator.get_class_weights(train_gen)
    
    return train_gen, val_gen, test_gen, class_weights

if __name__ == "__main__":
    train_gen, val_gen, test_gen, class_weights = main()