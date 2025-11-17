# src/data_preprocessing.py
import os
import shutil
import numpy as np
from pathlib import Path
import splitfolders
import cv2
from sklearn.model_selection import train_test_split
import yaml

class DataPreprocessor:
    def __init__(self, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # CORREGIDO: Apuntar a la ruta correcta
        self.raw_data_path = Path("data/raw/dataset-original/dataset-original")
        self.processed_path = Path("data/processed")
        
    def clean_dataset(self):
        """Limpia el dataset de archivos no deseados"""
        print("Limpiando dataset...")
        
        # Eliminar __MACOSX de todas las ubicaciones posibles
        macosx_locations = [
            self.raw_data_path.parent / "__MACOSX",
            self.raw_data_path / "__MACOSX",
            self.processed_path / "__MACOSX"
        ]
        
        for macosx_path in macosx_locations:
            if macosx_path.exists():
                shutil.rmtree(macosx_path)
                print(f"Archivos __MACOSX eliminados: {macosx_path}")
        
        # Verificar y limpiar cada clase
        classes = self.config['data']['classes']
        for class_name in classes:
            class_path = self.raw_data_path / class_name
            if class_path.exists():
                # Eliminar archivos no imagen
                non_image_files = []
                for file_path in class_path.iterdir():
                    if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                        non_image_files.append(file_path)
                
                for file_path in non_image_files:
                    file_path.unlink()
                    print(f"Eliminado archivo no imagen: {file_path}")
                
                if non_image_files:
                    print(f"{class_name}: {len(non_image_files)} archivos no imagen eliminados")
    
    def analyze_dataset(self):
        """Analiza el dataset y muestra estadísticas"""
        print("\nAnalizando dataset...")
        print(f"Buscando en: {self.raw_data_path.absolute()}")
        
        classes = self.config['data']['classes']
        stats = {}
        total_images = 0
        
        for class_name in classes:
            class_path = self.raw_data_path / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
                stats[class_name] = len(images)
                total_images += len(images)
                
                # Mostrar ejemplo de imagen
                if images:
                    img = cv2.imread(str(images[0]))
                    if img is not None:
                        print(f"{class_name}: {len(images)} imágenes - Ejemplo: {img.shape}")
                    else:
                        print(f"{class_name}: {len(images)} imágenes - Error al leer ejemplo")
                else:
                    print(f"{class_name}: 0 imágenes")
            else:
                print(f"{class_name}: Directorio no encontrado - {class_path}")
        
        print(f"\nTOTAL DATASET: {total_images} imágenes")
        return stats
    
    def cleanup_previous_splits(self):
        """Limpia divisiones anteriores"""
        print("\nLimpiando divisiones anteriores...")
        for split in ['train', 'val', 'test', 'validation']:
            split_path = self.processed_path / split
            if split_path.exists():
                shutil.rmtree(split_path)
                print(f"Eliminado: {split_path}")
    
    def split_dataset(self):
        """Divide el dataset en train/validation/test"""
        print("\nDividiendo dataset...")
        
        # Limpiar divisiones anteriores
        self.cleanup_previous_splits()
        
        # Obtener ratios de forma correcta
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        print(f"Ratios: Train {train_ratio}, Val {val_ratio}, Test {test_ratio}")
        print(f"Origen: {self.raw_data_path}")
        print(f"Destino: {self.processed_path}")
        
        # Verificar que el origen existe y tiene datos
        if not self.raw_data_path.exists():
            print(f"Error: No existe el directorio origen {self.raw_data_path}")
            return
        
        # Contar imágenes totales
        total_images = 0
        for class_name in self.config['data']['classes']:
            class_path = self.raw_data_path / class_name
            if class_path.exists():
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
                total_images += len(images)
        
        if total_images == 0:
            print("Error: No se encontraron imágenes en el directorio origen")
            return
        
        print(f"Imágenes a dividir: {total_images}")
        
        # Usar splitfolders para división balanceada
        try:
            splitfolders.ratio(
                str(self.raw_data_path),
                output=str(self.processed_path),
                seed=42,
                ratio=(train_ratio, val_ratio, test_ratio),
                group_prefix=None,
                move=False  # Copiar en lugar de mover
            )
            print("Dataset dividido exitosamente!")
        except Exception as e:
            print(f"Error al dividir dataset: {e}")
            # Fallback: división manual
            self._manual_split_dataset(train_ratio, val_ratio, test_ratio)
    
    def _manual_split_dataset(self, train_ratio, val_ratio, test_ratio):
        """División manual como fallback"""
        print("Usando división manual...")
        
        for class_name in self.config['data']['classes']:
            class_path = self.raw_data_path / class_name
            if not class_path.exists():
                print(f"Saltando {class_name}: directorio no existe")
                continue
                
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
            if not images:
                print(f"Saltando {class_name}: sin imágenes")
                continue
                
            print(f"Procesando {class_name}: {len(images)} imágenes")
            
            # Mezclar imágenes
            np.random.seed(42)
            np.random.shuffle(images)
            
            # Calcular splits
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Dividir
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Crear directorios destino
            for split in ['train', 'val', 'test']:
                (self.processed_path / split / class_name).mkdir(parents=True, exist_ok=True)
            
            # Copiar archivos
            for img in train_images:
                dest = self.processed_path / 'train' / class_name / img.name
                shutil.copy2(img, dest)
            
            for img in val_images:
                dest = self.processed_path / 'val' / class_name / img.name
                shutil.copy2(img, dest)
            
            for img in test_images:
                dest = self.processed_path / 'test' / class_name / img.name
                shutil.copy2(img, dest)
            
            print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    def verify_splits(self):
        """Verifica que las divisiones sean correctas"""
        print("\nVerificando divisiones...")
        
        splits = ['train', 'val', 'test']
        grand_total = 0
        
        for split in splits:
            split_path = self.processed_path / split
            total_images = 0
            print(f"\n{split.upper()}:")
            
            for class_name in self.config['data']['classes']:
                class_path = split_path / class_name
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
                print(f"  {class_name}: {len(images)} imágenes")
                total_images += len(images)
            
            print(f"  TOTAL {split}: {total_images} imágenes")
            grand_total += total_images
        
        print(f"\nGRAND TOTAL: {grand_total} imágenes")

# def main():
#     preprocessor = DataPreprocessor()
    
#     # Ejecutar pipeline completo
#     preprocessor.clean_dataset()
#     stats = preprocessor.analyze_dataset()
#     preprocessor.split_dataset()
#     preprocessor.verify_splits()

# if __name__ == "__main__":
#     main()

prueba = DataPreprocessor()
prueba.verify_splits()