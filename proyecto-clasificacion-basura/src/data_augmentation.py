import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path
import albumentations as A
import yaml

class AdvancedAugmentor:
    def __init__(self, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Albumentations transformations
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=15, 
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3)
        ])
    
    def augment_image(self, image):
        """Aplica aumentos a una imagen"""
        augmented = self.augmentation_pipeline(image=image)
        return augmented['image']
    
    def balance_dataset(self):
        """Balancea el dataset aplicando data augmentation a clases minoritarias"""
        print("Balanceando dataset...")
        
        train_path = Path("data/processed/train")
        
        # Contar imágenes por clase
        class_counts = {}
        for class_name in self.config['data']['classes']:
            class_path = train_path / class_name
            images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
            class_counts[class_name] = len(images)
        
        max_count = max(class_counts.values())
        target_count = int(max_count * 1.2)  # 20% más que la clase mayoritaria
        
        print(f"Target count por clase: {target_count}")
        
        # Aumentar clases minoritarias
        for class_name, count in class_counts.items():
            if count < target_count:
                print(f"Aumentando clase: {class_name} ({count} -> {target_count})")
                self.augment_class(class_name, count, target_count)
    
    def augment_class(self, class_name, current_count, target_count):
        """Aplica data augmentation a una clase específica"""
        class_path = Path("data/processed/train") / class_name
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
        
        needed = target_count - current_count
        augment_per_image = max(1, needed // current_count)
        
        augmented_count = 0
        for img_path in images:
            if augmented_count >= needed:
                break
                
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for i in range(augment_per_image):
                if augmented_count >= needed:
                    break
                    
                augmented_image = self.augment_image(image)
                
                new_filename = f"aug_{augmented_count}_{img_path.name}"
                new_path = class_path / new_filename
                
                augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(new_path), augmented_image_bgr)
                augmented_count += 1
        
        print(f"{class_name}: {augmented_count} imágenes aumentadas")

def main():
    augmentor = AdvancedAugmentor()
    augmentor.balance_dataset()

if __name__ == "__main__":
    main()