# organize_project.py
import os
import shutil
from pathlib import Path

def create_project_structure():
    """Crea la estructura completa del proyecto"""
    
    base_dir = Path("proyecto-clasificacion-basura")
    
    # Directorios principales
    directories = [
        "data/raw",
        "data/processed/train",
        "data/processed/validation", 
        "data/processed/test",
        "data/augmented",
        "notebooks",
        "src",
        "models/trained_models",
        "models/saved_models",
        "config",
        "reports/figures",
        "docs"
    ]
    
    # Crear directorios
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Creando: {base_dir / directory}")
    
    # Mover dataset original
    original_dataset = Path("dataset-original/dataset-original")
    if original_dataset.exists():
        dest = base_dir / "data/raw/dataset-original"
        shutil.copytree(original_dataset, dest, dirs_exist_ok=True)
        print(f"✓ Dataset movido a: {dest}")
    
    # Crear archivos básicos
    (base_dir / "requirements.txt").touch()
    (base_dir / "README.md").touch()
    (base_dir / "config/parameters.yaml").touch()
    
    print("\n🎯 Estructura creada exitosamente!")

if __name__ == "__main__":
    create_project_structure()