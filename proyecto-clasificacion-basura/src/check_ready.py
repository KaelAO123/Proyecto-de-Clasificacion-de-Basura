# check_ready.py
from pathlib import Path

def check_evaluation_ready():
    """Verifica si todo está listo para evaluación"""
    
    print("🔍 VERIFICANDO PREPARACIÓN PARA EVALUACIÓN")
    print("=" * 50)
    
    checks = {
        "Modelo entrenado": Path("models/trained_models/best_model.h5").exists(),
        "Datos de test": Path("data/processed/test").exists(),
        "Datos de validation": Path("data/processed/val").exists(),
        "Requirements": Path("requirements.txt").exists()
    }
    
    all_ok = True
    for check, exists in checks.items():
        status = "✅" if exists else "❌"
        print(f"{status} {check}")
        if not exists:
            all_ok = False
    
    if all_ok:
        print("\n🎯 Todo listo! Puedes ejecutar:")
        print("   python evaluate_only.py")
    else:
        print("\n⚠️  Faltan algunos elementos:")
        if not checks["Modelo entrenado"]:
            print("   - Ejecuta primero: python src/model_training.py")
        if not checks["Datos de test"]:
            print("   - Ejecuta primero: python src/data_preprocessing.py")
    
    return all_ok

if __name__ == '__main__':
    check_evaluation_ready()