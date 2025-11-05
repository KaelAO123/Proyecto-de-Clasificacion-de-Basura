# main.py
from src.data_preprocessing import DataPreprocessor
from src.data_augmentation import AdvancedAugmentor
from src.data_generators import DataGenerator
from src.model_builder import ModelBuilder
from src.model_training import ModelTrainer
from src.model_evaluation import evaluate_complete_pipeline
from src.data_generators import main as data_generator_main

def complete_pipeline():
    """Ejecuta el pipeline completo de ML"""
    
    print("🚀 INICIANDO PIPELINE COMPLETO DE CLASIFICACIÓN DE BASURA")
    print("=" * 60)
    
    # Etapa 1: Preprocesamiento
    print("\n📥 ETAPA 1: Preprocesamiento de Datos")
    print("-" * 40)
    preprocessor = DataPreprocessor()
    preprocessor.clean_dataset()
    preprocessor.analyze_dataset()
    preprocessor.split_dataset()
    preprocessor.verify_splits()
    
    # Etapa 2: Data Augmentation
    print("\n🎨 ETAPA 2: Data Augmentation")
    print("-" * 40)
    augmentor = AdvancedAugmentor()
    augmentor.balance_dataset()
    
    # Etapa 3: Preparación de Datos
    print("\n🔧 ETAPA 3: Preparación de Datos")
    print("-" * 40)
    data_gen = DataGenerator()
    train_gen, val_gen, test_gen, class_weights = data_generator_main()
    
    # Etapa 4: Construcción del Modelo
    print("\n🔨 ETAPA 4: Construcción del Modelo")
    print("-" * 40)
    builder = ModelBuilder()
    model = builder.build_model()
    model = builder.compile_model(model)
    builder.model_summary(model)
    
    # Etapa 5: Entrenamiento
    print("\n🏋️‍♂️ ETAPA 5: Entrenamiento del Modelo")
    print("-" * 40)
    trainer = ModelTrainer()
    trained_model, history = trainer.train_model(model, train_gen, val_gen, class_weights)
    
    # Etapa 6: Evaluación
    print("\n🧪 ETAPA 6: Evaluación del Modelo")
    print("-" * 40)
    results = evaluate_complete_pipeline(trained_model, test_gen)
    
    print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE!")
    print(f"📊 Accuracy Final: {results['test_metrics']['accuracy']:.4f}")
    
    return trained_model, results

if __name__ == "__main__":
    trained_model, results = complete_pipeline()