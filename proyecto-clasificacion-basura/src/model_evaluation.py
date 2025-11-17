# src/model_evaluation.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path
import yaml
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelEvaluator:
    def __init__(self, model, test_generator, config_path="config/parameters.yaml"):
        self.model = model
        self.test_generator = test_generator
        self.config = yaml.safe_load(open(config_path, 'r'))
        
        self.class_names = list(test_generator.class_indices.keys())
        self.results_dir = Path("reports/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self):
        """Evalúa el modelo en el conjunto de test"""
        
        print("EVALUANDO MODELO EN TEST SET")
        print("=" * 50)
        
        # Evaluación básica
        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            self.test_generator, verbose=1
        )
        
        print(f"\nResultados de Evaluación:")
        print(f"   - Loss: {test_loss:.4f}")
        print(f"   - Accuracy: {test_accuracy:.4f}")
        print(f"   - Precision: {test_precision:.4f}")
        print(f"   - Recall: {test_recall:.4f}")
        print(f"   - AUC: {test_auc:.4f}")
        
        # Predicciones
        print("\nGenerando predicciones...")
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        return y_true, y_pred, predictions
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Genera y guarda matriz de confusión"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(str(self.results_dir / "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def classification_report_df(self, y_true, y_pred):
        """Genera reporte de clasificación detallado"""
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convertir a DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Guardar reporte
        report_df.to_csv(self.results_dir / "classification_report.csv")
        report_df.to_markdown(self.results_dir / "classification_report.md")
        
        print("\nReporte de Clasificación:")
        print(report_df.round(4))
        
        return report_df
    
    def plot_class_distribution(self, y_true, y_pred):
        """Visualiza distribución de clases y aciertos"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distribución real vs predicciones
        real_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        ax1.bar(x - width/2, real_counts, width, label='Real', alpha=0.7)
        ax1.bar(x + width/2, pred_counts, width, label='Predicho', alpha=0.7)
        
        ax1.set_xlabel('Clases')
        ax1.set_ylabel('Cantidad')
        ax1.set_title('Distribución Real vs Predicha')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.legend()
        
        # Porcentaje de aciertos por clase
        correct_by_class = []
        for class_idx in range(len(self.class_names)):
            mask = y_true == class_idx
            correct = np.sum(y_true[mask] == y_pred[mask])
            total = np.sum(mask)
            accuracy = correct / total if total > 0 else 0
            correct_by_class.append(accuracy)
        
        ax2.bar(self.class_names, correct_by_class, color='green', alpha=0.7)
        ax2.set_xlabel('Clases')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy por Clase')
        ax2.set_ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(correct_by_class):
            ax2.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(str(self.results_dir / "class_distribution.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, y_true, y_pred, test_generator):
        """Analiza ejemplos de misclasificación"""
        
        misclassified_idx = np.where(y_true != y_pred)[0]
        
        if len(misclassified_idx) > 0:
            print(f"\nEjemplos de misclasificaciones ({len(misclassified_idx)} casos):")
            
            # Tomar algunos ejemplos aleatorios de los errores
            sample_idx = np.random.choice(
                misclassified_idx, 
                size=min(5, len(misclassified_idx)), 
                replace=False
            )

            # Obtener rutas de las imágenes correspondientes
            filepaths = np.array(test_generator.filepaths)
            
            # Cargar las imágenes reales desde disco según sus índices globales
            images = []
            true_labels = []
            for idx in sample_idx:
                img_path = filepaths[idx]
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=test_generator.image_shape[:2])
                img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img)
                true_labels.append(y_true[idx])

            images = np.array(images)
            true_labels = np.array(true_labels)
            
            # Visualizar misclasificaciones
            self.plot_misclassifications(images, true_labels, y_pred, sample_idx)

    
    def plot_misclassifications(self, images, true_labels, y_pred, sample_idx):
        """Visualiza ejemplos de misclasificación"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('off')

            true_class = self.class_names[true_labels[i]]
            pred_class = self.class_names[y_pred[sample_idx[i]]] 

            axes[i].set_title(f'Real: {true_class}\nPred: {pred_class}', color='red', fontsize=10)

        # Ocultar ejes vacíos
        for i in range(len(images), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(str(self.results_dir / "misclassifications.png"), dpi=300, bbox_inches='tight')
        plt.show()

    
    def save_evaluation_results(self, y_true, y_pred, test_loss, test_accuracy):
        """Guarda resultados completos de evaluación"""
        
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'per_class_accuracy': {},
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'model_performance': 'Excellent' if test_accuracy > 0.9 else 
                               'Good' if test_accuracy > 0.8 else 
                               'Fair' if test_accuracy > 0.7 else 'Needs Improvement'
        }
        
        # Accuracy por clase
        for class_idx, class_name in enumerate(self.class_names):
            mask = y_true == class_idx
            if np.sum(mask) > 0:
                class_accuracy = np.sum(y_true[mask] == y_pred[mask]) / np.sum(mask)
                evaluation_results['per_class_accuracy'][class_name] = float(class_accuracy)
        
        with open(self.results_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print(f"\nResultados guardados en: {self.results_dir}")

def evaluate_complete_pipeline(model, test_generator):
    """Ejecuta evaluación completa del pipeline"""
    
    evaluator = ModelEvaluator(model, test_generator)
    
    # Evaluación básica
    y_true, y_pred, predictions = evaluator.evaluate_model()
    
    # Métricas detalladas
    cm = evaluator.plot_confusion_matrix(y_true, y_pred)
    report_df = evaluator.classification_report_df(y_true, y_pred)
    evaluator.plot_class_distribution(y_true, y_pred)
    evaluator.analyze_misclassifications(y_true, y_pred, test_generator)
    
    # Obtener métricas de evaluación
    test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
        test_generator, verbose=0
    )
    
    evaluator.save_evaluation_results(y_true, y_pred, test_loss, test_accuracy)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'predictions': predictions,
        'confusion_matrix': cm,
        'classification_report': report_df,
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc
        }
    }

# if __name__ == "__main__":
#     from src.model_training import main as train_main
    
#     # Entrenar y evaluar
#     trained_model, history, test_gen = train_main()
#     results = evaluate_complete_pipeline(trained_model, test_gen)

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    from src.data_loader import create_test_generator  
    import yaml

    # Cargar configuración
    config = yaml.safe_load(open("config/parameters.yaml", "r"))
    model_path = config.get("model_path", "models/trained_models/20251105_094221/best_model.h5")
    print(f"Cargando modelo entrenado desde: {model_path}")
    model = load_model(model_path)
    print("Cargando conjunto de test...")
    test_generator = create_test_generator(config)

    # Evaluar el modelo cargado
    results = evaluate_complete_pipeline(model, test_generator)