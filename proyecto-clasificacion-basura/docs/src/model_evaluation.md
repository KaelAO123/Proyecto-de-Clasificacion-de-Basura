# Módulo: `model_evaluation.py`

Evaluación del modelo en el test set, generación de métricas y visualizaciones.

## Clase: `ModelEvaluator`

- __init__(model, test_generator, config_path="config/parameters.yaml")
  - Crea `results_dir = reports/evaluation` y define `class_names`.

- evaluate_model() -> (y_true, y_pred, predictions)
  - Ejecuta `model.evaluate` e imprime métricas básicas (loss, accuracy, precision, recall, AUC).
  - Calcula predicciones y etiquetas verdaderas.

- plot_confusion_matrix(y_true, y_pred) -> np.ndarray
  - Guarda `confusion_matrix.png`.

- classification_report_df(y_true, y_pred) -> pd.DataFrame
  - Genera `classification_report.csv` y `classification_report.md`.

- plot_class_distribution(y_true, y_pred)
  - Gráfico de distribución real vs. predicha y accuracy por clase (`class_distribution.png`).

- analyze_misclassifications(y_true, y_pred, test_generator)
  - Toma muestras de errores y llama a `plot_misclassifications(...)`.

- save_evaluation_results(y_true, y_pred, test_loss, test_accuracy)
  - Guarda `evaluation_results.json` con resumen y métricas por clase.

## Función: `evaluate_complete_pipeline(model, test_generator)`

Orquesta la evaluación completa y devuelve un diccionario con y_true/y_pred, probabilidades, matriz de confusión, reporte y métricas de test.
