# Reportes y Resultados de Evaluación

Los artefactos de evaluación se guardan en `reports/evaluation/`.

## Archivos generados

- `classification_report.csv` y `classification_report.md`: métricas por clase (precision, recall, f1-score, support).
- `evaluation_results.json`: resumen de métricas globales (loss, accuracy) y por clase, además de la matriz de confusión.
- `confusion_matrix.png`: heatmap con la matriz de confusión.
- `class_distribution.png`: comparación de distribución real vs. predicha y accuracy por clase.
- `misclassifications.png`: ejemplos visuales de errores (si hay suficientes casos).

Generación: se ejecutan desde `src/model_evaluation.py` al evaluar con un modelo y `test_generator`.
