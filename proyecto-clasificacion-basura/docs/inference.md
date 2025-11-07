# Inferencia y Uso con Cámara

El proyecto incluye dos interfaces para inferencia en tiempo real:

1) `src/camera_classifier.py`: Usa la cámara del equipo con overlay de clase y consejos de reciclaje.
2) `src/iphone.py`: Abre una página web para guiarte con la cámara del iPhone (visual) y usa la cámara del PC para clasificar.

## Requisitos previos

- Tener un modelo entrenado en `models/trained_models/**/best_model.h5`.
- Verifica `config/parameters.yaml` (tamaño de imagen y clases).

## Cámara del equipo

```powershell
python proyecto-clasificacion-basura\src\camera_classifier.py
```

Controles:

- Q: Salir
- S: Guardar frame actual
- R: Reiniciar estadísticas

El script buscará el modelo `.h5` más reciente y lo cargará automáticamente.

## iPhone (Safari) + cámara del PC

```powershell
python proyecto-clasificacion-basura\src\iphone.py
```

Flujo:

1. Levanta un servidor HTTP local y genera `camera_server.html`.
2. Muestra instrucciones y un QR/URL para abrir en Safari (misma red WiFi).
3. Safari solo sirve como guía visual; la clasificación real se realiza con la cámara del PC.
4. La ventana muestra predicción, confianza y barras por clase.
