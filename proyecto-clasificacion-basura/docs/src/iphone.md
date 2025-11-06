# Módulo: `iphone.py`

Interfaz web para usar la cámara del iPhone como guía visual (en Safari) mientras se clasifica con la cámara del PC en tiempo real.

## Clase: `iPhoneBrowserCamera`

- __init__(model_path=None, config_path="config/parameters.yaml")
  - Carga configuración, busca el `.h5` más reciente y define estado de predicción actual.

- preprocess_frame(frame), predict_frame(frame)
  - Igual que en `camera_classifier.py`.

- prediction_thread(frame)
  - Hilo para actualizar predicción asíncronamente.

- draw_modern_ui(frame, prediction, confidence, probs)
  - UI con header, categoría/contendor, barras de probabilidad y foco central.

- start_web_server(port=8000)
  - Crea `camera_server.html` y levanta un HTTP simple para servirlo.

- get_local_ip(), show_qr_code(url)
  - Utilidades para mostrar QR/URL y obtener IP local.

- setup_browser_camera() -> httpd
  - Muestra instrucciones y abre el servidor. El usuario debe abrir la URL/QR en el iPhone.

- capture_browser_frames() -> cv2.VideoCapture
  - Selecciona la cámara del PC disponible.

- process_video_stream(cap)
  - Bucle principal de visualización y predicción, una sola ventana.

- run_browser_classification()
  - Orquesta setup web + captura de cámara del PC + loop de UI.

### Uso

```powershell
python proyecto-clasificacion-basura\\src\\iphone.py
```
