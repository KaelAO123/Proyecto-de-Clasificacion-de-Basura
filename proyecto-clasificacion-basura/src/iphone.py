# src/iphone_browser_camera_fixed.py
import cv2
import tensorflow as tf
import numpy as np
import time
import threading
import http.server
import socketserver
from pathlib import Path
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class iPhoneBrowserCamera:
    def __init__(self, model_path=None, config_path="config/parameters.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        if model_path is None:
            models_dir = Path("models/trained_models")
            model_files = list(models_dir.glob("**/*.h5"))
            if not model_files:
                raise FileNotFoundError("No se encontraron modelos entrenados")
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        print(f"📦 Cargando modelo: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        print("✅ Modelo cargado exitosamente")
        
        self.img_size = tuple(self.config['data']['image_size'])
        self.class_names = self.config['data']['classes']
        
        # Variables de estado
        self.current_prediction = "trash"
        self.current_confidence = 0.0
        self.current_probs = np.zeros(len(self.class_names))
        
        # Información de reciclaje
        self.recycling_info = {
            "cardboard": {"category": "🟦 RECICLABLE", "container": "AZUL", "icon": "📦"},
            "glass": {"category": "🟩 RECICLABLE", "container": "VERDE", "icon": "🍶"},
            "metal": {"category": "🟨 RECICLABLE", "container": "AMARILLO", "icon": "🥫"},
            "paper": {"category": "🟦 RECICLABLE", "container": "AZUL", "icon": "📄"},
            "plastic": {"category": "🟨 RECICLABLE", "container": "AMARILLO", "icon": "🧴"},
            "trash": {"category": "⚫ NO RECICLABLE", "container": "GRIS", "icon": "🗑️"}
        }
        
        self.colors = {
            "cardboard": (255, 0, 0),      # Azul
            "glass": (0, 255, 0),          # Verde
            "metal": (0, 255, 255),        # Amarillo
            "paper": (0, 0, 255),          # Rojo
            "plastic": (0, 165, 255),      # Naranja
            "trash": (128, 128, 128)       # Gris
        }

    def preprocess_frame(self, frame):
        """Preprocesa el frame para predicción"""
        img = cv2.resize(frame, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict_frame(self, frame):
        """Realiza predicción en el frame"""
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        return predicted_class, confidence, predictions[0]

    def prediction_thread(self, frame):
        """Hilo para predicciones asíncronas"""
        try:
            pred_class, confidence, probs = self.predict_frame(frame)
            self.current_prediction = pred_class
            self.current_confidence = confidence
            self.current_probs = probs
        except Exception as e:
            print(f"⚠️ Error en predicción: {e}")

    def draw_modern_ui(self, frame, prediction, confidence, probs):
        """Interfaz moderna para el clasificador"""
        height, width = frame.shape[:2]
        info = self.recycling_info[prediction]
        color = self.colors[prediction]
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
        
        # Texto principal
        main_text = f"{info['icon']} {prediction.upper()}"
        cv2.putText(frame, main_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Confianza
        confidence_text = f"Confianza: {confidence:.1%}"
        cv2.putText(frame, confidence_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Información de reciclaje
        recycle_y = height - 150
        cv2.rectangle(frame, (0, recycle_y), (width, height), (0, 0, 0), -1)
        
        category_text = f"CATEGORÍA: {info['category']}"
        container_text = f"CONTENEDOR: {info['container']}"
        
        cv2.putText(frame, category_text, (20, recycle_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, container_text, (20, recycle_y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Barra de probabilidades
        bar_height = 25
        bar_width = width // len(self.class_names)
        bars_y = recycle_y - bar_height - 10
        
        for i, class_name in enumerate(self.class_names):
            prob = probs[i]
            bar_length = int(prob * bar_width)
            
            x_start = i * bar_width
            x_end = x_start + bar_length
            
            cv2.rectangle(frame, (x_start, bars_y), (x_end, bars_y + bar_height), 
                         self.colors[class_name], -1)
            cv2.rectangle(frame, (x_start, bars_y), (x_start + bar_width, bars_y + bar_height), 
                         (255, 255, 255), 1)
            
            if bar_width > 40:
                label = class_name[:3].upper()
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                text_x = x_start + (bar_width - text_size[0]) // 2
                text_y = bars_y + bar_height // 2 + 5
                cv2.putText(frame, label, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Rectángulo de enfoque
        focus_size = min(width, height) // 3
        center_x, center_y = width // 2, height // 2 - 50
        
        cv2.rectangle(frame, 
                     (center_x - focus_size // 2, center_y - focus_size // 2),
                     (center_x + focus_size // 2, center_y + focus_size // 2),
                     color, 3)
        
        # Puntos de guía
        guide_size = 20
        cv2.line(frame, (center_x - focus_size//2, center_y), 
                (center_x - focus_size//2 + guide_size, center_y), color, 3)
        cv2.line(frame, (center_x + focus_size//2, center_y), 
                (center_x + focus_size//2 - guide_size, center_y), color, 3)
        cv2.line(frame, (center_x, center_y - focus_size//2), 
                (center_x, center_y - focus_size//2 + guide_size), color, 3)
        cv2.line(frame, (center_x, center_y + focus_size//2), 
                (center_x, center_y + focus_size//2 - guide_size), color, 3)
        
        instruction = "APUNTA EL OBJETO AL CENTRO"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        cv2.putText(frame, instruction, (text_x, center_y - focus_size//2 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

    def start_web_server(self, port=8000):
        """Inicia un servidor web local para transmitir video"""
        print("🌐 Iniciando servidor web...")
        
        # Crear página HTML simple
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ECO AI - Cámara iPhone</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { 
                    margin: 0; 
                    padding: 20px; 
                    background: #1a1a1a; 
                    color: white;
                    font-family: Arial, sans-serif;
                    text-align: center;
                }
                .container { 
                    max-width: 100%; 
                    margin: 0 auto; 
                }
                video { 
                    width: 100%; 
                    max-width: 640px; 
                    border: 3px solid #00ff88; 
                    border-radius: 10px; 
                }
                .instructions { 
                    background: #2a2a2a; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin: 20px 0; 
                }
                button { 
                    background: #00ff88; 
                    color: black; 
                    border: none; 
                    padding: 15px 30px; 
                    font-size: 18px; 
                    border-radius: 25px; 
                    margin: 10px; 
                    cursor: pointer; 
                }
                .status { 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                }
                .connected { background: #00ff88; color: black; }
                .disconnected { background: #ff4444; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 ECO AI - Clasificador de Basura</h1>
                
                <div class="instructions">
                    <h3>📱 Instrucciones:</h3>
                    <p>1. Permite el acceso a la cámara cuando Safari lo pida</p>
                    <p>2. Apunta los objetos de basura a la cámara</p>
                    <p>3. Mira los resultados en tu computadora</p>
                    <p><strong>💡 Consejo:</strong> Usa la cámara de tu computadora para clasificar</p>
                </div>
                
                <button onclick="startCamera()">🎥 Activar Cámara iPhone</button>
                <button onclick="stopCamera()">⏹️ Detener Cámara</button>
                
                <div id="status" class="status disconnected">
                    Cámara iPhone: Desconectada
                </div>
                
                <video id="video" autoplay playsinline></video>
                
                <div style="margin-top: 20px;">
                    <p><strong>Modo Simulación:</strong> Esta pantalla es solo para guía visual</p>
                    <p>La clasificación real se hace con la cámara de tu computadora</p>
                </div>
            </div>

            <script>
                let stream = null;
                const video = document.getElementById('video');
                const status = document.getElementById('status');

                function updateStatus(message, isConnected) {
                    status.textContent = message;
                    status.className = isConnected ? 'status connected' : 'status disconnected';
                }

                async function startCamera() {
                    try {
                        // Solicitar acceso a la cámara
                        stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { 
                                width: { ideal: 1280 },
                                height: { ideal: 720 },
                                facingMode: 'environment' 
                            }, 
                            audio: false 
                        });
                        
                        video.srcObject = stream;
                        updateStatus('Cámara iPhone: Conectada ✅', true);
                        
                    } catch (err) {
                        updateStatus('Error: ' + err.message, false);
                        console.error('Error accessing camera:', err);
                    }
                }

                function stopCamera() {
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                        video.srcObject = null;
                        updateStatus('Cámara iPhone: Desconectada', false);
                    }
                }

                // Iniciar cámara automáticamente
                window.addEventListener('load', startCamera);
            </script>
        </body>
        </html>
        """
        
        # Guardar HTML temporal
        with open("camera_server.html", "w") as f:
            f.write(html_content)
        
        # Iniciar servidor HTTP simple
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", port), handler)
        
        # Ejecutar servidor en segundo plano
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        print(f"✅ Servidor web iniciado en puerto: {port}")
        return httpd

    def get_local_ip(self):
        """Obtiene la IP local para acceso desde iPhone"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "localhost"

    def show_qr_code(self, url):
        """Muestra código QR para fácil acceso desde iPhone"""
        try:
            import qrcode
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(url)
            qr.make(fit=True)
            
            print("\n📲 Escanea este código QR con tu iPhone:")
            qr.print_ascii()
            print(f"🔗 O ve a: {url}")
            
        except ImportError:
            print(f"🔗 Ve a esta URL en tu iPhone: {url}")

    def setup_browser_camera(self):
        """Configura la cámara del navegador"""
        print("\n" + "="*60)
        print("📱 CÁMARA DEL IPHONE VÍA SAFARI")
        print("="*60)
        
        local_ip = self.get_local_ip()
        port = 8000
        url = f"http://{local_ip}:{port}/camera_server.html"
        
        print("\n🎯 INSTRUCCIONES:")
        print("1. ✅ Asegúrate de que iPhone y computadora estén en la MISMA red WiFi")
        print("2. 📱 En tu iPhone, abre Safari")
        print("3. 🌐 Ve a la siguiente dirección:")
        
        self.show_qr_code(url)
        
        print("\n4. 📷 Safari pedirá permiso para la cámara → Permite")
        print("5. 🎥 La cámara se activará automáticamente")
        print("6. 💻 Regresa a esta ventana para ver las clasificaciones")
        print("\n🔮 MODO SIMULACIÓN:")
        print("   • Safari muestra la cámara del iPhone como GUÍA")
        print("   • Usamos cámara de computadora para CLASIFICACIÓN REAL")
        print("   • Apunta los mismos objetos a ambas cámaras")
        
        input("\n✅ Presiona Enter cuando Safari esté listo...")
        
        return self.start_web_server(port)

    def capture_browser_frames(self):
        """Captura frames usando la cámara de la computadora"""
        print("🎥 Activando cámara de computadora para clasificación...")
        
        # Probar diferentes índices de cámara
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Configurar resolución
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Verificar que funcione
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Cámara {camera_index} lista: {frame.shape[1]}x{frame.shape[0]}")
                    return cap
                else:
                    cap.release()
        
        print("❌ No se pudo abrir ninguna cámara de computadora")
        return None

    def process_video_stream(self, cap):
        """Procesa el stream de video - UNA SOLA VENTANA"""
        frame_count = 0
        last_prediction_time = time.time()
        last_fps_time = time.time()
        fps = 0
        
        print("\n🎮 CONTROLES:")
        print("   Q - Salir")
        print("   S - Guardar captura")
        print("   R - Rotar vista")
        
        # Crear UNA sola ventana
        window_name = 'ECO AI - Clasificador de Basura (Cámara Computadora)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error leyendo frame de la cámara")
                    break
                
                # Espejo para mejor UX (como un espejo real)
                frame = cv2.flip(frame, 1)
                
                # Predicción cada 0.5 segundos para mejor performance
                current_time = time.time()
                if current_time - last_prediction_time > 0.5:
                    # Usar hilo para no bloquear el video
                    prediction_thread = threading.Thread(
                        target=self.prediction_thread, 
                        args=(frame.copy(),)
                    )
                    prediction_thread.daemon = True
                    prediction_thread.start()
                    last_prediction_time = current_time
                
                # Dibujar interfaz
                frame = self.draw_modern_ui(
                    frame, 
                    self.current_prediction, 
                    self.current_confidence, 
                    self.current_probs
                )
                
                # Calcular y mostrar FPS
                frame_count += 1
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_fps_time = current_time
                
                fps_text = f"FPS: {fps}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Mostrar advertencia de modo simulación
                sim_text = "MODO SIMULACIÓN: Usando cámara computadora"
                cv2.putText(frame, sim_text, (10, frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Mostrar frame en UNA sola ventana
                cv2.imshow(window_name, frame)
                
                # Controles - usar waitKey correctamente
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' o ESC
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"capture_{timestamp}_{self.current_prediction}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Captura guardada: {filename}")
                elif key == ord('r'):
                    print("🔄 Vista rotada (simulación)")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Deteniendo...")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            # Cerrar solo la ventana que creamos
            if 'window_name' in locals():
                cv2.destroyWindow(window_name)
            cv2.destroyAllWindows()

    def run_browser_classification(self):
        """Ejecuta clasificación con cámara del navegador"""
        print("🎯 CLASIFICADOR CON SAFARI DEL IPHONE + CÁMARA COMPUTADORA")
        print("=" * 60)
        
        # Paso 1: Configurar servidor web para iPhone
        httpd = self.setup_browser_camera()
        
        # Paso 2: Usar cámara local como fuente de video real
        cap = self.capture_browser_frames()
        if cap is None:
            print("❌ No se pudo obtener fuente de video")
            if httpd:
                httpd.shutdown()
            return
        
        # Paso 3: Procesar video de la computadora
        self.process_video_stream(cap)
        
        # Limpiar
        cap.release()
        if httpd:
            httpd.shutdown()
        
        print("✅ Clasificación terminada")

def main():
    try:
        classifier = iPhoneBrowserCamera()
        classifier.run_browser_classification()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Soluciones:")
        print("1. Verifica que tengas cámara en la computadora")
        print("2. Ejecuta: pip install opencv-python")
        print("3. Asegúrate de tener un modelo entrenado")

if __name__ == "__main__":
    main()