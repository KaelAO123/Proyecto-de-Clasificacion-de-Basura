"""
Módulo principal de la aplicación Flask
"""

from flask import Flask, render_template
from flask_socketio import SocketIO
import os
import sys

from .classifier import RealtimeClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración
socketio = SocketIO(cors_allowed_origins="*", async_mode='eventlet')

def create_app():
    """Factory function para crear la aplicación Flask"""
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')
    
    # Configuración
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'clasificacion-basura-secret-key')
    
    # Inicializar extensiones
    socketio.init_app(app)

    global classifier
    try:
        classifier = RealtimeClassifier()
        print('Clasificador cargado correctamente')
    except Exception as e:
        print(f'Error cargando el clasificador: {e}')
        classifier = None
    
    # Rutas
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/health')
    def health():
        """Endpoint de salud para verificar que el servidor funciona"""
        return {
            'status': 'healthy',
            'classifier_loaded': classifier is not None,
            'service': 'waste-classification-server'
        }
    
    # Handlers de Socket.IO
    from . import socket_handlers  # Esto registrará los event handlers
    
    return app

# Variable global del clasificador
classifier = None

def get_classifier():
    """Obtener la instancia del clasificador"""
    return classifier