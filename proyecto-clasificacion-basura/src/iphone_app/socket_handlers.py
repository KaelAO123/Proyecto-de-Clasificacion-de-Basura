"""
Manejadores de eventos Socket.IO
"""

from flask import request
from flask_socketio import emit
import base64
import io
from PIL import Image
import numpy as np

from .app import socketio, get_classifier

@socketio.on('connect')
def handle_connect():
    """Manejar conexión de cliente"""
    client_ip = request.remote_addr
    print(f'Cliente conectado: {client_ip}')
    
    classifier = get_classifier()
    if classifier and classifier.class_names:
        emit('connected', {
            'status': 'success',
            'message': 'Conectado al servidor de clasificación',
            'classes': classifier.class_names,
            'model_ready': True
        })
    else:
        emit('connected', {
            'status': 'warning',
            'message': 'Servidor conectado pero clasificador no disponible',
            'model_ready': False
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Manejar desconexión de cliente"""
    print(f'Cliente desconectado: {request.remote_addr}')

@socketio.on('frame')
def handle_frame(data):
    """Procesar frame recibido desde el cliente"""
    classifier = get_classifier()
    if classifier is None:
        emit('prediction', {
            'error': 'Clasificador no disponible',
            'class_name': 'ERROR',
            'confidence': 0.0
        })
        return

    img_data = data.get('image', '')
    if not img_data:
        emit('prediction', {
            'error': 'No se recibió imagen',
            'class_name': 'ERROR', 
            'confidence': 0.0
        })
        return

    try:
        # Decodificar base64
        if ',' in img_data:
            b64_data = img_data.split(',', 1)[1]
        else:
            b64_data = img_data
            
        image_bytes = base64.b64decode(b64_data)
        
        # Realizar predicción
        result = classifier.predict_bytes(image_bytes)
        emit('prediction', result)
        
    except Exception as e:
        print(f'Error procesando frame: {e}')
        emit('prediction', {
            'error': str(e),
            'class_name': 'ERROR',
            'confidence': 0.0
        })

@socketio.on('ping')
def handle_ping(data):
    """Manejar ping para medir latencia"""
    emit('pong', {'timestamp': data.get('timestamp')})