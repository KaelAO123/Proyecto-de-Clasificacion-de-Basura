"""
Paquete de la aplicación iPhone para clasificación de basura en tiempo real
"""

from .app import create_app, socketio
from .classifier import RealtimeClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


__version__ = '1.0.0'
__all__ = ['create_app', 'socketio', 'RealtimeClassifier']