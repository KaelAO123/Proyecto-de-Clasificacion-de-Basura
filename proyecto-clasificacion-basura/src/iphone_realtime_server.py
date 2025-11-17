"""
Servidor Flask + Socket.IO mejorado para clasificación de basura en tiempo real desde iPhone

Características:
- Interfaz moderna y responsive
- Código modular y mantenible
- Manejo de errores robusto
- Estructura profesional para el proyecto
"""

import os
import sys
from pathlib import Path

# Añadir el directorio padre al path para imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from iphone_app.app import create_app, socketio

def main():
    """Función principal para ejecutar el servidor"""
    app = create_app()
    
    # Obtener configuración
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    
    print('Iniciando Servidor de Clasificación de Basura')
    print('Optimizado para iPhone')
    print('Usa Safari en tu iPhone para conectarte')
    
    # Mostrar IP local para acceso fácil
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f'\nURLs de acceso:')
        print(f'   Local: http://{local_ip}:{port}')
        print(f'   Red: http://{host}:{port}')
        print(f'\n Abre la URL en Safari desde tu iPhone')
        print('   (Asegúrate de estar en la misma red Wi-Fi)\n')
    except:
        print(f'\nServidor ejecutándose en: http://{host}:{port}\n')
    
    # Ejecutar servidor
    socketio.run(
        app, 
        host=host, 
        port=port, 
        debug=debug,
        allow_unsafe_werkzeug=True,
    )

if __name__ == '__main__':
    main()