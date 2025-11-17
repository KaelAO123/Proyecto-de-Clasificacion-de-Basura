/**
 * EcoClassify - Aplicación de Clasificación de Basura Mejorada
 * Características:
 * - Cuadro de enfoque para análisis específico
 * - Historial de clasificaciones
 * - Interfaz moderna y responsive
 * - Visualización avanzada de resultados
 * - Overlay móvil para resultados en tiempo real
 */

class EcoClassifyApp {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isProcessing = false;
        this.cameraActive = false;
        this.analysisExpanded = true;
        this.mobileResultsExpanded = false;
        
        // Métricas
        this.frameCount = 0;
        this.fps = 0;
        this.lastTimestamp = 0;
        this.lastPingTime = 0;
        this.latency = 0;
        
        // Historial
        this.classificationHistory = [];
        this.maxHistoryItems = 5;
        
        // Resultado actual
        this.currentResult = null;
        
        this.initializeApp();
    }
    
    initializeApp() {
        this.cacheElements();
        this.setupEventListeners();
        this.connectSocket();
        this.setupMetrics();
        this.loadHistory();
        this.updateDisplay();
        this.setupMobileDetection();
    }
    
    cacheElements() {
        this.elements = {
            // Estado
            serverStatus: document.getElementById('serverStatus'),
            modelStatus: document.getElementById('modelStatus'),
            
            // Cámara
            video: document.getElementById('video'),
            focusReticle: document.getElementById('focusReticle'),
            toggleCamera: document.getElementById('toggleCamera'),
            toggleProcessing: document.getElementById('toggleProcessing'),
            captureFrame: document.getElementById('captureFrame'),
            
            // Métricas
            fps: document.getElementById('fps'),
            latency: document.getElementById('latency'),
            
            // Resultados
            primaryResult: document.getElementById('primaryResult'),
            classIcon: document.getElementById('classIcon'),
            predictedClass: document.getElementById('predictedClass'),
            classDescription: document.getElementById('classDescription'),
            mainConfidence: document.getElementById('mainConfidence'),
            confidenceMeter: document.getElementById('confidenceMeter'),
            confidenceThumb: document.getElementById('confidenceThumb'),
            
            // Análisis
            analysisToggle: document.getElementById('analysisToggle'),
            probabilityGrid: document.getElementById('probabilityGrid'),
            
            // Historial
            clearHistory: document.getElementById('clearHistory'),
            historyItems: document.getElementById('historyItems'),
            
            // Nuevos elementos para overlay móvil
            overlayResult: document.getElementById('overlayResult'),
            overlayIcon: document.getElementById('overlayIcon'),
            overlayClass: document.getElementById('overlayClass'),
            overlayConfidence: document.getElementById('overlayConfidence'),
            
            // Panel móvil
            mobileResults: document.getElementById('mobileResults'),
            mobileToggle: document.getElementById('mobileToggle'),
            mobileResultContent: document.getElementById('mobileResultContent'),
            mobileIcon: document.getElementById('mobileIcon'),
            mobileClass: document.getElementById('mobileClass'),
            mobileDescription: document.getElementById('mobileDescription'),
            mobileConfidence: document.getElementById('mobileConfidence'),
            mobileMeterFill: document.getElementById('mobileMeterFill')
        };
    }
    
    setupEventListeners() {
        // Controles de cámara
        this.elements.toggleCamera.addEventListener('click', () => this.toggleCamera());
        this.elements.toggleProcessing.addEventListener('click', () => this.toggleProcessing());
        this.elements.captureFrame.addEventListener('click', () => this.captureFocusArea());
        
        // Análisis
        this.elements.analysisToggle.addEventListener('click', () => this.toggleAnalysis());
        
        // Historial
        this.elements.clearHistory.addEventListener('click', () => this.clearHistory());
        
        // Nuevos eventos para móvil
        this.elements.mobileToggle.addEventListener('click', () => this.toggleMobileResults());
        
        // Cerrar panel móvil al hacer clic fuera
        document.addEventListener('click', (e) => {
            if (this.mobileResultsExpanded && 
                !this.elements.mobileResults.contains(e.target) &&
                !this.elements.overlayResult.contains(e.target)) {
                this.collapseMobileResults();
            }
        });
        
        // Gestos táctiles para el retículo
        this.setupTouchGestures();
    }
    
    setupTouchGestures() {
        let startX, startY;
        const reticle = this.elements.focusReticle;
        
        reticle.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            startX = touch.clientX;
            startY = touch.clientY;
        });
        
        reticle.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const deltaX = touch.clientX - startX;
            const deltaY = touch.clientY - startY;
            
            // Mover retículo (limitado al área de video)
            const videoRect = this.elements.video.getBoundingClientRect();
            const newX = Math.max(0, Math.min(videoRect.width - 200, reticle.offsetLeft + deltaX));
            const newY = Math.max(0, Math.min(videoRect.height - 200, reticle.offsetTop + deltaY));
            
            reticle.style.left = newX + 'px';
            reticle.style.top = newY + 'px';
            
            startX = touch.clientX;
            startY = touch.clientY;
        });
    }
    
    setupMobileDetection() {
        // Detectar si es móvil y ajustar comportamientos
        this.isMobile = window.innerWidth <= 768;
        
        window.addEventListener('resize', () => {
            this.isMobile = window.innerWidth <= 768;
            this.adjustUIForMobile();
        });
        
        this.adjustUIForMobile();
    }
    
    adjustUIForMobile() {
        if (this.isMobile) {
            // Comportamientos específicos para móvil
            document.body.classList.add('mobile');
            this.collapseMobileResults(); // Iniciar con panel colapsado
        } else {
            document.body.classList.remove('mobile');
            this.expandMobileResults(); // Siempre expandido en desktop
        }
    }
    
    connectSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateConnectionStatus(true);
            this.isConnected = true;
        });
        
        this.socket.on('disconnect', () => {
            this.updateConnectionStatus(false);
            this.isConnected = false;
        });
        
        this.socket.on('connected', (data) => {
            this.updateModelStatus(data.model_ready);
            if (data.model_ready) {
                this.showNotification('Modelo IA cargado correctamente', 'success');
            }
        });
        
        this.socket.on('prediction', (data) => {
            this.handlePrediction(data);
        });
        
        // Medir latencia
        setInterval(() => {
            if (this.isConnected) {
                this.lastPingTime = Date.now();
                this.socket.emit('ping', { timestamp: this.lastPingTime });
            }
        }, 2000);
    }
    
    updateConnectionStatus(connected) {
        const statusElement = this.elements.serverStatus;
        statusElement.className = 'status-dot ' + (connected ? 'connected' : 'error');
    }
    
    updateModelStatus(ready) {
        const statusElement = this.elements.modelStatus;
        statusElement.className = 'status-dot ' + (ready ? 'connected' : 'error');
    }
    
    async toggleCamera() {
        if (this.cameraActive) {
            this.stopCamera();
        } else {
            await this.startCamera();
        }
    }
    
    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
            
            this.elements.video.srcObject = stream;
            this.cameraActive = true;
            
            // Actualizar UI
            this.elements.toggleCamera.innerHTML = '<i class="fas fa-stop"></i><span>Detener Cámara</span>';
            this.elements.toggleProcessing.disabled = false;
            this.elements.captureFrame.disabled = false;
            this.isProcessing = true;
            
            this.startFrameProcessing();
            this.showNotification('Cámara activada', 'success');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showNotification('Error al acceder a la cámara', 'error');
        }
    }
    
    stopCamera() {
        const stream = this.elements.video.srcObject;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            this.elements.video.srcObject = null;
        }
        
        this.cameraActive = false;
        this.isProcessing = false;
        
        // Actualizar UI
        this.elements.toggleCamera.innerHTML = '<i class="fas fa-camera"></i><span>Activar Cámara</span>';
        this.elements.toggleProcessing.disabled = true;
        this.elements.captureFrame.disabled = true;
        this.elements.toggleProcessing.innerHTML = '<i class="fas fa-pause"></i><span>Pausar</span>';
        
        this.hideResults();
        this.hideOverlay();
        this.showNotification('Cámara desactivada', 'info');
    }
    
    toggleProcessing() {
        this.isProcessing = !this.isProcessing;
        const icon = this.isProcessing ? 'pause' : 'play';
        const text = this.isProcessing ? 'Pausar' : 'Reanudar';
        
        this.elements.toggleProcessing.innerHTML = `<i class="fas fa-${icon}"></i><span>${text}</span>`;
        
        if (this.isProcessing) {
            this.showNotification('Análisis reanudado', 'success');
        } else {
            this.showNotification('Análisis pausado', 'warning');
        }
    }
    
    captureFocusArea() {
        if (!this.cameraActive) return;
        
        // Efecto visual de captura
        this.elements.focusReticle.style.transform = 'translate(-50%, -50%) scale(0.95)';
        setTimeout(() => {
            this.elements.focusReticle.style.transform = 'translate(-50%, -50%) scale(1)';
        }, 150);
        
        this.showNotification('Zona de enfoque capturada', 'info');
    }
    
    startFrameProcessing() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.lastFrameTime = 0;
        this.frameInterval = 200; // 5 FPS para balancear rendimiento y precisión
        
        this.processFrame();
    }
    
    processFrame() {
        if (!this.cameraActive || !this.isProcessing || !this.isConnected) {
            requestAnimationFrame(() => this.processFrame());
            return;
        }
        
        const now = Date.now();
        if (now - this.lastFrameTime < this.frameInterval) {
            requestAnimationFrame(() => this.processFrame());
            return;
        }
        
        this.lastFrameTime = now;
        
        try {
            // Capturar frame del área de enfoque
            this.captureFocusFrame();
            this.frameCount++;
            
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        
        requestAnimationFrame(() => this.processFrame());
    }
    
    captureFocusFrame() {
        const video = this.elements.video;
        const reticle = this.elements.focusReticle;
        
        // Calcular posición del retículo en coordenadas de video
        const videoRect = video.getBoundingClientRect();
        const reticleRect = reticle.getBoundingClientRect();
        
        const focusX = reticleRect.left - videoRect.left;
        const focusY = reticleRect.top - videoRect.top;
        const focusSize = 200; // Tamaño del retículo
        
        // Configurar canvas para el área de enfoque
        this.canvas.width = 224; // Tamaño esperado por el modelo
        this.canvas.height = 224;
        
        // Dibujar solo el área de enfoque
        this.ctx.drawImage(
            video,
            focusX, focusY, focusSize, focusSize, // Área fuente
            0, 0, 224, 224 // Área destino
        );
        
        // Convertir a JPEG y enviar
        this.canvas.toBlob(blob => {
            const reader = new FileReader();
            reader.onloadend = () => {
                if (this.socket && this.isConnected) {
                    this.socket.emit('frame', { 
                        image: reader.result,
                        timestamp: Date.now()
                    });
                }
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg', 0.8);
    }
    
    handlePrediction(data) {
        if (!data.success) {
            console.error('Prediction error:', data.error);
            this.showNotification('Error en la clasificación', 'error');
            return;
        }
        
        // Guardar resultado actual
        this.currentResult = data;
        
        // Actualizar resultados principales
        this.updatePrimaryResult(data);
                
        // Actualizar panel móvil
        this.updateMobileResult(data);
        
        // Actualizar análisis detallado
        this.updateProbabilityGrid(data.probabilities, data.class_name);
        
        // Agregar al historial
        this.addToHistory(data);
        
        // Mostrar sección de resultados
        this.showResults();
    }
    
    updatePrimaryResult(data) {
        const confidencePercent = Math.round(data.confidence * 100);
        
        // Actualizar clase y confianza
        this.elements.predictedClass.textContent = this.formatClassName(data.class_name);
        this.elements.mainConfidence.textContent = `${confidencePercent}%`;
        
        // Actualizar descripción e ícono
        this.updateClassDetails(data.class_name, confidencePercent);
        
        // Actualizar medidor de confianza
        this.elements.confidenceMeter.style.width = `${confidencePercent}%`;
        this.elements.confidenceThumb.style.left = `${confidencePercent}%`;
        
        // Efecto visual basado en confianza
        this.animateConfidence(confidencePercent);
    }
    
    // updateOverlayResult(data) {
    //     const confidencePercent = Math.round(data.confidence * 100);
    //     const className = this.formatClassName(data.class_name);
        
    //     // Actualizar contenido del overlay
    //     this.elements.overlayClass.textContent = className;
    //     this.elements.overlayConfidence.textContent = `${confidencePercent}%`;
        
    //     // Actualizar ícono
    //     const iconClass = this.getClassIcon(data.class_name);
    //     this.elements.overlayIcon.innerHTML = `<i class="fas fa-${iconClass}"></i>`;
        
    //     // Aplicar clase de tipo de basura
    //     this.elements.overlayIcon.className = `overlay-icon ${data.class_name}`;
        
    //     // Mostrar overlay con animación
    //     this.showOverlay();
        
    //     // Ocultar después de 3 segundos (solo si no hay nuevos resultados)
    //     setTimeout(() => {
    //         if (this.currentResult === data) {
    //             this.hideOverlay();
    //         }
    //     }, 3000);
    // }
    
    updateMobileResult(data) {
        const confidencePercent = Math.round(data.confidence * 100);
        const className = this.formatClassName(data.class_name);
        
        // Actualizar contenido móvil
        this.elements.mobileClass.textContent = className;
        this.elements.mobileConfidence.textContent = `${confidencePercent}%`;
        this.elements.mobileMeterFill.style.width = `${confidencePercent}%`;
        
        // Actualizar ícono y descripción
        const iconClass = this.getClassIcon(data.class_name);
        this.elements.mobileIcon.innerHTML = `<i class="fas fa-${iconClass}"></i>`;
        this.elements.mobileIcon.className = `mobile-icon ${data.class_name}`;
        
        const description = this.getClassDescription(data.class_name);
        this.elements.mobileDescription.textContent = description;
        
        // Expandir automáticamente en móvil cuando hay nuevo resultado
        if (this.isMobile && !this.mobileResultsExpanded) {
            this.expandMobileResults();
        }
    }
    
    formatClassName(className) {
        return className.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    updateClassDetails(className, confidence) {
        const description = this.getClassDescription(className);
        const iconClass = this.getClassIcon(className);
        
        this.elements.classDescription.textContent = description;
        this.elements.classIcon.innerHTML = `<i class="fas fa-${iconClass}"></i>`;
        
        // Color basado en confianza
        const colorClass = confidence >= 80 ? 'high' : confidence >= 60 ? 'medium' : 'low';
        this.elements.classIcon.className = `class-icon confidence-${colorClass}`;
    }
    
    getClassDescription(className) {
        const descriptions = {
            'organic': 'Material biodegradable como restos de comida, frutas, verduras, etc.',
            'plastic': 'Envases, botellas y otros productos de plástico',
            'paper': 'Periódicos, revistas, cartón y papel de oficina',
            'glass': 'Botellas, frascos y otros recipientes de vidrio',
            'metal': 'Latas, envases metálicos y objetos de aluminio'
        };
        
        return descriptions[className] || 'Material clasificado por IA';
    }
    
    getClassIcon(className) {
        const icons = {
            'organic': 'leaf',
            'plastic': 'wine-bottle',
            'paper': 'newspaper',
            'glass': 'glass-whiskey',
            'metal': 'cogs'
        };
        
        return icons[className] || 'question';
    }
    
    animateConfidence(confidence) {
        this.elements.primaryResult.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
        
        if (confidence >= 80) {
            this.elements.primaryResult.classList.add('confidence-high');
        } else if (confidence >= 60) {
            this.elements.primaryResult.classList.add('confidence-medium');
        } else {
            this.elements.primaryResult.classList.add('confidence-low');
        }
        
        // Efecto de pulso para alta confianza
        if (confidence >= 85) {
            this.elements.primaryResult.style.animation = 'pulse 2s ease-in-out';
            setTimeout(() => {
                this.elements.primaryResult.style.animation = '';
            }, 2000);
        }
    }
    
    updateProbabilityGrid(probabilities, topClass) {
        if (!probabilities) return;
        
        const sortedProbabilities = Object.entries(probabilities)
            .sort(([, a], [, b]) => b - a);
        
        let html = '';
        
        sortedProbabilities.forEach(([className, probability]) => {
            const percent = (probability * 100).toFixed(1);
            const isActive = className === topClass;
            
            html += `
                <div class="probability-item ${isActive ? 'active' : ''}">
                    <div class="probability-info">
                        <span class="probability-label">${this.formatClassName(className)}</span>
                        <span class="probability-value">${percent}%</span>
                    </div>
                    <div class="probability-bar-container">
                        <div class="probability-bar" style="width: ${percent}%"></div>
                    </div>
                </div>
            `;
        });
        
        this.elements.probabilityGrid.innerHTML = html;
    }
    
    addToHistory(data) {
        const historyItem = {
            class: data.class_name,
            confidence: data.confidence,
            timestamp: Date.now(),
            probabilities: data.probabilities
        };
        
        this.classificationHistory.unshift(historyItem);
        
        // Mantener solo los últimos elementos
        if (this.classificationHistory.length > this.maxHistoryItems) {
            this.classificationHistory.pop();
        }
        
        this.saveHistory();
        this.updateHistoryDisplay();
    }
    
    updateHistoryDisplay() {
        if (this.classificationHistory.length === 0) {
            this.elements.historyItems.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-inbox"></i>
                    <p>No hay clasificaciones recientes</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        this.classificationHistory.forEach((item, index) => {
            const timeAgo = this.getTimeAgo(item.timestamp);
            const confidencePercent = Math.round(item.confidence * 100);
            
            html += `
                <div class="history-item">
                    <div class="history-icon">
                        <i class="fas fa-${this.getClassIcon(item.class)}"></i>
                    </div>
                    <div class="history-content">
                        <div class="history-class">${this.formatClassName(item.class)}</div>
                        <div class="history-confidence">${confidencePercent}% • ${timeAgo}</div>
                    </div>
                </div>
            `;
        });
        
        this.elements.historyItems.innerHTML = html;
    }
    
    getTimeAgo(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / 60000);
        
        if (minutes < 1) return 'Ahora';
        if (minutes === 1) return 'Hace 1 min';
        if (minutes < 60) return `Hace ${minutes} min`;
        
        const hours = Math.floor(minutes / 60);
        if (hours === 1) return 'Hace 1 h';
        return `Hace ${hours} h`;
    }
    
    toggleAnalysis() {
        this.analysisExpanded = !this.analysisExpanded;
        const probabilityGrid = this.elements.probabilityGrid;
        const toggleIcon = this.elements.analysisToggle.querySelector('i');
        
        if (this.analysisExpanded) {
            probabilityGrid.style.display = 'grid';
            toggleIcon.className = 'fas fa-chevron-down';
            this.elements.analysisToggle.classList.add('active');
        } else {
            probabilityGrid.style.display = 'none';
            toggleIcon.className = 'fas fa-chevron-up';
            this.elements.analysisToggle.classList.remove('active');
        }
    }
    
    toggleMobileResults() {
        if (this.mobileResultsExpanded) {
            this.collapseMobileResults();
        } else {
            this.expandMobileResults();
        }
    }
    
    expandMobileResults() {
        this.elements.mobileResultContent.classList.add('expanded');
        this.elements.mobileToggle.querySelector('i').className = 'fas fa-chevron-down';
        this.mobileResultsExpanded = true;
    }
    
    collapseMobileResults() {
        this.elements.mobileResultContent.classList.remove('expanded');
        this.elements.mobileToggle.querySelector('i').className = 'fas fa-chevron-up';
        this.mobileResultsExpanded = false;
    }
    
    showOverlay() {
        this.elements.overlayResult.classList.add('show');
    }
    
    hideOverlay() {
        this.elements.overlayResult.classList.remove('show');
    }
    
    clearHistory() {
        this.classificationHistory = [];
        this.saveHistory();
        this.updateHistoryDisplay();
        this.showNotification('Historial limpiado', 'info');
    }
    
    saveHistory() {
        localStorage.setItem('classificationHistory', JSON.stringify(this.classificationHistory));
    }
    
    loadHistory() {
        const saved = localStorage.getItem('classificationHistory');
        if (saved) {
            this.classificationHistory = JSON.parse(saved);
            this.updateHistoryDisplay();
        }
    }
    
    setupMetrics() {
        // Contador FPS
        setInterval(() => {
            const now = performance.now();
            if (this.lastTimestamp > 0) {
                const delta = (now - this.lastTimestamp) / 1000;
                this.fps = Math.round(this.frameCount / delta);
                this.elements.fps.textContent = this.fps;
                this.frameCount = 0;
            }
            this.lastTimestamp = now;
        }, 1000);
        
        // Medidor de latencia
        this.socket.on('pong', (data) => {
            this.latency = Date.now() - data.timestamp;
            this.elements.latency.textContent = this.latency;
        });
    }
    
    showResults() {
        this.elements.primaryResult.classList.remove('hidden');
        this.elements.primaryResult.classList.add('fade-in');
    }
    
    hideResults() {
        this.elements.primaryResult.classList.add('hidden');
    }
    
    showNotification(message, type = 'info') {
        // Implementación básica de notificación
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // En una implementación real, aquí agregarías un sistema de notificaciones toast
        if (type === 'error') {
            alert(`Error: ${message}`);
        }
    }
    
    updateDisplay() {
        // Actualizaciones periódicas de UI
        requestAnimationFrame(() => this.updateDisplay());
    }
}

// Inicializar la aplicación cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    new EcoClassifyApp();
});

// Registrar Service Worker para funcionalidad offline (opcional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}