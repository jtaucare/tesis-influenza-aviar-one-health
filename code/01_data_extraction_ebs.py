#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autor: Jorge Eduardo Taucare Miranda
Programa: Doctorado en Ingeniería y Ciencias con la Industria (DICI)
Proyecto: Sistema de Vigilancia y Alerta Temprana para Influenza Aviar (Una Salud)

Descripción del flujo de trabajo:
    Este script simula el pipeline de Visión por Computadora y Edge-AI para monitoreo
    automatizado (Vigilancia EBS Avanzada). Integra modelos YOLO para detección, 
    conteo y clasificación taxonómica de aves, junto con análisis de comportamiento 
    neurológico y extracción de matrices termográficas para detectar estados febriles.

Datos de Entrada (Inputs):
    - Flujo de video RTSP de cámaras fijas o drones (simulado).
    - Matriz de pixeles térmicos e infrarrojos en zonas de riesgo.

Datos de Salida (Outputs):
    - Regresiones de conteo y alertas automáticas en data_processed/
    - Métricas visuales enviadas al prototipo de Gemelo Digital.

Dependencias de Librerías (Declaradas en environment.yml):
    - opencv-python (cv2), numpy, pandas, ultralytics (YOLOv8/v11)
"""

import os
import time
import datetime
import numpy as np
import pandas as pd

# Configuración de entornos y rutas relativas para reproducibilidad
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT_DIR = os.path.join(BASE_DIR, "data_processed")

class AvianIntelligentMonitor:
    def __init__(self, camera_id="CAM_HUMEDAL_01", comuna_cut=13111):
        self.camera_id = camera_id
        self.comuna_cut = comuna_cut
        print(f"[{self.camera_id}] Inicializando módulos de IA bajo enfoque Una Salud...")
        print(f"[{self.camera_id}] Cargando pesos neuronales optimizados (YOLOv8-Avian)...")
        time.sleep(1) # Simulación de carga de modelos en memoria GPU

    def process_frame_pipeline(self):
        """
        Simula el procesamiento secuencial de un frame de video multispectral
        (Óptico + Térmico) capturado en un punto crítico de monitoreo.
        """
        # 1. DETECCIÓN Y CONTEO (Simulación de salida de Bounding Boxes de YOLO)
        conteo_aves_silvestres = np.random.randint(15, 30)
        conteo_aves_traspatio = np.random.randint(0, 3)
        total_detectado = conteo_aves_silvestres + conteo_aves_traspatio
        
        # 2. ANÁLISIS DE COMPORTAMIENTO (Detección de patrones cinemáticos anómalos)
        # Simula clasificador de esqueleto/pose para detectar giros de cabeza o postración
        probabilidad_signos_neurologicos = round(np.random.uniform(0.02, 0.85), 2)
        signos_detectados = 1 if probabilidad_signos_neurologicos > 0.70 else 0

        # 3. EXTRACCIÓN DATOS TERMOGRÁFICOS (Regiones de Interés: cabeza y patas)
        # Simula la lectura de la matriz térmica calibrada acoplada a la cámara
        temperatura_max_detectada = round(np.random.uniform(38.5, 43.2), 1) 
        # En aves, temperaturas > 42.5°C pueden correlacionarse con alertas de fiebre por H5N1
        anomalia_termica = 1 if temperatura_max_detectada >= 42.5 else 0

        # 4. CALCULO DEL SCORE DE CONFIANZA DEL SISTEMA INTELIGENTE
        # Combina la precisión del modelo óptico con la severidad térmica/conductual
        score_confianza = round((0.4 * 0.95) + (0.3 * signos_detectados) + (0.3 * anomalia_termica), 2)

        # Estructuración del evento automatizado alineado con tu Diccionario de Datos
        event_payload = {
            "id_evento": f"AUT-{datetime.datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000,9999)}",
            "fecha_notificacion": datetime.date.today().strftime("%Y-%m-%d"),
            "mecanismo_vigilancia": "AUT",
            "codigo_comuna_cut": self.comuna_cut,
            "grupo_hospedero": "SIL" if conteo_aves_silvestres > conteo_aves_traspatio else "TRA",
            "abundancia_afectados": total_detectado,
            "signos_neurologicos": signos_detectados,
            "confianza_yolo": score_confianza,
            "temperatura_maxima_aviar": temperatura_max_detectada,
            "estado_alerta_final": 3 if (signos_detectados or anomalia_termica) else 1
        }
        
        return event_payload

    def save_telemetry(self, logs_list):
        """Almacena las detecciones automatizadas sin intervención manual."""
        df_telemetry = pd.DataFrame(logs_list)
        os.makedirs(DATA_OUT_DIR, exist_ok=True)
        
        output_file = os.path.join(DATA_OUT_DIR, f"{datetime.datetime.now().strftime('%Y%m%d')}_cv_telemetry.csv")
        df_telemetry.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\n[ÉXITO] Telemetría de visión artificial guardada de forma reproducible en: {output_file}")


if __name__ == "__main__":
    # Instanciar el monitor inteligente para la comuna de El Monte (Zona de alto riesgo industrial)
    monitor = AvianIntelligentMonitor(camera_id="CAM_EL_MONTE_02", comuna_cut=13111)
    
    # Simular el monitoreo continuo durante una ventana de tiempo (ej. 5 ciclos analíticos)
    telemetria_capturada = []
    print("\nIniciando bucle de inferencia en tiempo real...")
    
    for ciclo in range(1, 6):
        print(f"Procesando ciclo analítico {ciclo}/5...")
        alerta = monitor.process_frame_pipeline()
        telemetria_capturada.append(alerta)
        
        # Si se detecta un estado crítico, el script lo reporta inmediatamente en consola
        if alerta["estado_alerta_final"] == 3:
            print(f"  ⚠️ [ALERTA CRÍTICA IA] Detectada anomalía! Confianza: {alerta['confianza_yolo']} | Temp: {alerta['temperatura_maxima_aviar']}°C")
        else:
            print(f"  [Estado: Normal] Aves en escena: {alerta['abundancia_afectados']} | Sin signología crítica.")
            
        time.sleep(0.5) # Pausa simulada entre frames de video
        
    # Guardar los resultados en la base de datos procesada para su posterior uso en la g-fórmula
    monitor.save_telemetry(telemetria_capturada)
