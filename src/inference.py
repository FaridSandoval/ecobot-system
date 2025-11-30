import cv2
from ultralytics import YOLO
import time
import os

# --- CONFIGURACIÓN (Simulando el entorno de la Raspberry Pi) ---
# Ruta al modelo entrenado (Asegúrate de tener el archivo .pt en la carpeta models)
MODEL_PATH = '../models/best.pt' 
# Umbral de confianza (Solo aceptamos detecciones con > 50% de seguridad)
CONF_THRESHOLD = 0.5 

def load_model():
    """Carga el modelo YOLO en memoria una sola vez al iniciar."""
    print(f"🔄 Cargando modelo desde {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return None

def simulate_sensor_event(image_path, model):
    """
    Simula lo que pasa cuando el sensor detecta una botella:
    1. Recibe la ruta de la foto (tomada por la cámara).
    2. Ejecuta la inferencia.
    3. Retorna qué residuo es.
    """
    print(f"📸 Sensor activado. Procesando imagen: {image_path}")
    
    start_time = time.time()
    
    # Ejecutar inferencia (Task: Detect)
    results = model.predict(
        source=image_path, 
        conf=CONF_THRESHOLD, 
        save=True,  # Guarda la imagen con la caja dibujada para auditoría (MLOps)
        verbose=False
    )
    
    inference_time = (time.time() - start_time) * 1000 # Convertir a ms
    
    # Procesar resultados
    for result in results:
        # Si no detectó nada
        if len(result.boxes) == 0:
            return "Desconocido", 0.0, inference_time
        
        # Tomamos la detección con mayor confianza (la primera)
        box = result.boxes[0]
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        
        return class_name, confidence, inference_time

# --- BLOQUE PRINCIPAL (Main) ---
if __name__ == "__main__":
    # 1. Cargar el "cerebro"
    ai_model = load_model()
    
    if ai_model:
        # 2. Simular una prueba (Cambia esto por una imagen real que tengas en 'data')
        # Si no tienes imagen, el código fallará aquí, pero la estructura es la correcta.
        test_image = "../data/test_bottle.jpg" 
        
        # Solo corremos si existe la imagen de prueba para evitar errores
        if os.path.exists(test_image):
            residuo, conf, tiempo = simulate_sensor_event(test_image, ai_model)
            
            print("-" * 30)
            print(f"🧠 RESULTADO DE IA:")
            print(f"📦 Objeto: {residuo}")
            print(f"📊 Confianza: {conf:.2%}")
            print(f"⚡ Tiempo de Procesamiento: {tiempo:.0f} ms")
            print("-" * 30)
        else:
            print(f"⚠️ No se encontró imagen de prueba en {test_image}. Coloca una foto ahí para probar.")