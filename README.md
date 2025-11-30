# Ecobot Vision System

> Sistema de visión artificial optimizado para la clasificación automática de residuos en Máquinas de Vending Inverso (RVM) en Colombia.

![Status](https://img.shields.io/badge/Status-En_Desarrollo-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-YOLOv8_Nano-green)
![Hardware](https://img.shields.io/badge/Device-Raspberry_Pi_3-red)

## Descripción del Proyecto
Este proyecto busca modernizar la infraestructura de reciclaje de **Ecobot** mediante la implementación de Deep Learning en el borde (Edge Computing). El sistema clasifica en tiempo real los residuos depositados en las máquinas RVM para automatizar la trazabilidad y mejorar la experiencia de usuario.

### Clases a Detectar
1. **Botella PET** (Transparente/Color)
2. **Lata** (Aluminio)
3. **Tetrapack**
4. **Botellita de Amor** (Botella rellena de plásticos flexibles)

## Arquitectura y Modelado
Debido a las limitaciones de hardware (Raspberry Pi 3) y las condiciones de iluminación variables en las máquinas actuales, se ha optado por una arquitectura basada en eventos:

* **Trigger:** Sensor físico detecta ingreso.
* **Captura:** Cámara controlada + Iluminación LED sincronizada (Flash).
* **Inferencia:** Modelo YOLOv8 Nano (Optimizado para velocidad).

### Métricas de Rendimiento (Validación)
El modelo actual **YOLOv8 Nano** ha alcanzado resultados sobresalientes en el set de validación:

| Métrica | Valor | Interpretación |
| :--- | :--- | :--- |
| **mAP50** | **99.15%** | Precisión global casi perfecta. |
| **mAP50-95** | **86.98%** | Alta precisión en el ajuste de las cajas (bounding boxes). |
| **Recall (PET)** | **100%** | No se pierden botellas transparentes (clase crítica). |
| **Tiempo** | **~9.5 ms** | Apto para inferencia en tiempo real. |

## Estructura del Repositorio

```text
ecobot-system/
├── docs/       # Documentación del proyecto, arquitectura y papers.
├── models/     # Pesos entrenados (.pt, .onnx).
├── notebooks/  # Experimentos de entrenamiento (Colab/Jupyter).
├── results/    # Gráficas de rendimiento, matrices de confusión.
├── src/        # Código fuente para despliegue en Raspberry Pi.
└── data/       # (Local) Datasets crudos y procesados.
```

## Instalación y Uso
(Sección en construcción para el despliegue del script src/inference.py)

    Clonar el repositorio.
    Instalar dependencias: pip install -r requirements.txt
    Ejecutar inferencia.

## Equipo
* Andrés Cano
* Daniel García
* Farid Sandoval