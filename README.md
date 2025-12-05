# Ecobot Vision System: Edge AI for Waste Classification
This project is a part of the **Technological Innovation Project 2** course in the Applied Artificial Intelligence Master, Universidad Icesi, Cali, Colombia.

#### -- Project Status: [Active]

## Contributing Members

**Team Leader: Farid Sandoval (@FaridSandoval)**
**Instructor: Milton Orlando Sarria Paja & Jose Andres Moncada Quintero**

#### Other Members:

|Name     |  Role   | 
|---------|-----------------|
|Andrés Cano| Developer / Hardware Specialist |
|Daniel García| Data Scientist / ML Engineer |

## Contact
* Feel free to contact the team leader with any questions or if you are interested in contributing!

## Project Intro/Objective
The purpose of this project is to modernize **Ecobot's** recycling infrastructure by implementing Deep Learning at the edge (Edge Computing). [cite_start]The system classifies waste deposited in Reverse Vending Machines (RVM) in real-time to automate traceability and improve user experience.

### Partner
* **Ecobot Colombia**
* Website: [ecobot.com.co](https://ecobot.com.co)
* Partner contact: ian.rodriguez@u.icesi.edu.co

### Methods Used
* Deep Learning (Object Detection)
* Edge Computing (Inference on Device)
* Computer Vision
* Event-Driven Architecture

### Technologies
* **Hardware:** Raspberry Pi 3, Pi Camera, IR/Physical Sensors.
* **Software:** Python 3.10.
* **Libraries:** Ultralytics (YOLOv8), OpenCV, Pandas, Numpy, Torch.
* **Model:** YOLOv8 Nano (Optimized for speed).

## Project Description
This project addresses the challenge of deploying modern Artificial Intelligence models on hardware with limited resources. [cite_start]Since the current machines operate on a **Raspberry Pi 3**, processing a continuous video stream for detection is not viable due to computational constraints.

To solve this, we opted for an **Event-Driven Architecture**:
1.  **Trigger:** A physical sensor detects object entry.
2.  **Capture:** Synchronized LED lighting (Flash) is activated, and a static photo is taken.
3.  **Inference:** The YOLOv8 Nano model processes the single image.

### Classes to Detect 
1.  **PET Bottle** (Transparent/Color)
2.  **Can** (Aluminum)
3.  **Tetrapack**
4.  **"Botellita de Amor"** (Bottle filled with flexible plastics)

### Performance Metrics (Validation)
The current **YOLOv8 Nano** model has achieved outstanding results on the validation set:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **mAP50** | **99.15%** | Near-perfect global precision. |
| **mAP50-95** | **86.98%** | High precision in bounding box adjustments. |
| **Recall (PET)** | **100%** | No transparent bottles are missed (critical class). |
| **Inference Time** | **~9.5 ms** | Suitable for real-time inference. |

## Getting Started
Instructions for contributors to deploy the inference system on the Raspberry Pi.

1.  **Clone the repository:**
    ```bash
    git clone [repository-url]
    cd ecobot-system
    ```

2.  **Raw Data:**
    Raw and processed datasets are kept locally in the `data/` folder due to size and privacy constraints.

3.  **Install Dependencies:**
    The project requires specific libraries for computer vision and tensor handling.
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include `ultralytics`, `torch`, `opencv-python-headless`, `matplotlib`, `pandas`, and `numpy`.*

4.  **Run Inference (Simulation):**
    To test the system without the physical sensor hardware, execute the inference script:
    ```bash
    python src/inference.py
    ```
    *This script loads the `best.pt` model and processes a test image to simulate a sensor event.*

## Featured Notebooks/Analysis/Deliverables
* [Training Experiment - YOLOv8](notebooks/Yolov8.ipynb): Notebook used for training and fine-tuning the model, including hyperparameter setup.
* [Inference Script](src/inference.py): Production source code for deployment on Raspberry Pi.
* [Metrics Report](results/): Performance charts and confusion matrices.
