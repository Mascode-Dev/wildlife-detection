# 🐾 Wildlife Detection

A simple and flexible **wildlife detection system** built with **YOLO object detection models** to detect animals in images or video streams. This repository includes training scripts, detection application scripts, data processing tools, and pretrained YOLO models.

> The goal is to provide an easy-to-use pipeline for training and deploying animal detection models using modern deep learning techniques.

## Features
- Train custom YOLO models on wildlife datasets
- Run real-time detection on images or video
- Tools for converting bounding box formats (JSON → YOLO)
- Includes sample pretrained models for quick evaluation

## Repository Structure
- app_detection.py: Main script for running inference/detection
- training.py: Script to train YOLO models
- convert_json_yolo.py: Convert annotations to YOLO format
- data.yaml: Dataset configuration for training
- main.ipynb: Training and evaluation workflow
- res_process.ipynb: Result analysis notebook
- yolo11n.pt / yolov8s.pt: Pretrained YOLO models
- runs/: Detection output directory

## Quick Start
### 1. Clone the repository
git clone https://github.com/mascode-dev/wildlife-detection.git
cd wildlife-detection

### 2. Create a Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 3. Run Detection Application
python app_detection.py

## Example Models Included
- yolo11n.pt: Fast lightweight model
- yolov8s.pt: More accurate small model
- runs/detect/wildlife_detection_v1/weights/best.pt : The model trained on animals trap cam images

## Contributing
Contributions welcome!
