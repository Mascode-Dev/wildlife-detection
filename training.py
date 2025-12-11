from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 model
model = YOLO('yolov8s.pt') 


if __name__ == "__main__":
    # 2. Start training with custom dataset
    results = model.train(
        data='data.yaml',      # Data configuration file
        epochs=100,            # Number of epochs (adjust if needed)
        imgsz=640,             # Input image size (640x640)
        batch=-1,              # Automatic batch size
        name='wildlife_detection_v1', # Session name
        patience=50            # Stop if no improvement for 50 epochs (prevents overfitting)
    )

    print("Training completed. Results saved in runs/detect/wildlife_detection_v1")