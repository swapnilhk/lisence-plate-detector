from ultralytics import YOLO as yolo

# Load models
model = yolo("yolov8n.pt")

# Use the model
results = model.train(data="dataset_config.yaml", epochs=1)