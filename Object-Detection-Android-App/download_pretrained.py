from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Download pretrained PyTorch model
model.export(format="tflite")  # Creates 'yolov8n_float32.tflite'