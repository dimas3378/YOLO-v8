from ultralytics import YOLO

#load model
model = YOLO("yolov8n.yaml")

#use the model
results = model.train(data="config.yaml", epochs=20, imgsz=640, batch=2, workers=2) #patience=4

#train yang 1 epochs=10, imgsz=320, batch=2, workers=1