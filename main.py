from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(data='E:\python\projects\cv\Object_detection\Project2\config.yaml', epochs=250)