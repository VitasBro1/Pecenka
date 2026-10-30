from ultralytics import YOLO

model = YOLO("bestV2.pt")
model.export(format="onnx", opset=12, simplify=True)