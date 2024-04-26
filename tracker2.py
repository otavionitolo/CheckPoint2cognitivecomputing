import cv2
import numpy as np

# Carregar o modelo YOLOv5 ONNX
net = cv2.dnn.readNet('yolov5m.onnx')

# Inicializar o rastreador
tracker = cv2.TrackerKCF_create("video.mp4")

# Carregar os nomes das classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Inicializar a detecção e o rastreamento
def init_detection_and_tracking(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Encontrar o objeto com a maior confiança para rastreamento
    best_confidence = 0
    best_box = None
    rows = outputs[0].shape[1]
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence > best_confidence:
            best_confidence = confidence
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            left = int((cx - w/2) * frame.shape[1] / 640)
            top = int((cy - h/2) * frame.shape[0] / 640)
            width = int(w * frame.shape[1] / 640)
            height = int(h * frame.shape[0] / 640)
            best_box = (left, top, width, height)

    # Inicializar o rastreador com o melhor objeto encontrado
    if best_box is not None:
        tracker.init(frame, best_box)

# Atualizar a detecção e o rastreamento
def update_detection_and_tracking(frame):
    success, box = tracker.update(frame)
    if success:
        left, top, width, height = [int(v) for v in box]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture('video.mp4')
ret, frame = cap.read()
init_detection_and_tracking(frame)

# Loop de captura e processamento de vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break

    update_detection_and_tracking(frame)

    cv2.imshow('YOLO Object Detection and Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
