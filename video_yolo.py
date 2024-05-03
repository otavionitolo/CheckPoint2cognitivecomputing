import cv2
import numpy as np

# Carregar o modelo YOLOv5 ONNX
net = cv2.dnn.readNet('yolov5m.onnx')

# Abrir um vídeo
cap = cv2.VideoCapture("video.mp4")

classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)  # Cor para pessoas em espera
PURPLE = (128, 0, 128)  # Cor para os atendentes
GREEN = (0, 255, 0)     # Cor para a primeira pessoa sendo atendida

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_height, image_width = frame.shape[:2]
    # Definir regiões de interesse mais detalhadamente
    regions = [
        (0, int(image_height * 0.01)),  # Atendentes
        (int(image_height * 0.02), int(image_height * 0.15)),  # Primeira pessoa sendo atendida
        (int(image_height * 0.15), int(image_height * 0.25)),  # Pessoas em espera (primeiro meio)
        (int(image_height * 0.30), int(image_height * 0.45)),  # Pessoas em espera (segundo meio)
        (int(image_height * 0.50), image_height)  # Pessoas em espera (baixo)
    ]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward(net.getUnconnectedOutLayersNames())

    best_boxes = [None] * len(regions)
    best_confidences = [0.45] * len(regions)
    best_labels = ["" for _ in regions]

    rows = outputs[0].shape[1]
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence > 0.45:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes[class_id] == "person":
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]

                for idx, (start, end) in enumerate(regions):
                    if start <= top < end and confidence > best_confidences[idx]:
                        best_confidences[idx] = confidence
                        best_boxes[idx] = box
                        label_name = "Atendente" if idx == 0 else "Pessoa em Espera"
                        if idx == 1:  # Ajuste para a primeira pessoa sendo atendida
                            label_name = "Pessoa Sendo Atendida"
                        best_labels[idx] = f"{label_name}:{confidence:.2f}"

    # Desenhar as caixas com maior confiança em cada região
    for idx, box in enumerate(best_boxes):
        if box:
            left, top, width, height = box
            color = PURPLE if idx == 0 else YELLOW
            if idx == 1:
                color = GREEN  # Correção para a primeira pessoa sendo atendida
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 3)
            label = best_labels[idx]
            text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(frame, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
            cv2.putText(frame, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, 1, cv2.LINE_AA)

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()