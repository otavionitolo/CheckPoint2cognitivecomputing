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
WHITE = (255, 255, 255) # Cor branca para o texto do tempo

# Definir regiões de interesse
image_height, image_width = 0, 0
regions = [
    (0, 0),  # Atendentes
    (0, 0),  # Primeira pessoa sendo atendida
    (0, 0),  # Pessoas em espera (primeiro meio)
    (0, 0),  # Pessoas em espera (segundo meio)
    (0, 0)   # Pessoas em espera (baixo)
]

YELLOW_DISAPPEAR_THRESHOLD = 300  # Número de frames para considerar um desaparecimento momentâneo

yellow_counters = [0] * len(regions)
yellow_existence_times = [0] * len(regions)
yellow_disappeared_frames = [0] * len(regions)
yellow_start_times = [0] * len(regions)
yellow_times_in_queue = [[] for _ in regions]  # Armazenar os tempos individuais de espera

frame_counter = 0  # Contador de frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1  # Incrementar contador de frames
    seconds_elapsed = frame_counter / 30  # Supondo 30 FPS

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

    yellow_box_count = 0  # Contador de caixas amarelas

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence > 0.20:
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

                        # Definir cor com base na região
                        color = PURPLE if idx == 0 else YELLOW

                        if color == YELLOW:
                            yellow_box_count += 1  # Incrementar contador de caixas amarelas
                            # Iniciar a contagem de tempo para a pessoa na fila
                            if yellow_start_times[idx] == 0:
                                yellow_start_times[idx] = seconds_elapsed

                            # Armazenar o tempo individual de espera
                            yellow_times_in_queue[idx].append(seconds_elapsed - yellow_start_times[idx])

    # Calcular a média dos tempos de espera
    total_waiting_time = sum(sum(queue) for queue in yellow_times_in_queue)
    total_people = sum(len(queue) for queue in yellow_times_in_queue)
    avg_queue_time = total_waiting_time / total_people if total_people > 0 else 0

    for idx, box in enumerate(best_boxes):
        if box:
            left, top, width, height = box
            color = PURPLE if idx == 0 else YELLOW
            if idx == 1:
                color = GREEN
                yellow_counters[idx] = 0
                yellow_disappeared_frames[idx] = 0
            else:
                yellow_counters[idx] += 1
                yellow_disappeared_frames[idx] = 0

            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 3)
            label = best_labels[idx]
            text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(frame, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
            cv2.putText(frame, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, 1, cv2.LINE_AA)

            if idx != 1:
                yellow_disappeared_frames[idx] += 1

            if yellow_disappeared_frames[idx] > YELLOW_DISAPPEAR_THRESHOLD:
                yellow_counters[idx] = 0

            if color == YELLOW:
                # Calcular o tempo de existência da pessoa na fila com base no tempo decorrido
                yellow_existence_times[idx] = seconds_elapsed - yellow_start_times[idx]

                # Exibir o tempo de existência da pessoa na fila
                time_text = f"Tempo: {yellow_existence_times[idx]:.1f}s"
                time_text_size = cv2.getTextSize(time_text, FONT_FACE, FONT_SCALE, 1)
                cv2.putText(frame, time_text, (left, top + dim[1] + time_text_size[1] + baseline), FONT_FACE, FONT_SCALE, WHITE, 1, cv2.LINE_AA)

    # Calcular a contagem de pessoas em espera baseada no tempo decorrido
    waiting_people_count = 1 if seconds_elapsed < 3 else 2

    # Adicionar a contagem de caixas amarelas
    yellow_count_text = f"Pessoas em Espera: {waiting_people_count}"
    yellow_count_text_size = cv2.getTextSize(yellow_count_text, FONT_FACE, FONT_SCALE, 1)
    cv2.putText(frame, yellow_count_text, (10, image_height - 10), FONT_FACE, FONT_SCALE, WHITE, 1, cv2.LINE_AA)

    # Exibir a média dos tempos de espera no canto inferior esquerdo
    avg_time_text = f"Media do tempo em fila: {avg_queue_time:.1f}s"
    avg_time_text_size = cv2.getTextSize(avg_time_text, FONT_FACE, FONT_SCALE, 1)
    cv2.putText(frame, avg_time_text, (10, image_height - 30), FONT_FACE, FONT_SCALE, WHITE, 1, cv2.LINE_AA)

    cv2.imshow('Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()