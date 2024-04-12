# CheckPoint2cognitivecomputing
CheckPoint2 Desenvolver uma aplicação prática e inovadora de visão computacional que possa ser aplicada em vídeos em tempo real. A aplicação deve abordar um problema relevante em uma das seguintes áreas: jogos, entretenimento, saúde, bem-estar, agricultura ou segurança pública.


2
import cv2
3
from matplotlib import pyplot as plt
4
import numpy as np
5
​
6
# Carrega as imagens
7
img = cv2.imread("parte1.PNG")
8
img2 = cv2.imread("parte2.PNG")
9
​
10
# Exibe as imagens que serão usadas
11
plt.figure(figsize=(10, 10))
12
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
13
plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
14
plt.show()
15
​
16
# Cria o objeto Stitcher
17
stitcher = cv2.Stitcher.create()
18
​
19
# Tenta realizar a união das imagens
20
(status, result) = stitcher.stitch((img, img2))
21
​
22
# Verifica se a união foi bem-sucedida
23
if status == cv2.STITCHER_OK:
24
    print("Sucesso! Imagem panorâmica gerada.")
25
    plt.figure(figsize=(10, 10))
26
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
27
    plt.show()
28
    
29
    # Recorta a parte indesejada
30
    print("Dimensões originais:", result.shape)
31
    crop = result[50:850, 50:1300]
32
    plt.figure(figsize=(10, 10))
33
    plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
34
    plt.show()
35
else:
36
    print("Falha ao gerar a imagem panorâmica.")
37
​


JOBS



px
