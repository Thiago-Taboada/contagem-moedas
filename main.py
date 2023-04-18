import cv2
import numpy as np


def preprocess_image(frame):
    # Converte a imagen a escala de cinzas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplica um filtro Gaussiano para reduzir o ruido
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    # Detecta as bordas na imagem utilizando o algoritmo Canny
    edges = cv2.Canny(blurred, 30, 150)
    return edges


def detect_circles(edges):
    # Detecta os circulos na imagem utilizando a funçao de Hough
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=200, param2=30, minRadius=0, maxRadius=0)
    return circles


def main():
    cap = cv2.VideoCapture('video_monedas4.mp4')
    # cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))
        edges = preprocess_image(frame)
        circles = detect_circles(edges)
        total = 0

        # Cria uma copia da imagem original
        frame_copy = frame.copy()

        # Desenha os retangulos encontrados na imagem original
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Calcula as coordenadas do retangulo
                x1, y1 = x - r, y - r
                x2, y2 = x + r, y + r
                # Desenha o retangulo e mostra a area
                area = 4 * r ** 2

                if area > 27000 and area < 33000:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, "1 Real", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    total += 1
                elif area > 23500:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, "25 Centavos", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    total += 0.25
                elif area > 20800:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, "50 Centavos", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    total += 0.5
                elif area > 17500:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, "5 Centavos", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    total += 0.05
                elif area > 15000:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, "10 Centavos", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    total += 0.1
                else:
                    cv2.rectangle(frame_copy, (x - r, y - r), (x + r, y + r), (94,230,7), 2)
                    cv2.putText(frame_copy, f"Area: {area:.2f} px", (x - r, y - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame_copy, (430, 30), (600, 80), (37, 38, 38), -1)
        cv2.putText(frame_copy, f'R$ {total:.2f}', (440, 67), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1)

        # Mostra as imagens originais e processadas em janelas separadas
        cv2.imshow("Processada", edges)
        cv2.imshow("Original", frame_copy)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()