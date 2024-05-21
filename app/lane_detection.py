from array import ArrayType
import cv2
import numpy as np


# Detecta as bordas da imagem e retorna a mascara com as bordas
def canny(frame: ArrayType):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Remove qualquer parte da imagem menor que 5x5, reduçao de ruido
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny


# Seleciona apenas a regiao de interesse no caso e formato triangular
def region_of_interest(frame: ArrayType):
    height = frame.shape[0]
    width = frame.shape[1]
    mask = np.zeros_like(frame)
    triangle = np.array(
        [
            [
                (200, height),
                (800, 350),
                (1200, height),
            ]
        ],
        np.int32,
    )
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


# Utiliza a Transformada de Hough para deteccao de formas geometricas
def hough_lines(frame: ArrayType):
    return cv2.HoughLinesP(
        frame, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
    )


# Retorna um ponto para criaçao de uma linha
def make_points(image: ArrayType, line: ArrayType):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


# Baseado nas linhas determina quais sao importantes ou nao
def average_slope_intercept(frame: ArrayType, lines: ArrayType):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(frame, left_fit_average)
    right_line = make_points(frame, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def add_weighted(frame: ArrayType, line_image: ArrayType):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)


# Desenha uma linha no frame
def display_lines(frame: ArrayType, lines: ArrayType):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image


def main():
    video_path = input("Enter the video path: ")
    cap = cv2.VideoCapture(video_path)
    while cv2.waitKey(1) != ord("q"):
        _, frame = cap.read()
        if frame is None:
            break
        # Cria uma mascara com reduçao de ruidos
        canny_image = canny(frame)
        # Seleciona apenas a regiao de interesse da imagem e aplica a mascara
        masked_image = region_of_interest(canny_image)
        lines = hough_lines(masked_image)
        average_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, average_lines)
        combo_image = add_weighted(frame, line_image)
        cv2.imshow("Frame", combo_image)

    cap.release()
    cv2.destroyAllWindows()


if "__main__":
    main()
