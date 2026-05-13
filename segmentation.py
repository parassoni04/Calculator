import cv2
import numpy as np
from typing import Tuple

def loadImage(imagePath: str) -> np.ndarray:
    return cv2.imread(imagePath)

def toGrayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def sortBoxes(boxes: list, lineThreshold: int = 15) -> list:
    if not boxes:
        return []
    boxes.sort(key=lambda b: b[1])
    rows = []
    currentRow = [boxes[0]]

    for box in boxes[1:]:
        if abs(box[1] - currentRow[0][1]) <= lineThreshold:
            currentRow.append(box)
        else:
            rows.append(sorted(currentRow, key=lambda b: b[0]))
            currentRow = [box]
    rows.append(sorted(currentRow, key=lambda b: b[0]))

    return [box for row in rows for box in row]

def findTokenBoxes(gray: np.ndarray, minArea: int = 50) -> list:
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= minArea:
            boxes.append((x, y, w, h))
    return sortBoxes(boxes)

def cropAndResize(gray: np.ndarray, box: Tuple, padding: int, targetSize: int = 28) -> np.ndarray:
    x, y, w, h = box
    imgHeight, imgWidth = gray.shape
    x1 = np.clip(x - padding, 0, imgWidth)
    y1 = np.clip(y - padding, 0, imgHeight)
    x2 = np.clip(x + w + padding, 0, imgWidth)
    y2 = np.clip(y + h + padding, 0, imgHeight)
    crop = gray[y1:y2, x1:x2]
    return cv2.resize(crop, (targetSize, targetSize), interpolation=cv2.INTER_AREA)

def segment(imagePath: str) -> list[np.ndarray]:
    img = loadImage(imagePath)
    gray = toGrayscale(img)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    boxes = findTokenBoxes(gray)
    return [cropAndResize(gray, box, 8, 28) for box in boxes]