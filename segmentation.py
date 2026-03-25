import cv2#type:ignore
import numpy as np 
from typing import Tuple

def loadImage(imagePath : str) -> np.ndarray:
    img : np.ndarray = cv2.imread(imagePath)
    return img

def toGrayscale(img : np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def findTokenBoxes(gray : np.ndarray, minArea = 50) -> list:
    tempTuple : Tuple = cv2.findContours(gray, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    contours : np.ndarray = tempTuple[0]
    boxes : list = []
    
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w*h>=minArea:
            boxes.append((x,y,w,h))
    boxes.sort(key = lambda b : b[0])
    return boxes

def cropAndResize(gray : np.ndarray, box : Tuple , padding : int , targetSize : int = 28 ) ->  np.ndarray:
    x, y, w, h  =  box
    imgHieght , imgWieght  = gray.shape
    
    x1 = np.clip(x-padding, 0 , imgWieght)
    y1 = np.clip(y-padding, 0 , imgHieght)
    x2 = np.clip(x+w+padding, 0 , imgWieght)
    y2 = np.clip(y+h+padding, 0 , imgHieght)
    
    crop  = gray[y1:y2,x1:x2]
    
    resized = cv2.resize(crop , (targetSize,targetSize),interpolation = cv2.INTER_AREA)
    return resized

def segment(imagePath : str) -> list[np.ndarray]:
    img: np.ndarray = loadImage(imagePath)
    gray: np.ndarray = toGrayscale(img)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    boxes: list[Tuple[int , int , int ,int]] = findTokenBoxes(gray)
    crops: list[np.ndarray] = []
    for box in boxes :
        crop = cropAndResize(gray,box,8,28)
        crops.append(crop)
    return crops

