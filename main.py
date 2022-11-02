import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression
from Functions import Functions

DEVICE = 'cpu'
Func = Functions(DEVICE)

WEIGHTS_1 = "weights/mask.pt"
WEIGHTS_2 = "weights/yolov7.pt"

model_1 = attempt_load(WEIGHTS_1, map_location=DEVICE)
model_2 = attempt_load(WEIGHTS_2, map_location=DEVICE)

CLASSES_1 = Func.get_Classes(0)
CLASSES_2 = Func.get_Classes(1)
    
capture = cv2.VideoCapture(0)       # 初始化攝影功能
while(capture.isOpened()):
    ret, frame = capture.read()     # 讀取設請鏡頭的影像
    
    pred_1 = Func.predict(model_1,frame,416)       # 預設為 640x640
    pred_2 = Func.predict(model_2,frame,320)       # 預設為 640x640

    for x1, y1, x2, y2, conf, class_id in pred_1: 
        if conf > 0.7:
            Func.capture_rectangle(frame,x1, y1, x2, y2, conf, class_id, CLASSES_1,(0,0,255),0)
            
            
    for x1, y1, x2, y2, conf, class_id in pred_2:
        if conf > 0.3:
            Func.capture_rectangle(frame ,x1, y1, x2, y2, conf, class_id, CLASSES_2,type_=1)
    
    cv2.imshow('Frame',frame)       # 顯示攝影鏡頭的影像
    c = cv2.waitKey(1)              # 等待時間 1 毫秒 ms
    if c == 27:                     # 按 Esc 键, 結束
        break
capture.release()                   # 關閉攝影功能
cv2.destroyAllWindows()