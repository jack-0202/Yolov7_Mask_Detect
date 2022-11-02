import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import non_max_suppression

class Functions:
    def __init__(self,DEVICE):
        self.Device = DEVICE
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def predict(self,model,image, image_size):
        image = np.asarray(image)
        
        # Resize image to the inference size
        ori_h, ori_w = image.shape[:2]
        image = cv2.resize(image, (image_size, image_size))
        
        # Transform image from numpy to torch format
        image_pt = torch.from_numpy(image).permute(2, 0, 1).to(self.Device)
        image_pt = image_pt.float() / 255.0
        
        # Infer
        with torch.no_grad():
            pred = model(image_pt[None], augment=False)[0]
        
        # NMS
        pred = non_max_suppression(pred)[0].cpu().numpy()
        
        # Resize boxes to the original image size
        pred[:, [0, 2]] *= ori_w / image_size
        pred[:, [1, 3]] *= ori_h / image_size
        
        return pred
    
    def capture_rectangle(self,frame,x1,y1,x2,y2,conf,class_id,CLASSES,Color=(255,0,0),type_=0):
        # type_ = 0 : normally
        if type_ == 0:
            x1,y1,x2,y2,class_id = int(x1),int(y1),int(x2),int(y2),int(class_id)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color=Color,thickness=1)
            cv2.rectangle(frame,(x1,y1-20),(x1+60,y1),color=Color,thickness=-1)
            conf_ = " "+ str(round(conf*100 ,2)) + "%"
            cv2.putText(frame,CLASSES[class_id],(x1,y1-5),self.font ,0.5,(255,255,255),1)
            cv2.putText(frame,conf_ ,(x1+65,y1-5),self.font ,0.5,Color,1)
        # type_ = 1 : ignore the text of %
        elif type_ == 1:
            x1,y1,x2,y2,class_id = int(x1),int(y1),int(x2),int(y2),int(class_id)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color=Color,thickness=1)
            cv2.rectangle(frame,(x1,y1-20),(x1+60,y1),color=Color,thickness=-1)
            #conf_ = " "+ str(round(conf*100 ,2)) + "%"
            cv2.putText(frame,CLASSES[class_id],(x1,y1-5),self.font ,0.5,(255,255,255),1)
            #cv2.putText(frame,conf_ ,(x1+65,y1-5),self.font ,0.5,Color,1)
    
    def get_Classes(self,select):
        if select == 0:
            return ['Mask','NoMask']
        elif select == 1:
            return  [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                "sandwich", "orange", "broccoli", "car|rot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
                "teddy bear", "hair drier", "toothbrush"]
        