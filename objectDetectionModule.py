import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import distance

yolov3_weights = "YOLOv3_model/yolov3.weights"
yolov3_cfg = "YOLOv3_model/yolov3.cfg"
coco_names = "YOLOv3_model/coco.names"

class objectDetector():

    def __init__(self, yolov3_weights = yolov3_weights, yolov3_cfg = yolov3_cfg, coco_names = coco_names):
        self.yolov3_weights = yolov3_weights
        self.yolov3_cfg = yolov3_cfg
        self.coco_names = coco_names

        self.yolo = cv2.dnn.readNet(self.yolov3_weights, self.yolov3_cfg)
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.yolo.getUnconnectedOutLayers()]

        with open(self.coco_names, "r") as file:
            self.classes = [line.strip() for line in file.readlines()] 

        self.colorWhite = (255, 255, 255)   

    def object_detect(self, img, draw=True):
        height, width, channels = img.shape

        # detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
          
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        mid_points = []
        bboxes = []
        for i, conf in zip(range(len(boxes)), confidences):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i]
                class_label = 'person'
                if class_label:
                    text = label+ ' ' +str(round(conf, 2))
                    mid_points.append([int(x+w/2), int(y+h/2)])
                    bboxes.append([x, y, x+w, y+h])
                    if draw:
                        cv2.circle(img, (int(x+w/2), int(y+h/2)), 5, (0, 0, 255), -1)
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colorWhite, 1)


        self.calDistance(len(class_label), mid_points)
        
        return img

    def calDistance(self, n, mid_points):
        d = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                if i != j:
                    dst = distance.euclidean(mid_points[i], mid_points[j])
                    d[i][j] = dst            
        return d

    def redAlert(self):   
        pass 