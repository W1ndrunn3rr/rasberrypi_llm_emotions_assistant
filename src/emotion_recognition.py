from transformers import pipeline
import cv2
from PIL import Image
import time
class EmotionRecognition:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
        
        
    def read_emotion(self) -> str:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        
    
        result, image = cam.read()
            
        if result:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return self.pipe(image_pil)[0]["label"]
        else:
            return "neutral"    
                

er = EmotionRecognition()

print(er.read_emotion())