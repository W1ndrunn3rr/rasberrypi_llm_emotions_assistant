from transformers import pipeline
import cv2
import time
class EmotionRecognition:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="trpakov/vit-face-expression")
        
        
    def read_emotion(self) -> list[str]:
        emotions = []
        cam = cv2.VideoCapture(0)
        
        for i in range(25):
            result, image = cam.read()
            
            if result:
                cv2.imwrite("emotion.png", image)
            else:
                print(result,image)

            emotions.append(self.pipe("emotion.png")[0]["label"])
            time.sleep(0.0001)
            
        return emotions
        
    
    
