from transformers import pipeline
import cv2
class EmotionRecognition:
    def __init__():
        pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
        
    def read_emotion() -> None:
        image = cv2.imread("../czarny.png")
        print(image)
        
    
    
em = EmotionRecognition()

em.read_emotion()
        