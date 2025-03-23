import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

class EmotionRecognition:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()


        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_resized = cv2.resize(image_gray, (48, 48))

        image_normalized = image_resized / 255.0

        image_expanded = np.expand_dims(image_normalized, axis=-1)

    
        image_final = np.expand_dims(image_expanded, axis=0)

        return image_final.astype(np.float32)  

    def predict_emotion(self, image: np.ndarray) -> str:
        processed_image = self.preprocess_image(image)

        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)

        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        predicted_label = emotions[np.argmax(output_data)]

        return predicted_label

    def read_emotion(self) -> str:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        result, image = cam.read()
        cam.release()

        if result:
            return self.predict_emotion(image)
        else:
            return "neutral"  