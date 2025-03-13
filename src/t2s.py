from openai import OpenAI
import os
class T2S:
    def __init__(self,api_key):
        self.client = OpenAI(api_key=api_key)    
    
    def play_answer(self,answer : str):
        response = self.client.audio.speech.create(
        model="tts-1",
        voice="ash",
        input=answer
    )
        response.write_to_file("response.mp3")
        os.system("play response.mp3")