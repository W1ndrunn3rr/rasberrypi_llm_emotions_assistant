from src.LLM import LLM
import dotenv
import os
from src.t2s import T2S
from src.s2t import S2T
from src.emotion_recognition import EmotionRecognition
import statistics

API_KEY = os.getenv("OPENAI_API_KEY")



def main():
    dotenv.load_dotenv()
    
    llm = LLM(api_key=API_KEY)
    t2s = T2S(api_key=API_KEY)
    s2t = S2T(api_key=API_KEY)
    em = EmotionRecognition()
    emotions = []
    
    
    while(True):
        user_input = input("Press key to start...")
        
        if user_input == 'q':
            exit(0)
        
        emotions.extend(em.read_emotion())
        user_question = s2t.get_answer()
        emotions.extend(em.read_emotion())
        
        chosen_emotion = statistics.mode(emotions)
        
        
        print(user_question,emotions,chosen_emotion)
        answer = llm.invoke(user_question, chosen_emotion)
        t2s.play_answer(answer)
	emotions = []       
   

if __name__ == "__main__":
    main()
