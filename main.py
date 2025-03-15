from src.LLM import LLM
import dotenv
import os
from src.t2s import T2S
from src.s2t import S2T

API_KEY = os.getenv("OPENAI_API_KEY")



def main():
    dotenv.load_dotenv()
    
    llm = LLM(api_key=API_KEY)
    t2s = T2S(api_key=API_KEY)
    s2t = S2T(api_key=API_KEY)
    
    while(True):
        user_question = s2t.get_answer()
        answer = llm.invoke(user_question, "happy")
        t2s.play_answer(answer)
        
   

if __name__ == "__main__":
    main()
