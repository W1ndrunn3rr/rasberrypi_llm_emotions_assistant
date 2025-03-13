from src.LLM import LLM
import dotenv
import os
from src.t2s import T2S
import sounddevice as sd
import soundfile as sf

API_KEY = os.getenv("OPENAI_API_KEY")

def save_audio():
    # Parametry nagrywania
    SAMPLING_RATE = 44100  # Częstotliwość próbkowania (44.1 kHz)
    DURATION = 5  # Czas nagrywania w sekundach
    OUTPUT_FILENAME = "user.mp3"  # Nazwa pliku wyjściowego

    print("Nagrywanie...")
    print(f"Urządzenia :  {sd.query_devices()}")
    # Nagrywanie dźwięku
    audio_data = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
    sd.wait()  # Czekaj na zakończenie nagrywania
    print("Nagrywanie zakończone.")

    # Zapisz dźwięk do pliku
    sf.write(OUTPUT_FILENAME, audio_data, SAMPLING_RATE)
    print(f"Dźwięk zapisany do {OUTPUT_FILENAME}")



def main():
    dotenv.load_dotenv()
    save_audio()
    
    llm = LLM(api_key=API_KEY)
    t2s = T2S(api_key=API_KEY)
    
    answer = llm.invoke("Kto był ostatnim królem polski?", "neutral")
    t2s.play_answer(answer)
    
   

if __name__ == "__main__":
    main()
