from openai import OpenAI
import sounddevice as sd
import soundfile as sf

def save_audio():
        

        SAMPLING_RATE = 44100  # Częstotliwość próbkowania (44.1 kHz)
        DURATION = 5  # Czas nagrywania w sekundach
        OUTPUT_FILENAME = "user.mp3"  # Nazwa pliku wyjściowego

        print("Nagrywanie...")
        # Nagrywanie dźwięku
        audio_data = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
        sd.wait()  # Czekaj na zakończenie nagrywania
        print("Nagrywanie zakończone.")

        # Zapisz dźwięk do pliku
        sf.write(OUTPUT_FILENAME, audio_data, SAMPLING_RATE)
        print(f"Dźwięk zapisany do {OUTPUT_FILENAME}")


class S2T:
    def __init__(self,api_key):
        self.client = OpenAI(api_key=api_key)    
        
    def get_answer(self) -> str:
        save_audio()
        
        audio_file= open("user.mp3", "rb")
        return self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        ).text
        