# Dokumentacja projektu - Asystent głosowy z rozpoznawaniem emocji

## Opis projektu
System asystenta głosowego wykorzystujący:
- Rozpoznawanie mowy (STT - Speech-to-Text)
- Syntezę mowy (TTS - Text-to-Speech)
- Rozpoznawanie emocji z obrazu twarzy
- Model językowy dostosowujący odpowiedzi do emocji użytkownika

## Główne komponenty

### 1. T2S (Text-to-Speech)
Klasa odpowiedzialna za zamianę tekstu na mowę przy użyciu API OpenAI

Metody:
- `__init__(api_key)` - inicjalizacja klienta OpenAI
- `play_answer(answer: str)` - generuje i odtwarza odpowiedź audio

### 2. S2T (Speech-to-Text)
Klasa odpowiedzialna za zamianę mowy na tekst przy użyciu API OpenAI

Metody:
- `__init__(api_key)` - inicjalizacja klienta OpenAI
- `get_answer() -> str` - nagrywa i transkrybuje mowę użytkownika

### 3. EmotionRecognition
Klasa do rozpoznawania emocji z obrazu twarzy

Metody:
- `__init__(model_path)` - ładuje model TensorFlow Lite
- `preprocess_image(image)` - przygotowuje obraz do analizy
- `predict_emotion(image)` - przewiduje emocje z obrazu
- `read_emotion()` - przechwytuje obraz z kamery i rozpoznaje emocje

### 4. LLM (Language Model)
Klasa zarządzająca modelem językowym i przepływem konwersacji

Metody:
- `__init__(api_key)` - inicjalizacja modelu GPT-4
- `generate(state)` - generuje odpowiedź na podstawie stanu
- `generate_graph()` - tworzy graf przepływu konwersacji
- `invoke(message, emotion)` - wywołuje cały proces generowania odpowiedzi

## Przepływ działania
1. Użytkownik inicjuje rozmowę (naciśnięcie klawisza)
2. System nagrywa pytanie użytkownika (S2T)
3. System przechwytuje obraz twarzy i rozpoznaje emocje (EmotionRecognition)
4. Model językowy generuje odpowiedź dostosowaną do emocji (LLM)
5. System odtwarza odpowiedź głosem (T2S)

## Wymagania techniczne
- Python 3.8+
- Biblioteki:
  - openai
  - sounddevice
  - soundfile
  - cv2 (OpenCV)
  - numpy
  - tensorflow
  - langgraph
  - langchain
  - pytorch

## Konfiguracja
1. Wymagany klucz API OpenAI w zmiennej środowiskowej OPENAI_API_KEY
2. Model do rozpoznawania emocji w formacie .tflite

## Przykłady użycia
System obsługuje różne emocje i dostosowuje odpowiedzi:
- Smutek: odpowiedzi pełne współczucia
- Złość: spokojne i pomocne
- Radość: entuzjastyczne i pozytywne
- Neutralne: konkretne i rzeczowe

## Obsługa języków
System automatycznie wykrywa język użytkownika i odpowiada w tym samym języku (obsługa angielskiego i polskiego)

## Ograniczenia
- Czas odpowiedzi ograniczony do ~10 sekund
- Rozpoznawanie emocji działa tylko z widoczną twarzą
- Wymagane połączenie internetowe dla API OpenAI
