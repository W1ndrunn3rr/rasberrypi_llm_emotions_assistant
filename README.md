# Project Documentation - Voice Assistant with Emotion Recognition

## Project Description
A voice assistant system utilizing:
- Speech recognition (STT - Speech-to-Text)
- Speech synthesis (TTS - Text-to-Speech)
- Facial emotion recognition
- Language model adapting responses to user's emotions

## Main Components

### 1. T2S (Text-to-Speech)
Class responsible for converting text to speech using OpenAI API

Methods:
- `__init__(api_key)` - initializes OpenAI client
- `play_answer(answer: str)` - generates and plays audio response

### 2. S2T (Speech-to-Text)
Class responsible for converting speech to text using OpenAI API

Methods:
- `__init__(api_key)` - initializes OpenAI client
- `get_answer() -> str` - records and transcribes user's speech

### 3. EmotionRecognition
Class for recognizing emotions from facial images

Methods:
- `__init__(model_path)` - loads TensorFlow Lite model
- `preprocess_image(image)` - prepares image for analysis
- `predict_emotion(image)` - predicts emotions from image
- `read_emotion()` - captures camera image and recognizes emotions

### 4. LLM (Language Model)
Class managing the language model and conversation flow

Methods:
- `__init__(api_key)` - initializes GPT-4 model
- `generate(state)` - generates response based on state
- `generate_graph()` - creates conversation flow graph
- `invoke(message, emotion)` - invokes the entire response generation process

## Workflow
1. User initiates conversation (key press)
2. System records user's question (S2T)
3. System captures facial image and recognizes emotions (EmotionRecognition)
4. Language model generates emotion-adapted response (LLM)
5. System plays the response with voice (T2S)

## Technical Requirements
- Python 3.8+
- Libraries:
  - openai
  - sounddevice
  - soundfile
  - cv2 (OpenCV)
  - numpy
  - tensorflow
  - langgraph
  - langchain

## Configuration
1. OpenAI API key required in OPENAI_API_KEY environment variable
2. Emotion recognition model in .tflite format

## Usage Examples
The system handles different emotions and adapts responses:
- Sadness: compassionate responses
- Anger: calm and helpful
- Happiness: enthusiastic and positive
- Neutral: concrete and factual

## Language Support
The system automatically detects user's language and responds in the same language (supports English and Polish)

## Limitations
- Response time limited to ~10 seconds
- Emotion recognition works only with visible face
- Internet connection required for OpenAI API