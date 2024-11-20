"""Module for audio stuff"""
import pyaudio
import torch
from transformers import Wav2Vec2ForSequenceClassification
import librosa

# Load the Wav2Vec2 model for emotion classification
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 15
OUTPUT_FILENAME = "output.wav"

def classify_emotion_from_audio(filename=OUTPUT_FILENAME):
    """Function to classify emotion from the recorded audio using Wav2Vec2"""

    # Load the audio file and preprocess it
    speech, _ = librosa.load(filename, sr=SAMPLE_RATE)

    # Convert speech to tensor (matching the model input type)
    input_values = torch.tensor(speech).unsqueeze(0)

    # Pass the audio input to the model
    with torch.no_grad():
        logits = model(input_values).logits

    # Get the predicted emotion (highest logit score)
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Emotion labels based on the model's fine-tuning
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotion_labels[predicted_class]

    return predicted_emotion
