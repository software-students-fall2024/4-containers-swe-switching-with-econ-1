import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification
import wave
import librosa

# Load the Wav2Vec2 model for emotion classification
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 10
OUTPUT_FILENAME = "output.wav"

# Function to record audio from the microphone
def record_audio(filename=OUTPUT_FILENAME, duration=RECORD_SECONDS):
    p = pyaudio.PyAudio()

    print(f"Recording for {duration} seconds...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

# Function to classify emotion from the recorded audio using Wav2Vec2
def classify_emotion_from_audio(filename=OUTPUT_FILENAME):
    # Load the audio file and preprocess it
    speech, sr = librosa.load(filename, sr=SAMPLE_RATE)
    
    # Convert speech to tensor (matching the model input type)
    input_values = torch.tensor(speech).unsqueeze(0)  # Add batch dimension

    # Pass the audio input to the model
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Get the predicted emotion (highest logit score)
    predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Emotion labels based on the model's fine-tuning
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotion_labels[predicted_class]
    
    return predicted_emotion

# Main function to record audio and classify emotion
def main():
    # Record audio from the microphone
    record_audio()

    # Classify the emotion from the recorded audio
    emotion_result = classify_emotion_from_audio()

    print(f"Emotion Detected: {emotion_result}")

if __name__ == "__main__":
    main()
