import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForSequenceClassification
import wave
import librosa
from flask import Flask, request, jsonify
import pymongo
import os
import gridfs
from bson import ObjectId

# Load the Wav2Vec2 model for emotion classification
model = Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 15
OUTPUT_FILENAME = "../web-app/static/output.wav"

uri = os.getenv("MONGO_URI", "mongodb://localhost/emotions")
client = pymongo.MongoClient(uri)
db = client['audio-analysis']
fs = gridfs.GridFS(db)

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
    emotion_labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotion_labels[predicted_class]
    
    return predicted_emotion

# Main function to record audio and classify emotion
def main():
    # Record audio from the microphone
    record_audio()

    # Classify the emotion from the recorded audio
    emotion_result = classify_emotion_from_audio()

    print(f"Emotion Detected: {emotion_result}")

def create_flask_app():
    """
    Create and configure the Flask application.
    This function sets up the Flask app and defines the routes for the web application.
    Returns:
        Flask app: The configured Flask application instance.
    Routes:
        /detect-emotion, received an ObjectId from webapp and adds the emotion to the 
        corresponding document, then sends the emotion back to webapp
    """
    flask_app = Flask(__name__)
    flask_app.secret_key ="KEY"
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    ################### Routes ###################
    # Stop the audio recording
    @flask_app.route('/detect-emotion', methods=(['POST']))
    def emotion():
        webRequest = request.get_json()
        fileId = webRequest['fileId']
        emotion = classify_emotion_from_audio()
        db.fs.files.update_one(
            {"_id": ObjectId(fileId)},
            {"$set": {"emotion": emotion}}
        )
        print("Sending the emotion:", emotion)
        return jsonify({"emotion": emotion}), 200

    return flask_app

if __name__ == "__main__":
    app = create_flask_app()
    FLASK_PORT = 4000
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
