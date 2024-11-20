"""Module for audio stuff"""
import os
import pyaudio
import torch
from transformers import Wav2Vec2ForSequenceClassification
import librosa
from flask import Flask, request, jsonify
import pymongo
import gridfs
from bson import ObjectId
from pymongo.errors import ConnectionFailure, OperationFailure

# Load the Wav2Vec2 model for emotion classification
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
RECORD_SECONDS = 15
OUTPUT_FILENAME = "../web-app/static/output.wav"

# database connection
uri = os.getenv("MONGO_URI", "mongodb://localhost/emotions")
client = pymongo.MongoClient(uri)
db = client['audio-analysis']
fs = gridfs.GridFS(db)

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
    except ConnectionFailure:
        print("Failed to connect to MongoDB. Please check your connection.")
    except OperationFailure:
        print("Operation failed. Please verify your credentials or database configuration.")
    ################### Routes ###################
    # Stop the audio recording
    @flask_app.route('/detect-emotion', methods=['POST'])
    def emotion():
        web_request = request.get_json()
        file_id = web_request['fileId']
        emotion = classify_emotion_from_audio()
        db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": {"emotion": emotion}}
        )
        print("Sending the emotion:", emotion)
        return jsonify({"emotion": emotion}), 200

    return flask_app

if __name__ == "__main__":
    app = create_flask_app()
    FLASK_PORT = 4000
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
