"""
This module implements a Flask web application that records audio from the microphone,
saves it to a file, and stores the file in a MongoDB database using GridFS.
Modules:
    flask: A micro web framework for Python.
    pymongo: A Python driver for MongoDB.
    pyaudio: Provides Python bindings for PortAudio, the cross-platform audio library.
    wave: Provides a convenient interface to the WAV sound format.
    threading: Higher-level threading interface.
    gridfs: A Python library for working with MongoDB GridFS.
Constants:
    SAMPLE_RATE (int): The sample rate for audio recording.
    CHANNELS (int): The number of audio channels.
    FORMAT: The format for audio recording.
    CHUNK_SIZE (int): The chunk size for audio recording.
    OUTPUT_FILENAME (str): The default filename for the recorded audio.
Functions:
    store_audio_in_mongodb(filename):
    create_flask_app():
"""

import os
import random
from flask import Flask, render_template, redirect, url_for, jsonify, request
import pymongo
import pyaudio
import gridfs
import requests
from pymongo.errors import ConnectionFailure, OperationFailure


# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
OUTPUT_FILENAME = "static/output.wav"

uri = os.getenv("MONGO_URI", "mongodb://localhost/emotions")
client = pymongo.MongoClient(uri)
db = client["audio-analysis"]
fs = gridfs.GridFS(db)


def store_audio_in_mongodb(file_obj, filename):
    """
    Stores an audio file in a MongoDB database using GridFS.

    Args:
        file_obj: the file to be stored
        filename (str): The path to the audio file to be stored.

    Returns:
        str: The ObjectId of the stored file
    """
    file_id = fs.put(file_obj, filename=filename)
    print(f"Audio file '{filename}' stored in MongoDB with ObjectId: {file_id}")
    return str(file_id)


def get_advice(emotion):
    """
    Get advice based on the detected emotion.
    """
    angry_advice = [
        "It sounds like you're feeling really angry right now. It's [\
            okay to feel that way. Sometimes, just letting it out can help.",
        "I hear a lot of anger in your voice. It's totally understandable \
            to feel upset sometimes. You’re not alone in this.",
        "It seems like you're feeling frustrated. It's okay to express that \
            anger and I'm here for you no matter what you're going through.",
    ]

    disgust_advice = [
        "It seems like something has really bothered you. It’s okay to feel \
            disgusted—it’s just your body’s way of processing what doesn’t feel right.",
        "I hear your discomfort. You don’t have to like everything and it's \
            okay to feel uneasy sometimes. Take it easy, I’m here for you.",
        "I can sense that you're feeling repulsed by something. It's okay to \
            have those feelings and you don’t need to face them alone.",
    ]

    fear_advice = [
        "It sounds like you're feeling scared and I want you to know that it’s \
            okay to feel that way. You're safe here.",
        "I hear a lot of fear in your voice. It's a tough feeling, but you’re not \
            alone and I’m here to support you through it.",
        "It seems like you're feeling overwhelmed by fear. Whatever you're facing, \
            you're stronger than you think and I’m here to help however I can.",
    ]

    happy_advice = [
        "You sound so happy! I’m really glad to hear that. It’s wonderful to experience \
            joy and I’m here to share in that positivity with you.",
        "I hear the happiness in your voice and it’s contagious! It’s so nice to hear you \
            feeling this way and I’m really happy for you.",
        "It seems like you’re in a great mood today. Keep enjoying that positive energy \
            and I'm here to celebrate it with you!",
    ]

    neutral_advice = [
        "It sounds like you’re feeling calm and steady. Sometimes, being in a neutral \
            space can feel peaceful. I’m here if you want to talk or just relax.",
        "You seem composed and at ease right now, which can be such a comforting place to \
            be. I’m here to support you however you need.",
        "I hear that you’re in a neutral space and that’s okay. Not every day is full of \
            highs or lows. Take it one step at a time.",
    ]

    sad_advice = [
        "I can hear that you’re feeling sad and I’m really sorry you're going through this. \
            It's okay to feel this way and you don’t have to carry it alone.",
        "It seems like you're feeling heavy with sadness. I’m here for you and I want you to \
            know that it's okay to feel down sometimes.",
        "I hear the sadness in your voice. It’s tough to feel this way, but you're not alone \
            in this. Talk to anyone you need to.",
    ]

    suprise_advice = [
        "You sound surprised. Sometimes things catch us off guard. Whatever it is, \
            take as much as you need to process it at your own pace.",
        "It seems like something unexpected has happened. Surprises can be overwhelming, \
            but whatever you're going through, I’m here for you.",
        "I hear the surprise in your voice. It’s okay to feel a little shaken by the \
            unexpected and I’m here to support you through it.",
    ]
    advice = ""
    if emotion == "angry":
        advice = random.choice(angry_advice)
    elif emotion == "disgust":
        advice = random.choice(disgust_advice)
    elif emotion == "fear":
        advice = random.choice(fear_advice)
    elif emotion == "happy":
        advice = random.choice(happy_advice)
    elif emotion == "neutral":
        advice = random.choice(neutral_advice)
    elif emotion == "sad":
        advice = random.choice(sad_advice)
    elif emotion == "surprise":
        advice = random.choice(suprise_advice)
    else:
        advice = "Unknown emotion."
    return advice


def create_flask_app():
    """
    Create and configure the Flask application.
    This function sets up the Flask app and defines the routes for the web application.
    Returns:
        Flask app: The configured Flask application instance.
    Routes:
        /: Redirects to the index page.
        /index: Renders the index.html template.
        /stop: Stops the audio recording process.
    """
    flask_app = Flask(__name__)
    flask_app.secret_key = "KEY"

    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except ConnectionFailure:
        print("Failed to connect to MongoDB. Please check your connection.")
    except OperationFailure:
        print(
            "Operation failed. Please verify your credentials or database configuration."
        )

    ################### Routes ###################
    @flask_app.route("/")
    def home():
        return redirect(url_for("index"))

    @flask_app.route("/index", methods=["GET"])
    def index():
        return render_template("index.html")

    @flask_app.route("/stop", methods=["GET", "POST"])
    def stop():
        print("Made it to /stop!")
        if "file" not in request.files:
            return jsonify({"message": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"message": "No file selected"}), 400

        file_id = store_audio_in_mongodb(file, filename=OUTPUT_FILENAME)
        url = "http://ml_client:4000/detect-emotion"
        params = {"fileId": str(file_id)}
        response = requests.post(url, json=params, timeout=30)
        emotion = response.json()["emotion"]
        advice = get_advice(emotion)
        return jsonify({"emotion": emotion, "advice": advice})

    return flask_app


if __name__ == "__main__":
    app = create_flask_app()
    FLASK_PORT = 3000
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
