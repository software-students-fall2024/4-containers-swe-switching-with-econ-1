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

    @flask_app.route("/stop", methods=["POST"])
    def stop():
        print("Made it to /stop!")
        if "file" not in request.files:
            return jsonify({"message": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"message": "No file selected"}), 400
        try:
            file_id = store_audio_in_mongodb(file, filename=OUTPUT_FILENAME)
            url = "http://127.0.0.1:4000/detect-emotion"
            params = {"fileId": str(file_id)}
            response = requests.post(url, json=params, timeout=30)
            if response.status_code == 200:
                return jsonify(response.json())
            print("Error request")
            return jsonify({"status": "error sending request!"})
        except pymongo.errors.PyMongoError as mongo_err:
            print(f"MongoDB Error: {mongo_err}")
            return jsonify({"message": "Database error", "error": str(mongo_err)}), 500
        except requests.exceptions.RequestException as req_err:
            print(f"Request Error: {req_err}")
            return (
                jsonify(
                    {
                        "message": "Error connecting to emotion detection service",
                        "error": str(req_err),
                    }
                ),
                502,
            )

return flask_app


if __name__ == "__main__":
    app = create_flask_app()
    FLASK_PORT = 3000
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
