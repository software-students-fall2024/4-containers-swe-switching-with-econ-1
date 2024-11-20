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
    record_audio(filename=OUTPUT_FILENAME):
    store_audio_in_mongodb(filename):
    create_flask_app():
"""
import os
import wave
import threading
from flask import Flask, render_template, redirect, url_for, jsonify
import pymongo
# from dotenv import load_dotenv
# from pymongo.server_api import ServerApi
# from bson.objectid import ObjectId
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

# Global flag to control recording
recording_flag = threading.Event()

# Global fileId to hold ObjectId of the newest file
FILE_ID = None

# This lets us know when each recording is finished
recording_done = threading.Event()

uri = os.getenv("MONGO_URI", "mongodb://localhost/emotions")
client = pymongo.MongoClient(uri)
db = client['audio-analysis']
fs = gridfs.GridFS(db)

# Function to record audio from the microphone
def record_audio(filename=OUTPUT_FILENAME):
    """
    Records audio from the microphone and saves it to a file.

    Args:
        filename (str): The name of the file to save the recorded audio. 
        Defaults to OUTPUT_FILENAME.
    The function performs the following steps:
    1. Initializes the PyAudio object.
    2. Opens a stream to record audio with the specified format, 
    channels, sample rate, and chunk size.
    3. Continuously reads audio data from the stream while the recording_flag is set.
    4. Stops and closes the stream once recording is finished.
    5. Saves the recorded audio data to a file in WAV format.
    6. Stores the audio file in MongoDB using the store_audio_in_mongodb function.
    7. Stores the fileId to the global object
    """
    p = pyaudio.PyAudio()

    print("Recording...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    frames = []
    while recording_flag.is_set():
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    with wave.open(filename, 'wb') as wf:
        wf: "Wave_write"
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    # Store the audio file in MongoDB
    global FILE_ID
    FILE_ID = store_audio_in_mongodb(filename)
    recording_done.set()

def store_audio_in_mongodb(filename):
    """
    Stores an audio file in a MongoDB database using GridFS.

    Args:
        filename (str): The path to the audio file to be stored.

    Returns:
        The objectID of the stored file
    """
    with open(filename, 'rb') as f:
        file_id = fs.put(f, filename=filename)
    print(f"Audio file '{filename}' stored in MongoDB with ObjectId: {file_id}")
    return file_id

def create_flask_app():
    """
    Create and configure the Flask application.
    This function sets up the Flask app and defines the routes for the web application.
    Returns:
        Flask app: The configured Flask application instance.
    Routes:
        /: Redirects to the index page.
        /index: Renders the index.html template.
        /start: Starts the audio recording process.
        /stop: Stops the audio recording process.
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
    # Redirect to the index page
    @flask_app.route('/')
    def home():
        return redirect(url_for('index'))
    @flask_app.route('/index', methods=['GET'])
    def index():
        return render_template('index.html')
    # Start the audio recording
    @flask_app.route('/start', methods=['GET', 'POST'])
    def start():
        recording_flag.set()
        threading.Thread(target=record_audio).start()
        return jsonify({"status": "Audio processing started"})
    # Stop the audio recording
    @flask_app.route('/stop', methods=(['GET', 'POST']))
    def stop():
        if recording_flag.is_set():
            recording_flag.clear()
            recording_done.wait()
            recording_done.clear()
            if FILE_ID:
                # Send a request with the fileId to ML client
                url = "http://127.0.0.1:4000/detect-emotion"
                params = {"fileId": str(FILE_ID)}
                response = requests.post(url, json=params, timeout=30)
                if response.status_code == 200:
                    return jsonify(response.json())
                print("Error request")
                return jsonify({"status": "error sending request!"})
            return jsonify({
                "status": "Audio processing finished",
                "message": "No file stored"
            }), 200
        return jsonify({"error": "No recording in progress"}), 400


    return flask_app
if __name__ == "__main__":
    app = create_flask_app()
    FLASK_PORT = 3000
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True)
