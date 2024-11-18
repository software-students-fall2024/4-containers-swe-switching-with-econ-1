import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pymongo 
from dotenv import load_dotenv
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import pyaudio
import wave
import threading

# Constants for audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024
OUTPUT_FILENAME = "output.wav"

# Global flag to control recording
recording_flag = threading.Event()

# Function to record audio from the microphone
def record_audio(filename=OUTPUT_FILENAME):
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
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))


def create_app():
    app = Flask(__name__)
    app.secret_key ="KEY"


    # test conecctions
    uri = 'mongodb://localhost/project_4'
    client = pymongo.MongoClient(uri, server_api=ServerApi('1'))
    db = client['project_4']

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    ################### Routes ###################
    # Redirect to the index page
    @app.route('/')
    def home():
        return redirect(url_for('index'))
    
    @app.route('/index', methods=['GET'])
    def index():
        return render_template('index.html')
    
    # Start the audio recording
    @app.route('/start', methods=['GET', 'POST'])
    def start():
        global recording_flag
        recording_flag.set()
        threading.Thread(target=record_audio).start()
        return jsonify({"status": "Audio processing started"})
    
    # Stop the audio recording
    @app.route('/stop', methods=(['GET', 'POST']))
    def stop():
        global recording_flag
        recording_flag.clear()
        return jsonify({"status": "Audio processing finished"})

    return app
    
if __name__ == "__main__":
    app = create_app()
    flask_port = 3000
    app.run(port=flask_port, debug=True)