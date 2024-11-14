import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pymongo 
from dotenv import load_dotenv
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

def create_app():
    app = Flask(__name__)
    app.secret_key ="KEY"

    uri = os.getenv("MONGO_URI")
    client = pymongo.MongoClient(uri, server_api=ServerApi('1'))
    db = client[os.getenv("DB_NAME")]

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    @app.route('/')
    def index():
        return render_template('index.html')
    

    if __name__ == "__main__":
        app = create_app()
        flask_port = os.getenv("FLASK_PORT")
        app.run(port=flask_port, debug=True)