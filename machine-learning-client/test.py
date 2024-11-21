"""Modules for tests"""

from unittest import mock
import numpy as np
import pytest
import torch
from bson import ObjectId
import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure
import librosa
from transformers import Wav2Vec2ForSequenceClassification
from emotion_detector import classify_emotion_from_audio, create_flask_app


@mock.patch("librosa.load")
@mock.patch("torch.argmax")
@mock.patch("transformers.Wav2Vec2ForSequenceClassification.from_pretrained")
def test_classify_emotion_from_audio_sad(mock_model, mock_argmax, mock_load):
    """Test classify_emotion_from_audio for sad emotion"""
    # Mock librosa output (Simulated audio and sample rate)
    mock_load.return_value = (np.random.rand(16000).astype(np.float32), 16000)

    # Mock model's output
    mock_logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.1]], dtype=torch.float32
    )
    mock_model.return_value.return_value.logits = mock_logits

    # Mock argmax to return index for 'sad'
    mock_argmax.return_value = torch.tensor(5)

    # Call function
    emotion = classify_emotion_from_audio(filename="test_output.wav")

    # Assertions
    mock_load.assert_called_once_with("test_output.wav", sr=16000)
    assert emotion == "sad"


@mock.patch("librosa.load")
@mock.patch("torch.argmax")
@mock.patch("transformers.Wav2Vec2ForSequenceClassification.from_pretrained")
def test_classify_emotion_from_audio_noisy(mock_model, mock_argmax, mock_load):
    """Test classify_emotion_from_audio with noisy audio"""
    # Mock librosa to return random noise as audio data
    mock_load.return_value = (np.random.rand(16000).astype(np.float32), 16000)

    # Mock model's output
    mock_logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.1]], dtype=torch.float32
    )
    mock_model.return_value.return_value.logits = mock_logits

    # Mock argmax to return index for 'neutral' in noisy data
    mock_argmax.return_value = torch.tensor(4)

    # Call function
    emotion = classify_emotion_from_audio(filename="noisy_output.wav")

    # Assertions
    mock_load.assert_called_once_with("noisy_output.wav", sr=16000)
    assert emotion == "neutral"


@mock.patch("librosa.load")
def test_classify_emotion_from_audio_format_error(mock_load):
    """Test classify_emotion_from_audio when file format is not supported"""
    # Mock librosa to raise an error for unsupported file format
    mock_load.side_effect = librosa.util.exceptions.LibrosaError(
        "Unsupported file format"
    )

    # Call function and check if it raises expected error
    with pytest.raises(librosa.util.exceptions.LibrosaError):
        classify_emotion_from_audio(filename="unsupported_format_file.wav")


@mock.patch("librosa.load")
@mock.patch("torch.argmax")
@mock.patch("transformers.Wav2Vec2ForSequenceClassification.from_pretrained")
def test_classify_emotion_from_audio_all_emotions(mock_model, mock_argmax, mock_load):
    """Test classify_emotion_from_audio for all emotions"""

    # Mock librosa output (Simulated audio and sample rate)
    mock_load.return_value = (np.random.rand(16000).astype(np.float32), 16000)

    # Mock model's output
    mock_logits = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.1]], dtype=torch.float32
    )
    mock_model.return_value.return_value.logits = mock_logits

    # Loop through each emotion label
    for i, emotion in enumerate(
        ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    ):
        mock_argmax.return_value = torch.tensor(i)

        # Call function
        emotion_result = classify_emotion_from_audio(filename="test_output.wav")

        # Assertions
        assert emotion_result == emotion


@mock.patch("emotion_detector.classify_emotion_from_audio")
def test_emotion_route_missing_fileid(mock_classify_emotion):
    """Test Flask route when no fileId is provided"""
    mock_classify_emotion.return_value = "neutral"

    app = create_flask_app()

    with app.test_client() as client:
        # Send a request without fileId
        response = client.post("/detect-emotion", json={})

        # Assert that response returns a 400 error
        assert response.status_code == 400
        assert response.json["error"] == "fileId is required"


@mock.patch("emotion_detector.classify_emotion_from_audio")
def test_emotion_route_invalid_fileid(mock_classify_emotion):
    """Test Flask route when an invalid fileId is provided"""
    mock_classify_emotion.return_value = "neutral"

    app = create_flask_app()

    with app.test_client() as client:
        # Send a request with an invalid fileId
        response = client.post("/detect-emotion", json={"fileId": "invalid_id"})

        # Assert that response returns a 400 error
        assert response.status_code == 400
        assert response.json["error"] == "Invalid fileId"


@mock.patch("pymongo.MongoClient")
def test_mongo_connection_error_handling(mock_client):
    """Test MongoDB connection error handling"""
    # Simulate a connection failure
    mock_client.side_effect = ConnectionFailure("Failed to connect to MongoDB")

    with pytest.raises(ConnectionFailure):
        pymongo.MongoClient("mongodb://localhost:27017")


def test_mongo_operation_failure_with_update():
    """Test MongoDB operation failure when updating a document"""
    with mock.patch("pymongo.MongoClient") as mock_client:
        mock_db = mock.MagicMock()
        mock_client.return_value = mock_db
        mock_db["audio-analysis"].fs.files.update_one.side_effect = OperationFailure(
            "Operation failed"
        )

        with pytest.raises(OperationFailure):
            mock_db["audio-analysis"].fs.files.update_one(
                {"_id": ObjectId()}, {"$set": {"emotion": "happy"}}
            )


@mock.patch("transformers.Wav2Vec2ForSequenceClassification.from_pretrained")
def test_model_loading_error(mock_model):
    """Test model loading failure"""
    mock_model.side_effect = Exception("Failed to load model")

    # Ensure that the exception is raised when attempting to load the model
    with pytest.raises(Exception):
        Wav2Vec2ForSequenceClassification.from_pretrained("model-name")
