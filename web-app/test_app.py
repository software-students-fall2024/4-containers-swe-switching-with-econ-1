"""Module for web app tests"""

from unittest.mock import patch
from io import BytesIO
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_flask_app, store_audio_in_mongodb, get_advice


@pytest.fixture(name="app")
def fixture_app():
    """Fixture to create a Flask app instance for testing."""
    app = create_flask_app()
    app.config.update(
        {
            "TESTING": True,
            "DEBUG": False,
        }
    )
    yield app


@pytest.fixture(name="client")
def fixture_client(app: Flask):
    """Fixture to create a test client for the Flask app."""
    return app.test_client()


@patch("app.fs.put")
def test_store_audio_in_mongodb(mock_fs_put):
    """Test storing an audio file in MongoDB."""
    mock_fs_put.return_value = "mock_file_id"
    file_obj = BytesIO(b"mock_audio_data")
    filename = "test_audio.wav"

    result = store_audio_in_mongodb(file_obj, filename)

    assert result == "mock_file_id"
    mock_fs_put.assert_called_once_with(file_obj, filename=filename)


def test_home_redirect(client: FlaskClient):
    """Test that the home route redirects to the index page."""
    response = client.get("/")
    assert response.status_code == 302
    assert response.location.endswith("/index")


def test_index_page(client: FlaskClient):
    """Test rendering the index page."""
    response = client.get("/index")
    assert response.status_code == 200
    assert b"Emotion Recognizer" in response.data


@patch("app.store_audio_in_mongodb")
@patch("app.requests.post")
def test_stop_route_success(mock_requests_post, mock_store_audio, client: FlaskClient):
    """Test the /stop route with successful emotion detection."""
    mock_store_audio.return_value = "mock_file_id"
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {"emotion": "happy"}

    audio_data = BytesIO(b"mock_audio_data")
    data = {"file": (audio_data, "recording.wav")}

    response = client.post("/stop", data=data, content_type="multipart/form-data")

    assert response.status_code == 200

    mock_store_audio.assert_called_once()
    mock_requests_post.assert_called_once_with(
        "http://ml_client:4000/detect-emotion",
        json={"fileId": "mock_file_id"},
        timeout=30,
    )


def test_stop_route_no_file(client: FlaskClient):
    """Test the /stop route when no file is provided."""
    response = client.post("/stop", content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.json == {"message": "No file part in request"}


def test_stop_route_no_filename(client: FlaskClient):
    """Test the /stop route when an empty filename is provided."""
    empty_audio_file = BytesIO(b"")
    data = {"file": (empty_audio_file, "")}

    response = client.post("/stop", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.json == {"message": "No file selected"}


@pytest.mark.parametrize(
    "emotion, expected_phrases",
    [
        ("angry", ["angry", "anger"]),
        ("disgust", ["repulsed", "bothered", "uneasy"]),
        ("fear", ["scared", "fear", "overwhelmed"]),
        ("happy", ["happy", "joy", "positive"]),
        ("neutral", ["neutral", "calm", "composed"]),
        ("sad", ["sad", "heavy", "sorry"]),
        ("surprise", ["surprise", "unexpected", "shaken"]),
    ],
)
def test_get_advice_valid_emotions(emotion, expected_phrases):
    """
    Test get_advice with valid emotions and ensure returned advice is appropriate.
    """
    advice = get_advice(emotion)
    assert isinstance(advice, str), "Advice should be a string."
    assert any(
        phrase in advice.lower() for phrase in expected_phrases
    ), f"Advice for emotion '{emotion}' did not contain expected phrases. Advice: {advice}"


def test_get_advice_unknown_emotion():
    """
    Test get_advice with an unknown emotion.
    """
    advice = get_advice("confused")
    assert (
        advice == "Unknown emotion."
    ), f"Unexpected advice for unknown emotion: {advice}"
