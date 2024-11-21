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


def test_get_advice_angry():
    advice = get_advice("angry")
    assert advice in [
        "It sounds like you're feeling really angry right now. It's okay to feel that way. Sometimes, just letting it out can help.",
        "I hear a lot of anger in your voice. It's totally understandable to feel upset sometimes. You’re not alone in this.",
        "It seems like you're feeling frustrated. It's okay to express that anger and I'm here for you no matter what you're going through.",
    ]


def test_get_advice_digust():
    advice = get_advice("disgust")
    assert advice in [
        "It seems like something has really bothered you. It’s okay to feel disgusted—it’s just your body’s way of processing what doesn’t feel right.",
        "I hear your discomfort. You don’t have to like everything and it's okay to feel uneasy sometimes. Take it easy, I’m here for you.",
        "I can sense that you're feeling repulsed by something. It's okay to have those feelings and you don’t need to face them alone.",
    ]


def test_get_advice_fear():
    advice = get_advice("fear")
    assert advice in [
        "It sounds like you're feeling scared and I want you to know that it’s okay to feel that way. You're safe here.",
        "I hear a lot of fear in your voice. It's a tough feeling, but you’re not alone and I’m here to support you through it.",
        "It seems like you're feeling overwhelmed by fear. Whatever you're facing, you're stronger than you think and I’m here to help however I can.",
    ]


def test_get_advice_neutral():
    advice = get_advice("neutral")
    assert advice in [
        "It sounds like you’re feeling calm and steady. Sometimes, being in a neutral space can feel peaceful. I’m here if you want to talk or just relax.",
        "You seem composed and at ease right now, which can be such a comforting place to be. I’m here to support you however you need.",
        "I hear that you’re in a neutral space and that’s okay. Not every day is full of highs or lows. Take it one step at a time.",
    ]


def test_get_advice_happy():
    advice = get_advice("happy")
    assert advice in [
        "You sound so happy! I’m really glad to hear that. It’s wonderful to experience joy and I’m here to share in that positivity with you.",
        "I hear the happiness in your voice and it’s contagious! It’s so nice to hear you feeling this way and I’m really happy for you.",
        "It seems like you’re in a great mood today. Keep enjoying that positive energy and I'm here to celebrate it with you!",
    ]


def test_get_advice_sad():
    advice = get_advice("sad")
    assert advice in [
        "I can hear that you’re feeling sad and I’m really sorry you're going through this. It's okay to feel this way and you don’t have to carry it alone.",
        "It seems like you're feeling heavy with sadness. I’m here for you and I want you to know that it's okay to feel down sometimes.",
        "I hear the sadness in your voice. It’s tough to feel this way, but you're not alone in this. Talk to anyone you need to.",
    ]


def test_get_advice_surprise():
    advice = get_advice("surprise")
    assert advice in [
        "You sound surprised. Sometimes things catch us off guard. Whatever it is, take as much as you need to process it at your own pace.",
        "It seems like something unexpected has happened. Surprises can be overwhelming, but whatever you're going through, I’m here for you.",
        "I hear the surprise in your voice. It’s okay to feel a little shaken by the unexpected and I’m here to support you through it.",
    ]
