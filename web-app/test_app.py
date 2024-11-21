"""Module for web app tests"""

from unittest.mock import patch
from io import BytesIO
import pytest
from flask import Flask
from flask.testing import FlaskClient
import requests

from app import create_flask_app, store_audio_in_mongodb


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
    assert response.json == {"emotion": "happy"}

    mock_store_audio.assert_called_once()
    mock_requests_post.assert_called_once_with(
        "http://127.0.0.1:4000/detect-emotion",
        json={"fileId": "mock_file_id"},
        timeout=30,
    )


def test_stop_route_no_file(client: FlaskClient):
    """Test the /stop route when no file is provided."""
    response = client.post("/stop", content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.json == {"message": "No file part in request"}


def test_stop_route_request_error(client):
    """Test the /stop route when the external request fails."""
    data = {"file": (BytesIO(b"audio_data"), "recording.wav")}

    # Mock store_audio_in_mongodb to prevent actual database interactions
    with patch("app.store_audio_in_mongodb") as mock_store:
        mock_store.return_value = "mocked_file_id"

        # Mock requests.post to raise a RequestException
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException(
                "Request failed"
            )

            response = client.post(
                "/stop", data=data, content_type="multipart/form-data"
            )
            assert response.status_code == 502
            assert b"Error connecting to emotion detection service" in response.data


def test_stop_route_no_filename(client: FlaskClient):
    """Test the /stop route when an empty filename is provided."""
    empty_audio_file = BytesIO(b"")
    data = {"file": (empty_audio_file, "")}

    response = client.post("/stop", data=data, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.json == {"message": "No file selected"}
