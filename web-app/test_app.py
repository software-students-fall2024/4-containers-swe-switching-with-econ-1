"""Module for testing web app"""

import os
import pytest
from app import create_flask_app

@pytest.fixture(name="app")
def fixture_app():
    """Fixture for creating the Flask app for testing."""
    os.environ["MONGO_URI"] = "mongodb://localhost/test_db"  # Use a test database URI
    app = create_flask_app()
    yield app

@pytest.fixture(name="client")
def fixture_client(app):
    """Fixture for creating a test client for the Flask app."""
    return app.test_client()

# Test Flask routes

def test_home_route(client):
    """Test the home route redirects to /index."""
    response = client.get('/')
    assert response.status_code == 302
    assert response.location.endswith('/index')

def test_index_route(client):
    """Test the /index route renders the template."""
    response = client.get('/index')
    assert response.status_code == 200
    assert b"Emotion Recognizer" in response.data
