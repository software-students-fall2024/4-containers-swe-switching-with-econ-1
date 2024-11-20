"""Module for mock objects"""
from unittest import mock
import numpy as np
import torch
from emotion_detector import classify_emotion_from_audio

# Test classify_emotion_from_audio function
@mock.patch('librosa.load')
@mock.patch('torch.argmax')
@mock.patch('transformers.Wav2Vec2ForSequenceClassification.from_pretrained')
def test_classify_emotion_from_audio(mock_model, mock_argmax, mock_load):
    """Function to test classify_emotion_from_audio function"""
    # Mock librosa output (Simulated audio and sample rate)
    mock_load.return_value = (np.random.rand(16000).astype(np.float32), 16000)

    # Mock model's output
    mock_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.1, 0.5, 0.1]], dtype=torch.float32)
    mock_model.return_value.return_value.logits = mock_logits

    # Mock argmax to return the index for 'happy'
    mock_argmax.return_value = torch.tensor(3)

    # Call function
    emotion = classify_emotion_from_audio(filename='test_output.wav')

    # Assertions
    mock_load.assert_called_once_with('test_output.wav', sr=16000)
    assert emotion == 'happy'
