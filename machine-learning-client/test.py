import pytest
from unittest import mock
import numpy as np
import torch
import librosa
from EmotionDetector import record_audio, classify_emotion_from_audio 

# Test for record_audio function
@mock.patch('pyaudio.PyAudio')
@mock.patch('wave.open')
def test_record_audio(mock_wave_open, mock_pyaudio):
    mock_stream = mock.Mock()
    mock_pyaudio.return_value.open.return_value = mock_stream

    # Simulate reading data
    mock_stream.read.return_value = b'\x00' * 1024
    mock_wave_file = mock.Mock()
    mock_wave_open.return_value.__enter__.return_value = mock_wave_file

    # Mock sample width return value (2 bytes for 16-bit audio)
    mock_pyaudio.return_value.get_sample_size.return_value = 2  

    # Call function
    record_audio(filename='test_output.wav', duration=1)

    # Assertions to ensure stream is opened, read, and closed correctly
    mock_pyaudio.return_value.open.assert_called_once()
    mock_stream.read.assert_called()
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_pyaudio.return_value.terminate.assert_called_once()
    mock_wave_open.assert_called_once_with('test_output.wav', 'wb')

# Test classify_emotion_from_audio function
@mock.patch('librosa.load')
@mock.patch('torch.no_grad')
@mock.patch('torch.argmax')
@mock.patch('transformers.Wav2Vec2ForSequenceClassification.from_pretrained')
def test_classify_emotion_from_audio(mock_model, mock_argmax, mock_no_grad, mock_load):
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
