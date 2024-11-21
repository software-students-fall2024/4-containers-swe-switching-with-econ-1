FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    ffmpeg \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -r requirements.txt

ADD . .

CMD [ "python", "emotion_detector.py"]