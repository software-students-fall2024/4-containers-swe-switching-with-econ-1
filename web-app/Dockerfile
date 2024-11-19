FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN apt-get install gcc -y
RUN pip3 install -r requirements.txt

ADD . .

EXPOSE 3000

CMD [ "python", "app.py"]