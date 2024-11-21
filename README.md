# Build Badges

![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

Unit Test Badge Here


# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.

## Description

Our project is an audio-based recognition system for activity. It's designed to analyze various sound events, such as clapping, snapping, and hitting a desk and classify them accordingly. The system leverages Docker for scalability, and operates in a containerized environment, as per the instructions.

Our project is an Emotions Recogniton and Mental Health Advice app that used audio-based cognition to analyze and classify the user's emotions. The system analyzes the user's audio input and classifies the emotion, then provides the user with advice based on the detected emotions. The app uses multiple sub-systems and operates in a containerized environment, using Docker.  


## Configuration Instructions

[Make sure you have Python 3.10 downloaded from HERE](https://www.python.org/downloads/)

[Then, download Docker Desktop HERE](https://www.docker.com/products/docker-desktop/)

After installing Docker, make sure Docker Desktop is running.

Create a local clone of the remote Github repository by running the following command in terminal or VSCode: 

```
>>> git clone https://github.com/software-students-fall2024/4-containers-swe-switching-with-econ-1
```

Open terminal in local cloned repository (Terminal will automatically open at this directory if you open a project repository using VSCode) and run the following command in the terminal:
```
>>> docker-compose up
```

[Wait for build to finish then open the page in your browser at localhost 3000](http://localhost:3000/)


## Usage

The website will run on localhost:3000. The website will default to the home page with no login necessary. 

The home page will have a title, a description of the application, and a 'start' button to start recording audio. 

Once the 'start' button has been pressed it will begin to record audio data from the user's microphone. The 'start' button is replaced by a red 'stop button'.

Once the 'stop' button has been pressed, the audio recording will end and the audio data will be stored into the DB container which will then be accessed by the back-end ML cilent to be analyzed and classified. 

Once the audio has been classified, the back-end machine will send the information to the front-end web-app ad before being sent to the browser.

The user will then be presented with the results of the classification of the audio recording and advice based on the classification.


## Team members

- [Terry Cao](https://github.com/cao-exe)
- [William Cao](https://github.com/FriedBananaBan)
- [Leanne Lu](https://github.com/leannelu)
- [Samuel Tang](https://github.com/stango1234556)

## Environment Vars and Data Import Instructions

N/A