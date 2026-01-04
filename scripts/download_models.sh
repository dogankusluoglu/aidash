#!/bin/bash
mkdir -p /home/pi/aidash/models
cd /home/pi/aidash/models

if [ ! -d "vosk-small-en-us" ]; then
    echo "Downloading Vosk small model..."
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip
    mv vosk-model-small-en-us-0.15 vosk-small-en-us
    rm vosk-model-small-en-us-0.15.zip
    echo "Vosk model downloaded."
else
    echo "Vosk model already exists."
fi
