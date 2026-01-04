#!/bin/bash
# Check if Android PulseAudio source is available
DEVICE_NAME=$(grep "device:" config.yaml | awk '{print $2}' | tr -d '"')

echo "Checking for audio device: $DEVICE_NAME"
if pactl list short sources | grep -q "$DEVICE_NAME"; then
    echo "SUCCESS: Device $DEVICE_NAME found."
    
    echo "Attempting a 3-second test recording to test.wav..."
    ffmpeg -y -f pulse -i "$DEVICE_NAME" -t 3 test.wav
    
    if [ $? -eq 0 ]; then
        echo "Recording successful. Play 'test.wav' to verify."
    else
        echo "ERROR: ffmpeg recording failed."
    fi
else
    echo "ERROR: Device $DEVICE_NAME NOT found in 'pactl list short sources'."
    pactl list short sources
fi
