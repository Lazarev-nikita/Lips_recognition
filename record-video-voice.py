import threading
import os
import pathlib
import datetime
import cv2
import pyaudio
import wave
import time
import sys
from modules import settings
from modules import camera

import os.path
import json


def select_audio(p: pyaudio.PyAudio()):
    print('Select microphone from list:')
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        if d['maxInputChannels'] > 0:
            print(i, d['name'])
    return int(input())

def start_record():
    pass

def stop_record():
    pass

MICROPHONE_INDEX = -1
CAMERA_INDEX = -1
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 2
AUDIO_RATE = 44100

class GeneratorState:
    RECORD_NOT_STARTED  = 1
    RECORD_IN_PROGRESS  = 2
    RECORD_SAVING = 3
    DONE            = 4

RECORDS_OUT_PATH = './result-data'
pathlib.Path(RECORDS_OUT_PATH).mkdir(parents=True, exist_ok=True)

generatorSettings = settings.Settings('record-settings.json')
MICROPHONE_INDEX = generatorSettings.get_value('last-microphone', -1)
CAMERA_INDEX = generatorSettings.get_value('last-camera', -1)
FONT = cv2.FONT_HERSHEY_COMPLEX
iteration: int = generatorSettings.get_value('last-iteration', 0)
generatedDataMap: dict = generatorSettings.get_value('generated-data', {})


p = pyaudio.PyAudio()

if MICROPHONE_INDEX == -1:
    MICROPHONE_INDEX = select_audio(p)
    generatorSettings.set_value('last-microphone', MICROPHONE_INDEX)

if CAMERA_INDEX == -1:
    CAMERA_INDEX = camera.get_camera()
    generatorSettings.set_value('last-camera', CAMERA_INDEX)

print(f'Microphone: {MICROPHONE_INDEX}')
print(f'Camera: {CAMERA_INDEX}')

# connect to camera
camera = cv2.VideoCapture(CAMERA_INDEX)
frameWidth  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

video_stream = None

WAV_MASK = 'audio_iter_{iter}.wav'
VIDEO_MASK = 'video_iter_{iter}.avi'

video_path = os.path.join(RECORDS_OUT_PATH, VIDEO_MASK.format(iter=iteration))
audio_path = os.path.join(RECORDS_OUT_PATH, WAV_MASK.format(iter=iteration))

video_stream = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (frameWidth, frameHeight))

wf = wave.open (audio_path, "wb")
wf.setnchannels (AUDIO_CHANNELS)
wf.setsampwidth (p.get_sample_size(AUDIO_FORMAT))
wf.setframerate (AUDIO_RATE)

def callback (in_data, frame_count, time_info, status):
    wf.writeframes (in_data)
    return (in_data, pyaudio.paContinue)

font = cv2.FONT_HERSHEY_COMPLEX
isStarted = False

microphone_stream = None

while True:
    ret, frame = camera.read()

    if not isStarted:
        cv2.putText(frame, 'Press enter to start recording', (50, frameHeight - 50), font, 1, (0, 255, 255), 1)

    if isStarted:
        video_stream.write(frame)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    key = cv2.waitKey(delay=1)
    if key == 27:
        break
    elif key == 13:
        if not isStarted:
            isStarted = True
            microphone_stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                                       channels = wf.getnchannels(),
                                       rate = wf.getframerate(),
                                       input=True,
                                       stream_callback=callback)
            microphone_stream.start_stream()

if microphone_stream is not None:
    microphone_stream.stop_stream()
    microphone_stream.close()

wf.close()
p.terminate()

video_stream.release()
camera.release()

if isStarted:
    generatorSettings.set_value('last-iteration', iteration+1)

print ("* recording done!")

# Close all windows
cv2.destroyAllWindows()