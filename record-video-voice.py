import threading
import os
import pathlib
import datetime
import cv2
import pyaudio
import wave
import time
import sys
# import device

import os.path
import json

class Settings:
    path: str = None
    config: dict = None

    def __init__(self, path):
        self.config = {}
        self.path = path
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.config = json.load(f)

    def get(self):
        return self.config

    def set(self, config):
        if (config is None):
            config = {}

        self.config = config
        self.save()

    def has_value(self, name):
        return name in self.config

    def get_value(self, name: str, def_value: any = None):
        return self.get_value_cfg(self.config, name, def_value)

    def get_value_cfg(self, cfg: dict, name: str, def_value: any = None):
        if name not in cfg:
            cfg[name] = def_value
        return cfg[name]

    def set_value(self, name: str, value: any):
        self.config[name] = value
        self.save()

    def save(self):
        with open(self.path, 'w', encoding="utf-8") as fo:
            fo.write(json.dumps(self.config, indent=2, ensure_ascii=False))


# def get_camera():
#     cameraIndex = None
#     cameras = {}
#     device_list = device.getDeviceList()
#     index = 0
#     for camera in device_list:
#         cameras[index] = camera[0]
#         index += 1
#
#     if len(cameras) == 0:
#         print('No camera found. Exit')
#     elif len(cameras) == 1:
#         cameraIndex = cameras.keys()[0]
#     else:
#         while (cameraIndex is None):
#             print('Choose camera to use or -1 to exit:')
#             for c in cameras:
#                 print(str(c) + ': ' + cameras[c])
#             cameraIndex = int(input())
#             if cameraIndex > -1:
#                 if cameraIndex in cameras.keys():
#                     break
#                 else:
#                     cameraIndex = None
#             else:
#                 cameraIndex = None
#                 break
#
#     return cameraIndex


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

generatorSettings = Settings('record-settings.json')
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
    CAMERA_INDEX = 0 #camera.get_camera()
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