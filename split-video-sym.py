import pydub
import threading
import os
import pathlib
import datetime
import cv2
import numpy as np
import dlib
import pyaudio
import wave
import time
import sys
import collections
import math
from modules import settings
from modules import camera

import os.path
import json

def get_segments(audio_path: str):
    wav = pydub.AudioSegment.from_wav(audio_path)
    segments = pydub.silence.detect_nonsilent(wav, min_silence_len=400, silence_thresh = wav.dBFS-10)
    print(f'segments found = {len(segments)}; {segments}')

    #get average duration:
    avg_count = len(segments)
    avg_sum = 0
    start_segment = 0
    out_segments = list()
    for s in segments:
        if s[0] == 0:
            start_segment += 1
            avg_count -= 1
            print(f'skip noise segment at begin: {s}')
            continue
        out_segments.append((s[0], s[1], s[1] - s[0]))
        avg_sum += (s[1] - s[0])
    avg_dur = int(math.ceil(avg_sum / avg_count))

    return {'avg': avg_dur, 'segments': out_segments}


RECORDS_PATH = './result-data'
SPLIT_RECORDS_OUT_PATH = './split-result'

FRAMES_CROP_COUNT = 5

pathlib.Path(SPLIT_RECORDS_OUT_PATH).mkdir(parents=True, exist_ok=True)

generatorSettings = settings.Settings('record-settings.json')
SYMBOLS = generatorSettings.get_value('symbols', [])

# prepare face points detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Projects\\my\\python\\lips_data\\shape-predictor\\shape_predictor_68_face_landmarks.dat")

points = list()
points.extend(range(2, 15))     # bottom part of face contour
points.extend(range(30, 36))    # bottom part of nose
points.extend(range(48, 68))    # bottom part of nose

WIDTH = 128
HEIGHT = 128

files_list = os.listdir(RECORDS_PATH)

source_data = collections.defaultdict(dict)
for f in files_list:
    sym, ext = f.split('.')
    source_data[sym][ext] = f

num = 1

print('Collect audio segments for input files')
sum_avg = 0
count_avg = 0
for sym in source_data:
    print(f'search for segments in file {num} of {len(source_data)}: {sym}')
    pair = source_data[sym]
    src_audio = os.path.join(RECORDS_PATH, pair['wav'])

    audio_segments = get_segments(src_audio)
    print(f'Average segment duration: {audio_segments["avg"]}')
    source_data[sym]['avg'] = audio_segments['avg']
    source_data[sym]['segments'] = audio_segments['segments']
    sum_avg += audio_segments['avg']
    count_avg += 1
    num += 1

# {
#     "БА": {
#         "avi": "БА.avi",
#         "wav": "БА.wav",
#         "avg": 123,
#         "segments": [(s, f, l), (s, f, l), ()]
#     }
# }

avg = sum_avg / count_avg
print(f'Average segment duration over all files: {avg}')

X_left_crop = -1
X_right_crop = -1
Y_left_crop = -1
Y_right_crop = -1

num = 0
for sym in source_data:
    print(f'split video file {num} of {len(source_data)}: {sym}')
    sym_data = source_data[sym]
    src_video = os.path.join(RECORDS_PATH, pair['avi'])
    src_cap = cv2.VideoCapture(src_video)
    srcFrameCount = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx, s in enumerate(sym_data['segments']):  # (start, end, duration)
        print(f'{sym} - {idx} started')
        s_start = int(s[0] + s[2] / 2 - avg / 2)
        s_end = s_start + avg

        dst_video = os.path.join(SPLIT_RECORDS_OUT_PATH, '{symbol}_{iter}.avi'.format(symbol=sym, iter=str(idx).zfill(2)))
        crop_stream = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (WIDTH, HEIGHT))

        # seach for frames diapasone:
        src_cap.set(cv2.CAP_PROP_POS_MSEC, s_end)
        frmIndexEnd = int(src_cap.get(cv2.CAP_PROP_POS_FRAMES))
        frmIndexStart = frmIndexEnd - FRAMES_CROP_COUNT + 1
        if frmIndexStart < 0:
            frmIndexStart = 0

        print(f'Start crop from index {frmIndexStart}')
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, frmIndexStart)
        croppedFrames = 0
        while croppedFrames < FRAMES_CROP_COUNT:
            ret, frame = src_cap.read()
            if not ret:
                break
            croppedFrames += 1

            h, w, c = frame.shape
            grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = detector(grayFrame)
            if len(faces) > 0: #face detected
                face = faces[0]
                # Create landmark object
                landmarks = predictor(image=grayFrame, box=face)
                marks = np.zeros((2, len(points)))
                # Loop through all the points
                co = 0
                # Specific for the mouth.
                for n in points:
                    p = landmarks.part(n)
                    A = (p.x, p.y)
                    marks[0, co] = p.x
                    marks[1, co] = p.y
                    co += 1

                frameRect = (int(np.amin(marks, axis=1)[0]),
                             int(np.amin(marks, axis=1)[1]),
                             int(np.amax(marks, axis=1)[0]),
                             int(np.amax(marks, axis=1)[1]))
                frameCenter = ((frameRect[0] + frameRect[2]) // 2, (frameRect[1] + frameRect[3]) // 2)
                size = (frameRect[2] - frameRect[0], frameRect[3] - frameRect[1])
                sz = max(size)
                X_left_crop = int(frameCenter[0] - sz // 2)
                if X_left_crop < 0:
                    X_left_crop = 0
                X_right_crop = X_left_crop + sz
                if X_right_crop >= w:
                    X_right_crop = w-1
                    X_left_crop = X_right_crop - sz

                Y_left_crop = int(frameCenter[1] - sz // 2)
                if Y_left_crop < 0:
                    Y_left_crop = 0
                Y_right_crop = Y_left_crop + sz
                if Y_right_crop >= h:
                    Y_right_crop = h-1
                    Y_left_crop = Y_right_crop - sz

            if X_left_crop >= 0 and X_right_crop >= 0 and Y_left_crop >= 0 and Y_right_crop >= 0:
                mouth = frame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop]
                resized = cv2.resize(mouth, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
                crop_stream.write(resized)

        print(f'{dst_video} frames: {croppedFrames}')
        crop_stream.release()

    num += 1