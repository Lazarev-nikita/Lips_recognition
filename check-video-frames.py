import pydub
import pathlib
import cv2
import numpy as np
import dlib
import collections
import math
from modules import settings

import os.path


RECORDS_PATH = './split-result.01'

generatorSettings = settings.Settings('record-settings.json')
SYMBOLS = generatorSettings.get_value('symbols')

WIDTH = 128
HEIGHT = 128

files_list = os.listdir(RECORDS_PATH)

for f in files_list:
    sym, ext = f.split('.')
    src_video = os.path.join(RECORDS_PATH, f)
    src_cap = cv2.VideoCapture(src_video)
    srcFrameCount = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_cap.release()
    print(f'{sym}: {f} frames = {srcFrameCount}')
