import os
import cv2

def print_progress_percent(last_value, curr, total, step):
    percent = int(100 * curr / total)
    if (percent % step == 0) and last_value != percent:
        print(percent, '%')
    return percent

SOURCE_FOLDER = 'C:\\Projects\\my\\python\\lips_data'
#SOURCE_FOLDER = '/content/drive/My Drive/lips_data'
SOURCE_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video'
DEST_VIDEO_FOLDER = SOURCE_FOLDER + '/crop-video-2'

files_list = os.listdir(SOURCE_VIDEO_FOLDER)

WIDTH = 128
HEIGHT = 128

min_frames_count = 28
for idx, fname in enumerate(files_list):
    src_path = SOURCE_VIDEO_FOLDER + "\\" + fname
    dst_path = DEST_VIDEO_FOLDER + "/" + fname
    cap = cv2.VideoCapture(src_path)

    if not cap.isOpened():
        print(f'Video {src_path} not found, skip it')
        cap.release()
        continue

    crop_stream = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (WIDTH, HEIGHT), 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = frame.shape
        grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayFrame, (WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)
        crop_stream.write(resized)

    crop_stream.release()
    cap.release()

    print(f'File {idx+1} of {len(files_list)} handled ({100 * (idx+1) / len(files_list)}): {src_path}')

