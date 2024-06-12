import modules.settings as settings
import os
import numpy as np
import dlib
import cv2

IMAGES_OUT_PATH = 'result-images'


generatorSettings = settings.Settings('generator-settings.json')
symbolsVideoMap = generatorSettings.get_value('generated-video')

# prepare face points detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

totalVideoCount = 0
currVideoIndex = 0


print('Count videos...')
for symbol in symbolsVideoMap:
    iterations: {} = symbolsVideoMap[symbol]
    totalVideoCount += len(iterations)
print(f'found {totalVideoCount} videos')

points = list()
points.extend(range(2, 15))     # bottom part of face contour
points.extend(range(30, 36))    # bottom part of nose
points.extend(range(48, 68))    # bottom part of nose

generatedImagesMap = {}
cropMap = {}
maxCropSize = (None, None)
symbolIndex = 0
minFramesCount = None
print('Detect crop size and crop areas...')
for symbol in symbolsVideoMap:
    iterations: {} = symbolsVideoMap[symbol]
    for iter in iterations:
        path = iterations[iter]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f'Video {path} not found, skip it')
            cap.release()
            continue
        print(f'Check video {(currVideoIndex+1)} of {totalVideoCount}: {path}')

        framesCropCenterMap = {}

        # search max crop size for video
        size = (0, 0)
        frameIdx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = detector(grayFrame)
            for face in faces:
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
                size = (max(size[0], frameRect[2] - frameRect[0]), max(size[1], frameRect[3] - frameRect[1]))
                if maxCropSize[0] is None:
                    maxCropSize = (size[0] + 20, size[1] + 20)
                else:
                    maxCropSize = (max(maxCropSize[0], size[0] + 20), max(maxCropSize[1], size[1] + 20))
                framesCropCenterMap[frameIdx] = frameCenter
                frameIdx += 1
        cropMap[path] = framesCropCenterMap
        currVideoIndex += 1
        ######

print('Crop and split videos...')

currVideoIndex = 0
for symbol in symbolsVideoMap:
    generatedImagesMap[symbol] = {}
    iterations: {} = symbolsVideoMap[symbol]
    for iter in iterations:
        generatedImagesMap[symbol][iter] = []
        path = iterations[iter]

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f'Video {path} not found, skip it')
            cap.release()
            continue
        print(f'Check video {(currVideoIndex+1)} of {totalVideoCount}: {path}')

        framesCropCenterMap = cropMap[path]

        print(f'Split video {(currVideoIndex+1)} of {totalVideoCount}: {path}')
        img_path = IMAGES_OUT_PATH + '/' + str(symbolIndex).zfill(3) + '/' + str(iter).zfill(2)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        currVideoIndex += 1

        # extract lips from vide into separate frames using found crop size
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f'Video {path} not found, skip it')
            cap.release()
            continue

        frameIdx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            grayFrame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            frameCenter = framesCropCenterMap[frameIdx]
            X_left_crop = int(frameCenter[0] - maxCropSize[0] // 2)
            X_right_crop = int(frameCenter[0] + maxCropSize[0] // 2)
            if X_left_crop < 0:
                X_right_crop += (0 - X_left_crop)
                X_left_crop = 0
            Y_left_crop = int(frameCenter[1] - maxCropSize[1] // 2)
            Y_right_crop = int(frameCenter[1] + maxCropSize[1] // 2)
            if Y_left_crop < 0:
                Y_right_crop += (0 - Y_left_crop)
                Y_left_crop = 0

            mouth = grayFrame[Y_left_crop:Y_right_crop, X_left_crop:X_right_crop]

            img_full_path = img_path + '/' + 'frame' + '_' + str(frameIdx).zfill(3) + '.png'
            cv2.imwrite(img_full_path, mouth)
            generatedImagesMap[symbol][iter].append(img_full_path)
            frameIdx += 1

        cap.release()
        if minFramesCount is None:
            minFramesCount = len(generatedImagesMap[symbol][iter])
        else:
            minFramesCount = min(minFramesCount, len(generatedImagesMap[symbol][iter]))
    symbolIndex += 1

for symbol in generatedImagesMap:
    symbolItersMap: {} = generatedImagesMap[symbol]
    for iter in symbolItersMap:
        iterPaths: [] = symbolItersMap[iter]
        fromFirst = True
        img = None
        while minFramesCount < len(iterPaths):
            if fromFirst:
                img = iterPaths.pop(0)
                fromFirst = False
            else:
                img = iterPaths.pop(len(iterPaths)-1)
                fromFirst = True
            os.remove(img)

generatorSettings.set_value('generated-images', generatedImagesMap)

