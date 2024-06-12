import modules.settings as settings
#import modules.camera as camera
# import dlib
import cv2
import datetime


class GeneratorState:
    USER_NOT_READY  = 1
    USER_PREPARE    = 2
    USER_READ       = 3
    DONE            = 4

VIDEO_OUT_PATH = 'result-video_new_00'

appSettings = settings.Settings('app-settings.json')
generatorSettings = settings.Settings('generator-settings_new.json')
#symbolsVideoMap = settings.Settings('generator-video.json')



# request camera index
cameraIndex = 0
# if not appSettings.has_value('last_camera'):
#     cameraIndex = camera.get_camera()
# else:
#     cameraIndex = appSettings.get_value('last_camera')
#
# if cameraIndex is None:
#     exit(0)
# else:
#     appSettings.set_value('last_camera', cameraIndex)


# prepare face points detector
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

# connect to camera
camera = cv2.VideoCapture(cameraIndex)
frameWidth  = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
frameHeight = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`


font = cv2.FONT_HERSHEY_COMPLEX

key = None

repeats: int = generatorSettings.get_value('repeats', 10)
symbols: list = generatorSettings.get_value('symbols')
last_symbol: str = generatorSettings.get_value('last-symbol', symbols[0])
iteration: int = generatorSettings.get_value('last-iteration', 0)
generatedVideo: dict = generatorSettings.get_value('generated-video', {})

symbolIndex = symbols.index(last_symbol)

video_stream = None
start_timestamp = None

generatorState = GeneratorState.USER_NOT_READY
while True:
    ret, frame = camera.read()

    if generatorState == GeneratorState.USER_NOT_READY:
        cv2.putText(frame, 'Press enter to start recording', (50, frameHeight - 50), font, 1, (0, 255, 255), 1)
    else:
        curr_time = datetime.datetime.now()
        if start_timestamp is not None:
            delta = curr_time - start_timestamp
            if generatorState == GeneratorState.USER_PREPARE:
                if delta.total_seconds() >= 0.8:
                    generatorState = GeneratorState.USER_READ
                    start_timestamp = datetime.datetime.now()
            elif generatorState == GeneratorState.USER_READ:
                if delta.total_seconds() >= 1:
                    generatorState = GeneratorState.USER_PREPARE
                    start_timestamp = datetime.datetime.now()
                    iteration += 1
                    if iteration >= repeats:
                        iteration = 0
                        symbolIndex += 1
                    generatorSettings.set_value('last-iteration', iteration)

                    if symbolIndex >= len(symbols):
                        generatorState = GeneratorState.DONE
                        break

                    last_symbol = symbols[symbolIndex]
                    generatorSettings.set_value('last-symbol', last_symbol)

        if generatorState == GeneratorState.USER_PREPARE:
            if video_stream is not None:
                video_stream.release()
                video_stream = None

            if start_timestamp is None:
                start_timestamp = datetime.datetime.now()

        elif generatorState == GeneratorState.USER_READ:
            if video_stream is None:
                fname = last_symbol + '_' + str(iteration).zfill(2) + '.avi'
                path = VIDEO_OUT_PATH + '/' + fname
                video_stream = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 30, (frameWidth, frameHeight))
                if last_symbol not in generatedVideo:
                    generatedVideo[last_symbol] = {}
                generatedVideo[last_symbol][str(iteration)] = path

            video_stream.write(frame)

        if generatorState == GeneratorState.USER_PREPARE:
            cv2.putText(frame, symbols[symbolIndex], (frameWidth//2, frameHeight - 50), font, 0.6, (0, 255, 0), 1)
        elif generatorState == GeneratorState.USER_READ:
            cv2.putText(frame, symbols[symbolIndex], (frameWidth//2, frameHeight - 50), font, 1, (0, 0, 255), 2)

    cv2.putText(frame, 'Press ESC to exit', (50, frameHeight - 20), font, 0.7, (0, 255, 255), 1)

    # show the image
    cv2.imshow(winname="Face", mat=frame)

    # Exit when escape is pressed
    key = cv2.waitKey(delay=1)
    if key == 27:
        break
    elif key == 13:
        if generatorState == GeneratorState.USER_NOT_READY:
            generatorState = GeneratorState.USER_PREPARE

# When everything done, release the video capture and video write objects
if video_stream is not None:
    video_stream.release()
    video_stream = None
camera.release()

generatorSettings.set_value("generated-video", generatedVideo)

# Close all windows
cv2.destroyAllWindows()