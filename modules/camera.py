import device

def get_camera():
    cameraIndex = None
    cameras = {}
    device_list = device.getDeviceList()
    index = 0
    for camera in device_list:
        cameras[index] = camera[0]
        index += 1

    if len(cameras) == 0:
        print('No camera found. Exit')
    elif len(cameras) == 1:
        cameraIndex = cameras.keys()[0]
    else:
        while (cameraIndex is None):
            print('Choose camera to use or -1 to exit:')
            for c in cameras:
                print(str(c) + ': ' + cameras[c])
            cameraIndex = int(input())
            if cameraIndex > -1:
                if cameraIndex in cameras.keys():
                    break
                else:
                    cameraIndex = None
            else:
                cameraIndex = None
                break

    return cameraIndex
