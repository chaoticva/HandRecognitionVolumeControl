import cv2
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

width, height = 640, 360

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detection = htm.HandDetection(max_hands=1, colorize=False, draw_lines=True)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
volBar = 300
vol = 0

while True:
    success, img = cap.read()

    lmList = detection.get_landmark_position(hand=0, img=img, draw=True)
    if len(lmList[0]) == 21:
        length = detection.find_distance(pos1=lmList[0][4][0], pos2=lmList[0][8][0], img=img, draw=True, radius=8, t=3)
        vol = np.interp(length[0], [15, 140], [min_volume, max_volume])
        volBar = np.interp(length[0], [15, 140], [300, 50])
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 50), (85, 300), (0, 106, 255), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 300), (0, 106, 255), cv2.FILLED)
    cv2.putText(img, str(int(np.interp(vol, [min_volume, max_volume], [0, 100]))), (50, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
