import sched, time
import mediapipe as mp
import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandDetection:
    """
    hand_detection=HandDetection()
    hand_detection.findHands()
    """

    def __init__(self, open_camera=True, mode=False, maxHands=2, modelComp=1, detectionCon=0.5, trackCon=0.5):
        self.wCam, self.hCam = 1280, 720
        s = sched.scheduler(time.time, time.sleep)
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.open_camera = open_camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.volRange = self.volume.GetVolumeRange()

        self.minVol = self.volRange[0]
        self.maxVol = self.volRange[1]
        self.vol = 0
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.open_camera = open_camera
        self.wait = 0
        self.tıpIds = [4, 8, 12, 16, 20]

    def findHands(self, draw=True):

        while (self.open_camera == True):
            self.lmList = []
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            if self.results.multi_hand_landmarks:
                if len(self.results.multi_hand_landmarks) > 0:
                    for i in range(len(self.results.multi_hand_landmarks)):
                        if (self.results.multi_hand_landmarks):
                            self.myHand = self.results.multi_hand_landmarks[i]
                            # for handLMS in self.results.multi_hand_landmarks:
                            if draw:
                                for id, lm in enumerate(self.myHand.landmark):
                                    h, w, c = img.shape
                                    cx, cy = int(lm.x * w), int(lm.y * h)
                                    self.lmList.append([id, cx, cy])
                                    # cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                                    self.mpDraw.draw_landmarks(img, self.myHand,
                                                               self.mpHands.HAND_CONNECTIONS)
                        if (len(self.lmList)) != 0:
                            fingers = []
                            if (self.lmList[self.tıpIds[0]][1] > self.lmList[self.tıpIds[0] - 1][1]):
                                fingers.append(1)
                            else:
                                fingers.append(0)
                            for id in range(1, 5):
                                if (self.lmList[self.tıpIds[id]][2] < self.lmList[self.tıpIds[id] - 2][2]):
                                    fingers.append(1)
                                else:
                                    fingers.append(0)
                            thmbx = self.lmList[4][1]
                            thmby = self.lmList[4][2]
                            indexx = self.lmList[8][1]
                            indexy = self.lmList[8][2]
                            secondx = self.lmList[12][1]
                            secondy = self.lmList[12][2]
                            thirdx = self.lmList[16][1]
                            thirdy = self.lmList[16][2]
                            fourthx = self.lmList[20][1]
                            fourthy = self.lmList[20][2]

                            cv2.circle(img, (thmbx, thmby), 15, (255, 255, 0), cv2.FILLED)
                            cv2.circle(img, (indexx, indexy), 15, (255, 255, 0), cv2.FILLED)
                            cv2.circle(img, (secondx, secondy), 15, (255, 255, 0), cv2.FILLED)
                            cv2.circle(img, (thirdx, thirdy), 15, (255, 255, 0), cv2.FILLED)
                            cv2.circle(img, (fourthx, fourthy), 15, (255, 255, 0), cv2.FILLED)
                            if len(self.lmList) > 21:
                                thmb2x = self.lmList[25][1]
                                thmb2y = self.lmList[25][2]
                                index2x = self.lmList[29][1]
                                index2y = self.lmList[29][2]
                                second2x = self.lmList[33][1]
                                second2y = self.lmList[33][2]
                                third2x = self.lmList[37][1]
                                third2y = self.lmList[37][2]
                                fourth2x = self.lmList[41][1]
                                fourth2y = self.lmList[41][2]
                                cv2.circle(img, (thmb2x, thmb2y), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (index2x, index2y), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (second2x, second2y), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (third2x, third2y), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (fourth2x, fourth2y), 15, (255, 255, 0), cv2.FILLED)

                                # Ses açıp kapama özelliği
                                """mx, my = (thmbx + indexx) // 2, (thmby + indexy) // 2
                                cv2.circle(img, (thmbx, thmby), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (indexx, indexy), 15, (255, 255, 0), cv2.FILLED)
                                cv2.circle(img, (mx, my), 15, (255, 255, 0), cv2.FILLED)
                                cv2.line(img, (thmbx, thmby), (indexx, indexy), (255, 0, 255))
                                length = math.hypot(indexx - thmbx, indexy - thmby)
                                self.vol=np.interp(length,[50,300],[self.minVol,self.maxVol])
                                self.volume.SetMasterVolumeLevel(self.vol,None)"""
            cv2.imshow("Hand", img)
            cv2.waitKey(1)

# hand_detection = HandDetection()
# hand_detection.findHands()
