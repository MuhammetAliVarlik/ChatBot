# import math
# import tkinter as tk
import os
import sched, time
from bs4 import BeautifulSoup
import requests
import datetime
import mediapipe as mp
import cv2
# import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import trnlp
from chatbot import ChatBot
from multiprocessing import Process

now = datetime.datetime.now()

with  open("datas.txt", "r") as file:
    past_lines = file.readlines()
    if past_lines:
        past = datetime.datetime.strptime(past_lines[0].rstrip("\n"), '%Y-%m-%d %H:%M:%S.%f')


def turkishCh(str):
    turkish_ones = ['Ç', 'ç', 'Ğ', 'ğ', 'ı', 'İ', 'Ö', 'ö', 'Ş', 'ş', 'Ü', 'ü']
    eng_ones = ['C', 'c', 'G', 'g', 'i', 'I', 'O', 'o', 'S', 's', 'U', 'u']
    for specialChar in turkish_ones:
        rep_no = turkish_ones.index(specialChar)
        str = str.replace(specialChar, eng_ones[rep_no])
    return str


class Hava:
    """
    İnput olarak şehir girmemiz yeterli.
    Ntv hava kullanarak bugünkü en yüksek ve en düşük sıcaklıkları tespit ediyoruz
    5 günlük hava durumunu dictionary olarak veriyor

    hava_durumu=Hava("izmir")       Hava classı Obje tanımlama
    hava_durumu.sehir.upper()       Şehir ismi
    hava_durumu.weather[2][0]       5 günlük hava durumu derece
    hava_durumu.weather[2][1]       5 günlük hava durumu hava olayı
    hava_durumu.weather[0][0]       yarın
    hava_durumu.weather[5][0]       5 gün sonra
    """

    def __init__(self, sehir):
        self.weather = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        self.sehir = sehir
        if (now - past).seconds >= 3600 or sehir.split() != past_lines[1].split():
            self.URL = "https://www.ntvhava.com/".format(turkishCh(sehir).lower())
            self.page = requests.get(self.URL)
            self.soup = BeautifulSoup(self.page.content, "html.parser")
            self.h_temp = self.soup.find("div", attrs={"class": "search-content-degree-big"}).text.strip()
            self.l_temp = self.soup.find("div", attrs={"class": "search-content-degree-small"}).text.strip()
            self.text = self.soup.find("div", attrs={"class": "search-content-degree-text"}).text.strip()
            self.five_day = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for key in self.five_day:
                self.five_day[key] = ["{}{}".format(self.soup.find_all("div", attrs={
                    "class": "daily-report-tab-content-pane-item-box-bottom-degree-big"})[key].text.strip(),
                                                    self.soup.find_all("div", attrs={
                                                        "class": "daily-report-tab-content-pane-item-box-bottom"
                                                                 "-degree-small"})[
                                                        key].text.strip()),
                                      self.soup.find_all("div",
                                                         attrs={"class": "daily-report-tab-content-pane-item-text"})[
                                          key].text.strip()]
            with  open("datas.txt", "w") as file:
                file.write("{}\n".format(now))
                file.write("{}\n".format(sehir))
                for i in self.five_day:
                    file.write("{}\n".format(self.five_day[i]))
                file.close()
            self.weather = self.five_day
        elif ((now - past).seconds < 3600 or sehir.split() == past_lines[1].split()):
            self.five_days_nc = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
            for i in self.five_days_nc:
                if (i != 0 or i != 1 and i + 2 < len(self.five_days_nc)):
                    self.five_days_nc[i] = past_lines[i + 2].replace('[', ' ').replace(']', ' ').replace("'",
                                                                                                         "").replace(
                        ', ', ',').strip().split(',')
            self.weather = self.five_days_nc


class Vikipedi:
    def __init__(self, aranan):
        self.URL = "https://tr.wikipedia.org/wiki/{}".format(turkishCh(aranan).lower())
        self.page = requests.get(self.URL)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        # self.h_temp = self.soup.find("div", attrs={"class": "search-content-degree-big"}).text.strip()


class Open:
    """
    open_smthg=Open()
    open_smthg.help()
    open_smthg.start("Spotify")
    """

    def __init__(self):
        self.app_list = os.listdir("C:/Users/Administrator/Desktop")

    def start(self, app_name):
        app_name = "{}.lnk".format(app_name)
        self.app_index = self.app_list.index(app_name)
        os.startfile("C:/Users/Administrator/Desktop/{}".format(self.app_list[self.app_index]))

    def help(self):
        a = 0
        print("Açabileceğim uygulamalar:")
        print(self.app_list)
        for i in self.app_list:
            a += 1
            print(a, ")", i.strip(".lnk"))


class HandDetection():
    """
    hand_detection=HandDetection()
    hand_detection.findHands()
    """

    def __init__(self, open_camera=True, mode=False, maxHands=1, modelComp=1, detectionCon=0.5, trackCon=0.5):
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

    def findHands(self, draw=True, handNo=0):

        while (self.open_camera == True):
            self.lmList = []
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            if (self.results.multi_hand_landmarks):
                self.myHand = self.results.multi_hand_landmarks[handNo]
                # for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(self.myHand.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.lmList.append([id, cx, cy])
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
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
                indexx = self.lmList[5][1]
                indexy = self.lmList[5][2]
                cv2.circle(img, (thmbx, thmby), 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (indexx, indexy), 15, (255, 255, 0), cv2.FILLED)

                """thmbx=self.lmList[4][1]
                thmby=self.lmList[4][2]
                indexx = self.lmList[8][1]
                indexy = self.lmList[8][2]
                mx,my=(thmbx+indexx)//2,(thmby+indexy)//2
                cv2.circle(img, (thmbx, thmby), 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (indexx, indexy), 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (mx, my), 15, (255, 255, 0), cv2.FILLED)
                cv2.line(img,(thmbx,thmby),(indexx,indexy),(255,0,255))
                length=math.hypot(indexx-thmbx,indexy-thmby)
                self.vol=np.interp(length,[50,300],[self.minVol,self.maxVol])
                self.volume.SetMasterVolumeLevel(self.vol,None)"""
            cv2.imshow("Hand", img)
            cv2.waitKey(1)


class PoseDetection():
    def __init__(self, open_camera=True, mode=False, complexity=1,
                 smoothLandmarks=True, enableSegmentation=False, smoothSegmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.open_camera = open_camera
        self.cap = cv2.VideoCapture(0)
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smoothLandmarks,
                                     self.enableSegmentation, self.smoothSegmentation, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.open_camera = open_camera

    def findPose(self, draw=True, partNo=0):
        lmList = []
        while (self.open_camera == True):
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            if (self.results.pose_landmarks):
                myPart = self.results.pose_landmarks
                # for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(myPart.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(id, cx, cy)
                        lmList.append([id, cx, cy])
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                        self.mpDraw.draw_landmarks(img, myPart,
                                                   self.mpPose.POSE_CONNECTIONS)
            cv2.imshow("İmage", img)
            cv2.waitKey(1)


class FaceDetection():
    """
    face_detection=FaceDetection().findFace()
    face_detection
    """

    def __init__(self, open_camera=True, minDetectionConfidence=0.5, modelSelection=1):
        self.minDetectionConfidence = minDetectionConfidence
        self.modelSelection = modelSelection
        self.open_camera = open_camera
        self.cap = cv2.VideoCapture(0)
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.minDetectionConfidence, self.modelSelection)
        self.mpDraw = mp.solutions.drawing_utils
        self.open_camera = open_camera

    def findFace(self, draw=True,opencam=1):
        bboxs = []
        success, img = self.cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        if (self.results.detections):
            myPart = self.results.detections
            # for handLMS in self.results.multi_hand_landmarks:
            if draw:
                for id, lm in enumerate(myPart):
                    bboxC = lm.location_data.relative_bounding_box
                    h, w, c = img.shape
                    bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
                    bboxs.append([id, bbox, lm.score])
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, f'{int(lm.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                    x, y, w, h = bbox
                    x1, y1 = x + w, y + h
                    t = 5
                    l = 30
                    # Top Left
                    cv2.line(img, (x, y), (x + l, y), (255, 255), t)
                    cv2.line(img, (x, y), (x, y + l), (255, 255), t)
                    # Top Right
                    cv2.line(img, (x1, y), (x1 - l, y), (255, 255), t)
                    cv2.line(img, (x1, y), (x1, y + l), (255, 255), t)
                    # Bottom Left
                    cv2.line(img, (x, y1), (x + l, y1), (255, 255), t)
                    cv2.line(img, (x, y1), (x, y1 - l), (255, 255), t)
                    # Bottom Right
                    cv2.line(img, (x1, y1), (x1 - l, y1), (255, 255), t)
                    cv2.line(img, (x1, y1), (x1, y1 - l), (255, 255), t)
        if opencam==0:
            cv2.imshow("Image", img)
            cv2.waitKey(1)


class Responses(FaceDetection,ChatBot):

    def giveResponse(self):
        while True:
            sehir="ankara"
            FaceDetection.findFace(self)
            if (self.results.detections):
                tag,response,inp=ChatBot().chat()
                if tag=="selamlama2":
                    if 'günler'in inp.split():
                        response=response.replace("+", "günler")
                    elif 'akşamlar'in inp.split():
                        response=response.replace("+", "akşamlar")
                    elif 'geceler'in inp.split():
                        response=response.replace("+", "geceler")
                    elif 'günaydın'in inp.split():
                        response=response.replace("+", "günler")
                    elif 'tünaydın'in inp.split():
                        response=response.replace("+", "öğlenler")
                    else:
                        response = response.replace("+", "günler")
                if tag=="hava durumu1":
                    if 'bugün' in inp.split() or 'bugünün' in inp.split() or 'bugünkü' in inp.split():
                        response=response.replace("-", "bugün")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response=response.replace("+",hava_durumu.weather[0][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[0][0].split()[2])
                        response=response.replace("?", hava_durumu.weather[0][1])
                    elif 'yarın' in inp.split() or 'yarının'in inp.split() or 'yarınki'in inp.split():
                        print("nope")
                        response=response.replace("-", "yarın")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response = response.replace("+", hava_durumu.weather[1][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[1][0].split()[2])
                        response = response.replace("?", hava_durumu.weather[1][1])
                    else:
                        response = response.replace("-", "bugün")
                        hava_durumu = Hava(sehir)
                        hava_durumu.sehir.upper()
                        response = response.replace("+", hava_durumu.weather[0][0].split()[0])
                        response = response.replace("*", hava_durumu.weather[0][0].split()[2])
                        response = response.replace("?", hava_durumu.weather[0][1])
                print(response)








if past < now:
    time_delta = now - past

Responses().giveResponse()

