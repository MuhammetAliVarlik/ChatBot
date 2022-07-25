import mediapipe as mp
import cv2

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
                        lmList.append([id, cx, cy])
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
                        self.mpDraw.draw_landmarks(img, myPart,
                                                   self.mpPose.POSE_CONNECTIONS)
            cv2.imshow("Pose", img)
            cv2.waitKey(1)
# PoseDetection().findPose()