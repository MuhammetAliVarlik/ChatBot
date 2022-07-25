import cv2
import mediapipe as mp


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

    def findFace(self, draw=True, opencam=1, cont=1):
        """
        draw:Bbox'un çizilip çizilmeyeceğini belirler
        opencam: Kameranın çıktı verip vermeyeceğini belirler
        cont: Kameranın sürekli çalışıp çalışmayacağını belirler
        """
        if cont == 0:
            while True:
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
                if opencam == 0:
                    cv2.imshow("Image", img)
                    cv2.waitKey(1)
        elif cont==1:
            bboxs = []
            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.face.process(imgRGB)
            if self.results.detections:
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
            if opencam == 0:
                cv2.imshow("Image", img)
                cv2.waitKey(1)
            return self.results

#print(FaceDetection().findFace(opencam=0, cont=1))
