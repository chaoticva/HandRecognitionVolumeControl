import math

import cv2
import mediapipe as mp


class HandDetection:
    def __init__(self, max_hands=2, colorize=False, draw_lines=False):
        self.max_hands = max_hands
        self.colorize = colorize
        self.draw_lines = draw_lines
        self.lmList = []

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=max_hands)
        self.mpDraw = mp.solutions.drawing_utils

    def draw(self, _img):
        img_rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                if self.draw_lines:
                    self.mpDraw.draw_landmarks(_img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

                for _id, landmark in enumerate(handLandmarks.landmark):
                    h, w, c = _img.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    if self.colorize:
                        if _id in (4, 8, 12, 16, 20):
                            cv2.circle(_img, (cx, cy), 5, (0, 106, 255), cv2.FILLED)
                        if _id in (3, 7, 11, 15, 19):
                            cv2.circle(_img, (cx, cy), 5, (0, 106, 100), cv2.FILLED)
                        if _id in (2, 6, 10, 14, 18):
                            cv2.circle(_img, (cx, cy), 5, (100, 106, 255), cv2.FILLED)
                        if _id in (1, 5, 9, 13, 17):
                            cv2.circle(_img, (cx, cy), 5, (100, 106, 0), cv2.FILLED)
                        if _id == 0:
                            cv2.circle(_img, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
                    else:
                        cv2.circle(_img, (cx, cy), 5, (255, 255, 0))
                        cv2.circle(_img, (cx, cy), 4, (0, 106, 255), cv2.FILLED)

        return _img

    def get_landmark_position(self, hand=0, img=None, draw=False):
        x_list = []
        y_list = []
        bbox = []
        self.lmList = []
        results = self.hands.process(img)
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[hand]

            if self.draw:
                self.mpDraw.draw_landmarks(img, my_hand, self.mpHands.HAND_CONNECTIONS)

            for _id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lmList.append([_id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingers_up(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] > self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, pos1, pos2, img, draw=False, radius=15, t=3):
        x1, y1 = self.lmList[pos1][1:]
        x2, y2 = self.lmList[pos2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), radius, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        if length < 30:
            cv2.circle(img, (cx, cy), radius, (0, 255, 0), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]
