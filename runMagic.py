import cv2
import mediapipe as mp
import numpy as np
import sklearn

from numpy import sqrt, square
from sklearn.preprocessing import StandardScaler
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from sklearn.neighbors import KNeighborsClassifier, KDTree

import pandas as pd

from handleEffects import *
import Constants

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
segment = SelfiSegmentation()



def main():

    neighbours = KNeighborsClassifier(n_neighbors=10)
    closeOrApart = KNeighborsClassifier(n_neighbors=5)
    nfit = pd.read_csv("csv_file.csv",skip_blank_lines=True)
    nfit.head()

    #get dataset
    x, y = nfit.iloc[:, :-1], nfit.iloc[:, [-1]]
    xCloseOrApart = nfit.iloc[:, Constants.NUMBER_OF_LANDMARKS]
    xCloseOrApart = xCloseOrApart.values.reshape(-1,1)

    kd = KDTree(x)
    closeOrApart.fit(xCloseOrApart, y.values.ravel())
    neighbours.fit(x.values, y.values.ravel())

    cap = cv2.VideoCapture(0)

    #number of frames actually processed (half of frames shown)
    count = 0

    handleEffectsInstance = HandleEffects()
    handleEffectsInstance.loadFlameImages()

    handleEffectsInstance.loadLightningImages()

    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:

        #what has the prediction been in the last few frames?
        mostRecentPredictions = np.ones(Constants.MOST_RECENT_PREDICTIONS)

        shownImage = np.empty(0)

        finishingAction = np.zeros(Constants.HOLD_END_CLICK)

        #we only want to show every other frame, otherwise the project lags
        showCurrentFrame = False
        while cap.isOpened():

            #flip showCurrentFrame on/off every frame, so that only half of the frames are shown
            showCurrentFrame = not showCurrentFrame
            success, image = cap.read()

            if not success:
                print("No video found. Is your camera set up properly?")
                # If loading a video, use 'break' instead of 'continue'.
                break

            if not shownImage.any():
                shownImage = image

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            image = cv2.flip(image, 1)
            if showCurrentFrame:
                count += 1
                shownImage = processFrame(handleEffectsInstance, count, image, hands, neighbours, closeOrApart, mostRecentPredictions, finishingAction)

            cv2.imshow('MediaPipe Hands', shownImage)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()

def handleSnapEffect(handleEffectsInstance, hand_landmarks, image, count, image_width, image_height):
    thumbXCoordinate = int(hand_landmarks.landmark[Constants.THUMB_TIP].x * image_width)
    thumbYCoordinate = int(hand_landmarks.landmark[Constants.THUMB_TIP].y * image_height)
    #draw the fire on the hand, just above the thumb
    return handleEffectsInstance.snapDrawer(image, count, thumbXCoordinate , thumbYCoordinate, int(hand_landmarks.landmark[0].y * image_height))

def handleLightningEffect(handleEffectsInstance, multi_hand_landmarks, image, count, image_width, image_height):
    firstHand = multi_hand_landmarks[0]
    secondHand = multi_hand_landmarks[1]

    #get coords of first hand
    indexBaseXCoordinate = int(firstHand.landmark[Constants.INDEX_BASE].x * image_width)
    indexBaseYCoordinate = int(firstHand.landmark[Constants.INDEX_BASE].y * image_height)
    baseYCoordinate = int(firstHand.landmark[Constants.BASE].y * image_height)

    #get coords of second hand
    indexBaseXCoordinateSecondHand = int(secondHand.landmark[Constants.INDEX_BASE].x * image_width)
    indexBaseYCoordinateSecondHand = int(secondHand.landmark[Constants.INDEX_BASE].y * image_height)

    return handleEffectsInstance.apartDrawer(image, count, indexBaseXCoordinate, indexBaseYCoordinate, indexBaseXCoordinateSecondHand, indexBaseYCoordinateSecondHand, baseYCoordinate)

def processFrame(handleEffects, count, image, hands, neighbours, closeOrApart, recentPredictions, finishingAction):
    imageHeight, imageWidth, _ = image.shape
    annotatedImage = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    results = hands.process(annotatedImage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    if not results.multi_hand_landmarks:
        handleNoHands(count, finishingAction, recentPredictions)
        return image

    if count > 0:
        if handleEffects.snapActive:
            for handLandmarks in results.multi_hand_landmarks:
                image = handleSnapEffect(handleEffects, handLandmarks, image, count, imageWidth, imageHeight)

        if handleEffects.apartActive and len(results.multi_hand_landmarks) == 2:
            image = handleLightningEffect(handleEffects, results.multi_hand_landmarks, image, count, imageWidth, imageHeight)

        relativeCoords, indexBase, allLandmarks = findHandLandmarkRelativeCoordinates(results.multi_hand_landmarks, imageHeight, imageWidth)

        diff = calculateHandDistance(relativeCoords, indexBase)
        thumbs = predictHandPosition(allLandmarks, diff, neighbours, closeOrApart)

        image = cv2.putText(
            image, str(thumbs), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA
        )

        recentPredictions[count % Constants.MOST_RECENT_PREDICTIONS] = thumbs
        finishingAction[count % Constants.HOLD_END_CLICK] = thumbs

    handleEffects.snapChecker(recentPredictions, finishingAction)
    handleEffects.apartChecker(recentPredictions, finishingAction)

    return image


def handleNoHands(count, finishingAction, recentPredictions):
    finishingAction[count % Constants.HOLD_END_CLICK] = 100
    finishingAction[(count+1) % Constants.HOLD_END_CLICK] = 100
    recentPredictions[count % Constants.MOST_RECENT_PREDICTIONS] = 100
    recentPredictions[(count+1) % Constants.MOST_RECENT_PREDICTIONS] = 100


def calculateHandDistance(relativeCoords, indexBase):
    if len(relativeCoords) == 2:
        return Constants.DIFFERENCE_VALUE_ONE_HAND

    zx, zy = abs(relativeCoords[0] - relativeCoords[2]), abs(relativeCoords[1] - relativeCoords[3])
    indexTipX, indexTipY = abs(relativeCoords[2] - indexBase[0]), abs(relativeCoords[3] - indexBase[1])
    indexTipDiff = sqrt(indexTipX**2 + indexTipY**2)
    return sqrt(zx**2 + zy**2) / indexTipDiff


def predictHandPosition(allLandmarks, diff, neighbours, closeOrApart):
    hand1Coords = allLandmarks[:42]
    hand2Coords = allLandmarks[42:]

    normalizedHand1 = sklearn.preprocessing.normalize([hand1Coords])[0]
    normalizedHand2 = (sklearn.preprocessing.normalize([hand2Coords])[0] if hand2Coords[0] != Constants.DIFFERENCE_VALUE_ONE_HAND else hand2Coords)

    features = np.append(normalizedHand1, normalizedHand2)
    features = np.append(features, [diff])
    thumbs = neighbours.predict([features])[0]

    if thumbs in {Constants.HANDS_APART, Constants.HANDS_TOGETHER}:
        thumbs = closeOrApart.predict(np.array([diff]).reshape(-1, 1))[0]

    return thumbs

def findHandLandmarkRelativeCoordinates(multiHandLandmarks, image_height, image_width):

    #coordinates of the base each hand (point at bottom of palm)
    relativeCoordinate = np.array([])
    #array of all of the landmarks for processing
    relativeCoordinatesOfAllLandmarks = np.full(Constants.NUMBER_OF_LANDMARKS,Constants.DIFFERENCE_VALUE_ONE_HAND)
    indexBaseCoordinates = np.array([])
    totalLandmarksProcessed = 0
    for handNumber in range(min(len(multiHandLandmarks), 2)):
        hand_landmarks = multiHandLandmarks[handNumber]
        relativeCoordinate = np.append(relativeCoordinate,(hand_landmarks.landmark[Constants.BASE].x * image_width,hand_landmarks.landmark[Constants.BASE].y * image_height))
        indexBaseCoordinates = (hand_landmarks.landmark[Constants.INDEX_BASE].x * image_width,hand_landmarks.landmark[Constants.INDEX_BASE].y * image_height)
        baseXCoordinate = hand_landmarks.landmark[Constants.BASE].x * image_width
        baseYCoordinate = hand_landmarks.landmark[Constants.BASE].y * image_height

        for landmarkNumber in range(Constants.NUMBER_OF_LANDMARKS_ON_EACH_HAND):
            #add the x and y coordinates of the specific hand landmark to the list
            relativeCoordinatesOfAllLandmarks[totalLandmarksProcessed*2] = hand_landmarks.landmark[landmarkNumber].x * image_width - baseXCoordinate
            relativeCoordinatesOfAllLandmarks[totalLandmarksProcessed*2+1] = (hand_landmarks.landmark[landmarkNumber].y * image_height - baseYCoordinate)
            totalLandmarksProcessed += 1

    return relativeCoordinate, indexBaseCoordinates, relativeCoordinatesOfAllLandmarks

if __name__ == "__main__":
    main()