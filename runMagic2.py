import time
from numba import vectorize, cuda, jit, njit
import cv2
import mediapipe as mp
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from numpy import sqrt, square
from sklearn.preprocessing import StandardScaler
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from sklearn.neighbors import KNeighborsClassifier, BallTree, KDTree
from keras.models import load_model

import tensorflow as tf

import csv
import pandas as pd
from sklearn.decomposition import PCA
from handleEffects import snapChecker, snapDrawer, gunChecker, apartChecker, apartDrawer, drawChecker

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
segment = SelfiSegmentation()

def main():

    neigh = KNeighborsClassifier(n_neighbors=10)
    closeOrApart = KNeighborsClassifier(n_neighbors=5)
    line_count = 0
    nfit = pd.read_csv("C:\\Users\\Simon\\Documents\\Personal\\videoMagic\csv_file.csv",skip_blank_lines=True)
    nfit.head()
    #get dataset
    x, y = nfit.iloc[:, :-1], nfit.iloc[:, [-1]]
    xCloseApart = nfit.iloc[:, 84]
    xCloseApart = xCloseApart.values.reshape(-1,1)
    print(xCloseApart)
    kd = KDTree(x)
    closeOrApart.fit(xCloseApart, y.values.ravel())
    neigh.fit(x.values, y.values.ravel())

    cap = cv2.VideoCapture(0)

    #frames actually processed (half of count)
    count = 0
    #number of frames in vid so far
    count2 = 0
    flame = []
    lightning = []
    lastThumb = 0
    lastDot = (0,0)
    #images for flame animation
    for x in range(1,9):
        flame.append(cv2.imread("C:\\Users\\Simon\\Documents\\Personal\\videoMagic\\sprites\\flame" + str(x) + ".png",cv2.IMREAD_UNCHANGED))
    #images for lightning animation
    for x in range(1,6):
        lightning.append(cv2.imread("C:\\Users\\Simon\\Documents\\Personal\\videoMagic\\sprites\\lightning" + str(x) + ".png",cv2.IMREAD_UNCHANGED))
    with mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3) as hands:
        lostItCount = 0

        #what has the prediction been in the last 15 frames?
        handConsistent = np.ones(15)
        #""" 7 frames
        handEnd = np.zeros(7)
        #last 50 frames, contains a 1 if there was a snap
        justSnapped = np.ones(50)
        justShot = np.ones(50)
        justDraw = np.ones(50)
        #currently snapping (fire)?
        snapActive = False
        #currently hands apart (lightning)?
        apartActive = False
        drawActive = False
        gunActive = False
        gunEnded = 0
        lastIndexPositionsX = np.array([])
        lastIndexPositionsY = np.array([])
        #thumb coordinates of prev frame (for frame when hand isnt processed)
        thumbCoords = (0,0)
        while cap.isOpened():

            start = time.time()

            count2 = count2 + 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            #only process 1/2 of the frames (laggy otherwise)
            if count2%2==0:
                count = count + 1

                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                #run mediapipe on the image, find the hands
                results = hands.process(annotated_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)


                #if there are any hands in the image
                if results.multi_hand_landmarks:

                    #if it is not the first frame
                    if count>0:
                        #number of landmarks processed in total (all hands)
                        counti = 0
                        #array of all of the landmarks for processing
                        addUp = np.full(84,-100)
                        #coordinates of the zero coordinates for each hand (point at bottom of palm)
                        zCoords = np.array([])
                        nineCoords = np.array([])
                        #if currently snapping..
                        if snapActive == True:
                            #for each hand...
                            for hand_landmarks in results.multi_hand_landmarks:
                                imgx = int(hand_landmarks.landmark[4].x * image_width)
                                imgy = int(hand_landmarks.landmark[4].y * image_height)
                                #draw the fire on the hand, just above the thumb
                                image = snapDrawer(image, endSnapCount, flame, count, imgx , imgy, int(hand_landmarks.landmark[0].y * image_height))
                        #if the hands are apart after being together..
                        if apartActive == True:
                            #check which hand is currently being processed
                            runThroughTwice = 0
                            for hand_landmarks in results.multi_hand_landmarks:
                                #get coords of first hand
                                if runThroughTwice == 0:
                                    imgx = int(hand_landmarks.landmark[9].x * image_width)
                                    imgy = int(hand_landmarks.landmark[9].y * image_height)
                                    zeroy = int(hand_landmarks.landmark[0].y * image_height)
                                    runThroughTwice=1
                                #get coords of second hand
                                else:
                                    if runThroughTwice == 1:
                                        imgx2 = int(hand_landmarks.landmark[9].x * image_width)
                                        imgy2 = int(hand_landmarks.landmark[9].y * image_height)
                                        runThroughTwice=2
                            if runThroughTwice == 2:
                                #if there are 2 hands, draw the lightning effect between them
                                image = apartDrawer(image, endApartCount, lightning, count2, imgx, imgy, imgx2,imgy2,zeroy)
                        runThroughTwice=0
                        for hand_landmarks in results.multi_hand_landmarks:

                            #first hand
                            if runThroughTwice==0:
                                #get relevant coordinates to use for when the hands arent being processed next frame
                                thumbCoords = (int(hand_landmarks.landmark[4].x * image_width),int(hand_landmarks.landmark[4].y * image_height))
                                middleCoords = (int(hand_landmarks.landmark[9].x * image_width),int(hand_landmarks.landmark[9].y * image_height))
                                fingerCoordsX = hand_landmarks.landmark[8].x*image_width
                                fingerCoordsY = hand_landmarks.landmark[8].y*image_height
                                print(fingerCoordsX)
                                zeroy = int(hand_landmarks.landmark[0].y * image_height)

                                runThroughTwice=1
                            else:
                                #second hand
                                if runThroughTwice==1:
                                    thumbCoords2 = (int(hand_landmarks.landmark[4].x * image_width),int(hand_landmarks.landmark[4].y * image_height))
                                    middleCoords2 = (int(hand_landmarks.landmark[9].x * image_width),int(hand_landmarks.landmark[9].y * image_height))
                                    fingerCoords2 = (int(hand_landmarks.landmark[8].x)*image_width,int(hand_landmarks.landmark[8].y)*image_height)
                                    zeroy2 = int(hand_landmarks.landmark[0].y * image_height)
                                    runThroughTwice=2
                            #make sure only 2 hands are processed (if counti is above 30 then the 2nd hand has already been processed)
                            if counti < 30:
                                for i in range(21):
                                    if i==0:
                                        id0x = hand_landmarks.landmark[i].x * image_width
                                        id0y = hand_landmarks.landmark[i].y * image_height
                                        zCoords = np.append(zCoords,(hand_landmarks.landmark[0].x * image_width,hand_landmarks.landmark[0].y * image_height))
                                        nineCoords = (hand_landmarks.landmark[9].x * image_width,hand_landmarks.landmark[9].y * image_height)
                                #add the x and y coordinates of the specific hand landmark to the list
                                    addUp[counti*2] = hand_landmarks.landmark[i].x * image_width - id0x
                                    addUp[counti*2+1] = (hand_landmarks.landmark[i].y * image_height - id0y)
                                    counti += 1
                    #if there's 1 hand, the difference between 2 hands is -100
                    if len(zCoords) == 2:
                        diff = -100
                    #if there are 2 hands, the difference is the distance between them
                    if len(zCoords) == 4:
                        #how the x and y coords of the 0 positions of both hands are layed out in the np array
                        zx = abs(zCoords[0] - zCoords[2])
                        zy = abs(zCoords[1] - zCoords[3])
                        nineDiffX = abs(zCoords[0] - nineCoords[0])
                        nineDiffY = abs(zCoords[1] - nineCoords[1])
                        nineDiff = sqrt(square(nineDiffY) + square(nineDiffX))
                        diff = sqrt(square(zx) + square(zy))
                        diff = diff/nineDiff
                    #run the data on the model to predict the hand position
                    addUp1 = addUp[:42]
                    addUp2 = addUp[42:]
                    #print(addUp2)
                    interest = sklearn.preprocessing.normalize([addUp1])[0]
                    if not addUp2[0] == -100:
                        interest2 = sklearn.preprocessing.normalize([addUp2])[0]
                    else:
                        interest2 = addUp2
                    interest = np.append(interest,interest2)

                    interest = np.append(interest,[diff])
                    thumbs = neigh.predict([interest])[0]
                    print(diff)
                    if thumbs == 7 or thumbs == 8:
                        diff = np.array([diff])
                        updatedThumbs = closeOrApart.predict(diff.reshape(-1,1))[0]
                        thumbs = updatedThumbs
                    lastThumb = thumbs
                    image = cv2.putText(image, str(thumbs), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255,255,255), 1, cv2.LINE_AA)
                    dist, ind = kd.query([interest], k=3)
                    print(runThroughTwice)
                    print(str(thumbs) + " prediction")

                    #add the prediction to this frame's location
                    handConsistent[count%15] = thumbs
                    handEnd[count%7] = thumbs
                    #for the one index finger drawing mode
                    if drawActive == True:
                        if len(lastIndexPositionsX)>99:
                            lastIndexPositionsX = np.roll(lastIndexPositionsX,-1)
                            lastIndexPositionsY = np.roll(lastIndexPositionsY,-1)
                            lastIndexPositionsX[99] = fingerCoordsX
                            lastIndexPositionsY[99] = fingerCoordsY

                        else:
                            lastIndexPositionsX = np.append(lastIndexPositionsX,fingerCoordsX)
                            lastIndexPositionsY = np.append(lastIndexPositionsY,fingerCoordsY)
                        if(len(lastIndexPositionsX)>0):
                            #so a line isn't drawn from the first point to the last of the next frame
                            lastDot = (int(lastIndexPositionsX[0]),int(lastIndexPositionsY[0]))
                        for lx,ly in zip(lastIndexPositionsX,lastIndexPositionsY):
                            pt = (int(lx),int(ly))
                            image = cv2.line(image, lastDot,pt,(0,0,0),1)
                            lastDot = pt

                    else:
                        lastIndexPositionsX = np.array([])
                        lastIndexPositionsY = np.array([])

                else:

                    handEnd[count%7] = 100
                    handEnd[(count+1)%7] = 100
                    handConsistent[count%15] = 100
                    handConsistent[(count+1)%15] = 100
                #check whether a snap has occured
                snapActive, snapCount, endSnapCount = snapChecker(handConsistent, handEnd, snapActive)
                #check whether a draw has occured
                drawActive, drawCount = drawChecker(handConsistent, handEnd, drawActive)
                #check whether 'apart' has occured
                apartActive, apartCount, endApartCount = apartChecker(handConsistent, handEnd, apartActive)
                #gunActive, gunEnded, gunCount, endGunCount = gunChecker(handConsistent, handEnd, gunActive, gunEnded, justShot, count)
            else:
                #if this is an 'off' frame, draw the snap or apart if it should be there
                image = cv2.putText(image, str(lastThumb), (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255,255,255), 1, cv2.LINE_AA)
                #for the one index finger drawing mode
                if drawActive == True:
                    if len(lastIndexPositionsX)>99:
                        lastIndexPositionsX = np.roll(lastIndexPositionsX,-1)
                        lastIndexPositionsY = np.roll(lastIndexPositionsY,-1)
                        lastIndexPositionsX[99] = fingerCoordsX
                        lastIndexPositionsY[99] = fingerCoordsY

                    else:
                        lastIndexPositionsX = np.append(lastIndexPositionsX,fingerCoordsX)
                        lastIndexPositionsY = np.append(lastIndexPositionsY,fingerCoordsY)
                    if(len(lastIndexPositionsX)>0):
                        #so a line isn't drawn from the first point to the last of the next frame
                        lastDot = (int(lastIndexPositionsX[0]),int(lastIndexPositionsY[0]))
                    for lx,ly in zip(lastIndexPositionsX,lastIndexPositionsY):
                        pt = (int(lx),int(ly))
                        image = cv2.line(image, lastDot,pt,(0,0,0),1)
                        lastDot = pt

                else:
                    lastIndexPositionsX = np.array([])
                    lastIndexPositionsY = np.array([])
                if snapActive == True:
                    image = snapDrawer(image, endSnapCount, flame, count, thumbCoords[0] , thumbCoords[1], zeroy)
                if apartActive == True:
                    if runThroughTwice==2:
                        image = apartDrawer(image, endApartCount, lightning, count2, middleCoords[0], middleCoords[1], middleCoords2[0],middleCoords2[1],zeroy)
                #if gunActive == True:
                    #print("hi2")
                    #image = cv2.circle(image,(fingerCoords[0],fingerCoords[1]),5,(0,0,0),-1)

            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break





    cap.release()

if __name__ == "__main__":
    main()
