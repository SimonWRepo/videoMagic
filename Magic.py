import cv2
import mediapipe as mp
import numpy as np
import sklearn
from matplotlib import pyplot as plt
from numpy import sqrt, square
from sklearn.preprocessing import StandardScaler
import csv
import pandas as pd
from sklearn.decomposition import PCA

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
click1 = [0.05989882389735636, -0.08207983528801958, 0.08227919559538949, -0.1532171918072946, 0.11412131664618848, -0.20369223401359784, 0.14861043055649084, -0.2480432395448568, -0.04771569002295643, -0.22626168176235845, 0.07200137144525953, -0.25725370078521603, 0.11293940416794439, -0.21268874183297512, 0.12455328933543898, -0.17110083483928204]
click2 = [0.027631604741618337, -0.08027480111958724, 0.05013913795600728, -0.14962232293200053, 0.09196686742800453, -0.18930409934689377, 0.13110945482090525, -0.21271233961653263, -0.0036493004788155006, -0.2132306360095482, 0.10011357334105359, -0.2499530842040199, 0.1381265502911328, -0.21695111476721196, 0.14644060288932229, -0.18690041772639795]
click3 = [0.07101092123209361, -0.08144251943071767, 0.0943466932946996, -0.15659560095533223, 0.11999831310859542, -0.21025458074830666, 0.15113836992782448, -0.25334230994669116, -0.024600566075961828, -0.21518483173793732, 0.07744003046007668, -0.25378859019743055, 0.12692851420615778, -0.22706132952530078, 0.14961376413140762, -0.20295471894623546]
null1= [0.07623438932269491, -0.015703191975629204, 0.14198989003681692, -0.06274493310596456, 0.17751483310240043, -0.10796092239262485, 0.1909152707988129, -0.13027636836500095, 0.06801370574765962, -0.17518909384693784, 0.0822774306902408, -0.23856243141326627, 0.0755642446386457, -0.2427316298928054, 0.07506877928677842, -0.2521549628191228]

real2 = [0.03181149875968187, 0.0013236917235170556, 0.03241887391504103, 0.005113737217109912, 0.020556393499099986, 0.015363686361036123, 0.014687871078407437, 0.02979531157544959, 0.03231535233237002, 0.009556100218727517, 0.020615614723582306, 0.005353785492877161, 0.018470573823671503, 0.010539897641038819, 0.018377681517043668, 0.02335951501176584]
avg = [0.05284711662368944, -0.08126571861277483, 0.07558834228203211, -0.1531450385648758, 0.10869549906092947, -0.20108363803626608, 0.14361941843507353, -0.23803262970269354, -0.02532185219257792, -0.21822571650328135, 0.08318499174879661, -0.25366512506222216, 0.125998156221745, -0.21890039537516262, 0.14020255211872298, -0.18698532383730515]
sub = []
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
nfit = [click1] + [click2] + [null1]


neigh.fit(nfit, [1,1,0])
# open the file in the write mode
f = open('C:\\Users\\Simon\\Documents\\Personal\\videoMagic\csv_file.csv', 'a',newline="")

# create the csv writer
writer = csv.writer(f)

cap = cv2.VideoCapture(0)
count = 0
import tensorflow as tf

from keras.models import load_model
model = load_model('keypoint_classifier.hdf5')

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    addUpAll = np.zeros(42)
    snapConsistent = np.zeros(10)
    while cap.isOpened():
        count = count + 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        if not results.multi_hand_landmarks:
            continue
        if count%10==0:
            counti = 0
            addUp = np.full(84,-100)
            zeroCoords = np.array([])
            nineCoords = np.array([])
            pointerCoords = np.zeros(100)
            for hand_landmarks in results.multi_hand_landmarks:
                if counti < 30:
                    for i in range(21):
                        if i==0:
                            id0x = hand_landmarks.landmark[i].x * image_width
                            id0y = hand_landmarks.landmark[i].y * image_height
                            zeroCoords = np.append(zeroCoords,(hand_landmarks.landmark[0].x * image_width,hand_landmarks.landmark[0].y * image_height))
                            nineCoords = np.append(nineCoords,(hand_landmarks.landmark[9].x * image_width,hand_landmarks.landmark[9].y * image_height))
                            #pointerCoords[count%100] = (hand_landmarks.landmark[8].x * image_width,hand_landmarks.landmark[8].y * image_height)
                        addUp[counti*2] = hand_landmarks.landmark[i].x * image_width - id0x
                        addUp[counti*2+1] = (hand_landmarks.landmark[i].y * image_height - id0y)
                        counti += 1
            if len(zeroCoords) == 2:
                diff = -100
            print(len(zeroCoords))
            if len(zeroCoords) == 4:
                #how the x and y coords of the 0 positions of both hands are layed out in the np array
                x = abs(zeroCoords[0] - zeroCoords[2])
                y = abs(zeroCoords[1] - zeroCoords[3])
                nineDiffX = abs(zeroCoords[0] - nineCoords[0])
                nineDiffY = abs(zeroCoords[1] - nineCoords[1])
                nineDiff = sqrt(square(nineDiffY) + square(nineDiffX))

                diff = sqrt(square(x) + square(y))
                print(diff)
                print(nineDiff)
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
            addUp = interest
            #0 is start click
            #1 is null
            #2 is open
            #3 is end click
            #4 is halfclosed
            #5 is fingergun
            #6 is fingergun up (ending)
            #7 is 2 hands apart
            #8 is 2 hands together
            #9 is roiling hand
            #10 is one finer
            addUp = np.append(addUp,[diff])
            addUp = np.append(addUp,[10])
            if addUp2[8] == -100:
                writer.writerow(addUp)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # write a row to the csv file




        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
            # close the file
    f.close()






cap.release()