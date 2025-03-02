
#Indicies representing parts of the hand in mediapipe
BASE = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_BASE = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

#Labels given to different hand positions in the dataset
START_CLICK = 0
NO_POSITION = 1
OPEN_PALM = 2
END_CLICK = 3
HALF_CLOSED = 4
FINGER_GUN = 5
UPWARDS_FINGER_GUN = 6
HANDS_APART = 7
HANDS_TOGETHER = 8
ROILING_HAND = 9
ONE_FINGER_UP = 10

NUMBER_OF_LANDMARKS = 84
NUMBER_OF_LANDMARKS_ON_EACH_HAND = 20
#The value in the dataset given to the distance between two hands when there is only one hand
DIFFERENCE_VALUE_ONE_HAND = -100
#The number of frames checked for the "end click" action (
HOLD_END_CLICK = 7
#The number of frames checked when predicting what action the user is doing
MOST_RECENT_PREDICTIONS = 15



