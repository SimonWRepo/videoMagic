import cv2
import Constants

class handleEffects:

    def __init__(self):
        self.flame = []
        self.lightning = []
        self.snapActive = False
        self.apartActive = False
        self.snapCount = 0
        self.endSnapCount = 0
        self.apartCount = 0
        self.endApartCount = 0

    def loadFlameImages(self):
        #get all images for flame animation
        for flameNumber in range(1,9):
            self.flame.append(cv2.imread("sprites\\flame" + str(flameNumber) + ".png",cv2.IMREAD_UNCHANGED))

    def loadLightningImages(self):
        #images for lightning animation
        for x in range(1,6):
            self.lightning.append(cv2.imread("sprites\\lightning" + str(x) + ".png",cv2.IMREAD_UNCHANGED))

    def merge_image(self, back, front, x,y,alpha):

        # convert to rgba
        if back.shape[2] == 3:
            back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
        if front.shape[2] == 3:
            front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

        # crop the overlay from both images
        bh,bw = back.shape[:2]
        fh,fw = front.shape[:2]
        x1, x2 = max(x, 0), min(x+fw, bw)
        y1, y2 = max(y, 0), min(y+fh, bh)
        front_cropped = front[y1-y:y2-y, x1-x:x2-x]
        back_cropped = back[y1:y2, x1:x2]

        alpha_front = front_cropped[:,:,3:4] / 255 * alpha
        alpha_back = back_cropped[:,:,3:4] / 255

        # replace an area in result with overlay
        result = back.copy()
        result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
        result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

        return result

    def gunChecker(self, mostRecentPredictions, finishingAction, gunActive, gunEnded, justShot, count):
        gunCount = 0
        endGunCount = 0

        for s in mostRecentPredictions:
            if s == Constants.FINGER_GUN:
                gunCount = gunCount+1

        for s in finishingAction:
            if s == Constants.UPWARDS_FINGER_GUN:
                endGunCount += 1

        if gunActive == True:
            if endGunCount<4:
                if gunEnded>2:
                    gunActive=False
                    gunEnded=0
                else:
                    gunEnded += 1
            else:
                gunEnded = 0

        if gunCount > 12:
            justShot[count%50] = 1
        else:
            justShot[count%50] = 0

        if 1 in justShot:
            if endGunCount > 2:
                gunActive = True

        return gunActive, gunEnded, gunCount, endGunCount

    def snapChecker(self, mostRecentPredictions, finishingAction):
        self.snapCount = 0
        self.endSnapCount = 0

        for handPosition in mostRecentPredictions:
            if handPosition == Constants.START_CLICK:
                self.snapCount += 1
        for handPosition in finishingAction:
            if handPosition == Constants.END_CLICK:
                self.endSnapCount += 1
        if self.snapActive == True:
            if self.endSnapCount<4:
                self.snapActive=False

        if self.snapCount > 6:
            if self.endSnapCount > 3:
                self.snapActive = True

    def apartChecker(self, mostRecentPredictions, finishingAction):
        self.apartCount = 0
        self.endApartCount = 0

        for s in mostRecentPredictions:
            if s == Constants.HANDS_TOGETHER:
                self.apartCount += 1
        for s in finishingAction:
            if s == Constants.HANDS_APART:
                self.endApartCount += 1
        if self.apartActive == True:

            if self.endApartCount<4:
                self.apartActive=False

        if self.apartCount > 8:
            if self.endApartCount > 2:
                self.apartActive = True

    def snapDrawer(self, image, count, thumbXCoordinate, thumbYCoordinate, zeroY):
        overlay = image.copy()
        alpha = self.endSnapCount/Constants.HOLD_END_CLICK
        image = self.flame[count%8]
        imgOffset = int(abs(thumbYCoordinate - zeroY) * 0.5)
        image = cv2.resize(image, (imgOffset,imgOffset), interpolation = cv2.INTER_AREA)
        image = self.merge_image(overlay, image, thumbXCoordinate - int(imgOffset * 0.5), thumbYCoordinate - imgOffset, alpha)
        return image

    def apartDrawer(self, image, count, indexBaseXCoordinate, indexBaseYCoordinate, indexBaseXCoordinateSecondHand, indexBaseYCoordinateSecondHand, baseYCoordinate):
        overlay = image.copy()
        alpha = self.endApartCount / Constants.HOLD_END_CLICK
        image = self.lightning[count%5]
        imgOffset = int(abs(indexBaseYCoordinate - baseYCoordinate) * 0.5)

        image = cv2.resize(image, (abs((indexBaseXCoordinate+1)-indexBaseXCoordinateSecondHand),imgOffset), interpolation = cv2.INTER_AREA)
        if indexBaseXCoordinate < indexBaseXCoordinateSecondHand:
            image = self.merge_image(overlay,image,indexBaseXCoordinate,int((indexBaseYCoordinate+indexBaseYCoordinateSecondHand)/2),alpha)
        else:
            image = self.merge_image(overlay,image,indexBaseXCoordinateSecondHand,int((indexBaseYCoordinate+indexBaseYCoordinateSecondHand)/2),alpha)
        return image