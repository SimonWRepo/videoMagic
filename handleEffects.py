import cv2
import Constants

class HandleEffects:

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
        for flameNumber in range(1, 9):
            self.flame.append(cv2.imread(f"sprites\\flame{flameNumber}.png", cv2.IMREAD_UNCHANGED))

    def loadLightningImages(self):
        for lightningNumber in range(1, 6):
            self.lightning.append(cv2.imread(f"sprites\\lightning{lightningNumber}.png", cv2.IMREAD_UNCHANGED))

    def mergeImage(self, back, front, x, y, alpha):
        if back.shape[2] == 3:
            back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
        if front.shape[2] == 3:
            front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

        bh, bw = back.shape[:2]
        fh, fw = front.shape[:2]
        x1, x2 = max(x, 0), min(x + fw, bw)
        y1, y2 = max(y, 0), min(y + fh, bh)
        frontCropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
        backCropped = back[y1:y2, x1:x2]

        alphaFront = frontCropped[:, :, 3:4] / 255 * alpha
        alphaBack = backCropped[:, :, 3:4] / 255

        result = back.copy()
        result[y1:y2, x1:x2, :3] = alphaFront * frontCropped[:, :, :3] + (1 - alphaFront) * backCropped[:, :, :3]
        result[y1:y2, x1:x2, 3:4] = (alphaFront + alphaBack) / (1 + alphaFront * alphaBack) * 255

        return result

    def gunChecker(self, mostRecentPredictions, finishingAction, gunActive, gunEnded, justShot, count):
        gunCount = sum(1 for s in mostRecentPredictions if s == Constants.FINGER_GUN)
        endGunCount = sum(1 for s in finishingAction if s == Constants.UPWARDS_FINGER_GUN)

        if gunActive:
            if endGunCount < 4:
                gunEnded = 0 if gunEnded > 2 else gunEnded + 1
                gunActive = gunEnded <= 2

        justShot[count % 50] = 1 if gunCount > 12 else 0
        if 1 in justShot and endGunCount > 2:
            gunActive = True

        return gunActive, gunEnded, gunCount, endGunCount

    def snapChecker(self, mostRecentPredictions, finishingAction):
        self.snapCount = sum(1 for pos in mostRecentPredictions if pos == Constants.START_CLICK)
        self.endSnapCount = sum(1 for pos in finishingAction if pos == Constants.END_CLICK)

        if self.snapActive and self.endSnapCount < 4:
            self.snapActive = False
        elif self.snapCount > 6 and self.endSnapCount > 3:
            self.snapActive = True

    def apartChecker(self, mostRecentPredictions, finishingAction):
        self.apartCount = sum(1 for s in mostRecentPredictions if s == Constants.HANDS_TOGETHER)
        self.endApartCount = sum(1 for s in finishingAction if s == Constants.HANDS_APART)

        if self.apartActive and self.endApartCount < 4:
            self.apartActive = False
        elif self.apartCount > 8 and self.endApartCount > 2:
            self.apartActive = True

    def snapDrawer(self, image, count, thumbX, thumbY, zeroY):
        overlay = image.copy()
        alpha = self.endSnapCount / Constants.HOLD_END_CLICK
        image = self.flame[count % 8]
        imgOffset = int(abs(thumbY - zeroY) * 0.5)
        image = cv2.resize(image, (imgOffset, imgOffset), interpolation=cv2.INTER_AREA)
        return self.mergeImage(overlay, image, thumbX - int(imgOffset * 0.5), thumbY - imgOffset, alpha)

    def apartDrawer(self, image, count, indexBaseX, indexBaseY, indexBaseXSecond, indexBaseYSecond, baseY):
        overlay = image.copy()
        alpha = self.endApartCount / Constants.HOLD_END_CLICK
        image = self.lightning[count % 5]
        imgOffset = int(abs(indexBaseY - baseY) * 0.5)
        image = cv2.resize(image, (abs(indexBaseX - indexBaseXSecond + 1), imgOffset), interpolation=cv2.INTER_AREA)

        minX = min(indexBaseX, indexBaseXSecond)
        avgY = int((indexBaseY + indexBaseYSecond) / 2)
        return self.mergeImage(overlay, image, minX, avgY, alpha)