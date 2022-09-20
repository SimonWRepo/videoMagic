import cv2

def merge_image(back, front, x,y,alpha):
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
def gunChecker(handConsistent, handEnd, gunActive, gunEnded, justShot, count):
    gunCount = 0
    endGunCount = 0
    for s in handConsistent:
        if s == 5:
            gunCount = gunCount+1
    for s in handEnd:
        if s == 6:
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

def drawChecker(handConsistent, handEnd, drawActive):
    handEnded = 0
    drawCount = 0

    for s in handConsistent:
        if s == 10:
            drawCount = drawCount+1
    if drawActive == True:
        if drawCount<4:
            drawActive=False

    if drawCount > 3:
        drawActive = True
    print(drawCount, handEnded)
    return drawActive,drawCount
def snapChecker(handConsistent, handEnd, snapActive):
    handEnded = 0
    snapCount = 0
    endSnapCount = 0

    for s in handConsistent:
        if s == 0:
            snapCount = snapCount+1
    for s in handEnd:
        if s == 3:
            endSnapCount += 1
    if snapActive == True:
        if endSnapCount<4:
                snapActive=False

    if snapCount > 6:
        if endSnapCount > 3:
            snapActive = True
    print(snapCount,endSnapCount, handEnded)
    return snapActive,snapCount, endSnapCount

def apartChecker(handConsistent, handEnd, apartActive):
    apartCount = 0
    endApartCount = 0

    for s in handConsistent:
        if s == 8:
            apartCount = apartCount+1
    for s in handEnd:
        if s == 7:
            endApartCount += 1
    if apartActive == True:

        if endApartCount<4:
                apartActive=False

        else:
            handEnded = 0
    if apartCount > 8:
        if endApartCount > 2:
            apartActive = True
    return apartActive, apartCount, endApartCount

def snapDrawer(image, endSnapCount, flame, count, imgX,imgY, zeroY):
    overlay = image.copy()
    alpha = endSnapCount/7
    image = flame[count%8]
    imgOffset = int(abs(imgY - zeroY) * 0.5)
    image = cv2.resize(image, (imgOffset,imgOffset), interpolation = cv2.INTER_AREA)
    image = merge_image(overlay,image,imgX-int(imgOffset*0.5),imgY-imgOffset,alpha)
    return image

def apartDrawer(image, endApartCount, lightning, count, imgX,imgY, imgX2,imgY2,zeroY):
    overlay = image.copy()
    alpha = endApartCount/7
    image = lightning[count%5]
    imgOffset = int(abs(imgY - zeroY) * 0.5)
    print(imgY)
    print(zeroY)
    print(imgX)
    print(imgX2)
    image = cv2.resize(image, (abs((imgX+1)-imgX2),imgOffset), interpolation = cv2.INTER_AREA)
    if imgX < imgX2:
        image = merge_image(overlay,image,imgX,int((imgY+imgY2)/2),alpha)
    else:
        image = merge_image(overlay,image,imgX2,int((imgY+imgY2)/2),alpha)
    return image