import cv2
import scipy
import numpy as np

from scipy.stats import entropy


class Image:
    window = 11

    @staticmethod
    def SobelImage(image):
        image = image/256.0
        kSize = 3
        sobelx = cv2.Sobel(src=image, dx=1, dy=0,
                           ddepth=cv2.CV_64F, ksize=kSize)
        sobelx = np.absolute(sobelx)
        sobelx = np.sum(a=sobelx, axis=2)
        sobely = cv2.Sobel(src=image, dx=0, dy=1,
                           ddepth=cv2.CV_64F, ksize=kSize)
        sobely = np.absolute(sobely)
        sobely = np.sum(a=sobely, axis=2)
        sobel = sobelx+sobely
        return sobel

    @staticmethod
    def Constrast(image, window):
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        image = image/256.0
        kSize = 3
        sobelx = cv2.Sobel(src=image, dx=1, dy=0,
                           ddepth=cv2.CV_64F, ksize=kSize)
        sobelx = np.multiply(sobelx, sobelx)
        sobelx = 1-sobelx
        sobely = cv2.Sobel(src=image, dx=0, dy=1,
                           ddepth=cv2.CV_64F, ksize=kSize)
        sobely = np.multiply(sobely, sobely)
        sobely = 1-sobely
        sobel = sobelx+sobely

        padded = np.zeros(np.array(sobel.shape)+window)
        padded[:-window, :-window] = sobel
        sobelWindowSum = padded
        sobelWindowSum = np.cumsum(a=sobelWindowSum, axis=0)
        sobelWindowSum[window:, :] = sobelWindowSum[window:, :] - \
            sobelWindowSum[:-window, :]
        sobelWindowSum = np.cumsum(a=sobelWindowSum, axis=1)
        sobelWindowSum[:, window:] = sobelWindowSum[:, window:] - \
            sobelWindowSum[:, :-window]

        sobelWindowSum = sobelWindowSum[window/2:-window/2, window/2:-window/2]
        sobelWindowSum = np.sqrt(sobelWindowSum)
        temp = np.ones(shape=sobel.shape)
        padded = np.zeros(np.array(temp.shape)+window)
        padded[:-window, :-window] = temp
        temp = padded
        tempWindowSum = np.cumsum(a=temp, axis=0)
        tempWindowSum[window:, :] = tempWindowSum[window:, :] - \
            tempWindowSum[:-window, :]
        tempWindowSum = np.cumsum(a=tempWindowSum, axis=1)
        tempWindowSum[:, window:] = tempWindowSum[:, window:] - \
            tempWindowSum[:, :-window]
        tempWindowSum = tempWindowSum[window/2:-window/2, window/2:-window/2]

        constrast = sobelWindowSum/tempWindowSum
        return constrast

    def __init__(self, image, cameraPose, kLDivergenceEnergy):
        self.image = image
        self.cameraPose = cameraPose
        self.sobel = Image.SobelImage(image)
        self.kLDivergenceEnergy = kLDivergenceEnergy
        self.contrast = Image.Constrast(image, Image.window)
        pass


def klDivergenceList(referenceImage, images):
    referenceHist = cv2.calcHist(
        images=[referenceImage],
        channels=[0, 1, 2],
        mask=None,
        histSize=[256, 256, 256],
        ranges=[0, 256, 0, 256, 0, 256, ]
    )
    e=0.001
    referenceHist = referenceHist + (referenceHist==0)*e
    referenceHist = referenceHist/np.sum(referenceHist)
    klDivList = []
    for image in images:
        hist = cv2.calcHist(
            images=[image],
            channels=[0, 1, 2],
            mask=None,
            histSize=[256, 256, 256],
            ranges=[0, 256, 0, 256, 0, 256, ]
        )
        hist = hist + (hist==0)*e
        hist  =hist/np.sum(hist)
        klDiv = entropy(pk=referenceHist.flatten(), qk=hist.flatten(), base=2)
        klDivList.append(klDiv)
    return klDivList
