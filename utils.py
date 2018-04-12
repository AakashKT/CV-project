import cv2
import scipy
import numpy as np


def klDivergenceList(referenceImage, images):
    referenceHist = cv2.calcHist(
        images=[referenceImage],
        channels=[0, 1, 2],
        mask=None,
        histSize=[256, 256, 256],
        ranges=[0, 256, 0, 256, 0, 256, ]
    )
    referenceHist = referenceHist/np.size(referenceImage)
    klDivList = []
    for image in images:
        hist = cv2.calcHist(
            images=[image],
            channels=[0, 1, 2],
            mask=None,
            histSize=[256, 256, 256],
            ranges=[0, 256, 0, 256, 0, 256, ]
        )
        hist = hist/np.size(image)
        klDiv = scipy.stats.entropy(pk=referenceHist, qk=hist, base=2)
        klDivList.append(klDiv)
    return klDivList
