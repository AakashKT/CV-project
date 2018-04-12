import cv2
import numpy as np
import utils


def GetApperanceEnergy(referenceImage, images):
    klDivList = utils.klDivergenceList(
        referenceImage=referenceImage, images=images)
    enumeratedKlDivList = list(enumerate(klDivList))
    sortedEnumeratedList = sorted(enumeratedKlDivList, key=lambda x: x[1])
    appearanceCost = [0 for _ in range(len(images))]
    for i in range(len(sortedEnumeratedList)):
        appearanceCost[sortedEnumeratedList[i][0]] = (i+1)/len(images)
    return appearanceCost
