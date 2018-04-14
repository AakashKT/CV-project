import cv2
import numpy as np
import utils
import alpha_expansion

appearanceEnergy = []
windowSize = 11

ReferenceLabel = 0

"""
@type Images:list[utils.Image]
"""
Images = []

TargetImageSize = (100, 100)

ReprojectedImage = []


alpha1 = 10
alpha2 = 10
alpha3 = 10


def GetApperanceEnergy(referenceImage, images):
    klDivList = utils.klDivergenceList(
        referenceImage=referenceImage, images=images)
    enumeratedKlDivList = list(enumerate(klDivList))
    sortedEnumeratedList = sorted(enumeratedKlDivList, key=lambda x: x[1])
    appearanceCost = [0 for _ in range(len(images))]
    for i in range(len(sortedEnumeratedList)):
        appearanceCost[sortedEnumeratedList[i][0]] = (i)/len(images)
    return appearanceCost


def AppearanceEnergy(pixel, label):
    return Images[label].kLDivergenceEnergy


def ReferenceEnergy(pixel, label):
    k = (pixel[0]+Images[ReferenceLabel].image.shape[0]/2 -
         1), (pixel[1]+Images[ReferenceLabel].image.shape[1]/2-1)
    if (label == ReferenceLabel):
        return 0
    elif (label == -1):
        return 10000
    elif(k[0] >= 0 and k[0] < Images[ReferenceLabel].image.shape[0] and k[1] >= 0 and k[1] < Images[ReferenceLabel].image.shape[1]):
        return 1000000000
    else:
        return 100


def CompressPixelList(pixelList):
    bucket = {}

    plow = np.zeros(2)
    phigh = np.zeros(2)
    p = np.zeros(2)
    color = np.zeros(3)
    appearance = 0
    contrast = 0
    reference = 0
    sobel = 0
    freq = 0
    if len(pixelList)==0:
        return {}
    key = pixelList[0]["label"]
    for pixel in pixelList:
        newKey = pixel["label"]
        if(newKey!=key):
            plow = plow/freq
            phigh = phigh/freq
            p = p/freq
            appearance = appearance/freq
            contrast = contrast/freq
            reference = reference/freq
            sobel = sobel/freq
            reference = reference/freq
            color = color/freq
            low = np.linalg.norm(plow-p)
            high = np.linalg.norm(phigh-p)
            geometry = max(low, high)
            unary = geometry + alpha1*appearance+alpha2*contrast+alpha3*reference
            bucket[key] = {"color": color,
                                "unary": unary, "edge": sobel}

            plow = np.zeros(2)
            phigh = np.zeros(2)
            p = np.zeros(2)
            color = np.zeros(3)
            appearance = 0
            contrast = 0
            reference = 0
            sobel = 0
            freq = 0
            key=newKey

        plow = plow+pixel["pnear_location"]
        phigh = phigh+pixel["pfar_location"]
        p = p+pixel["p_location"]
        oPixel = pixel["original_location"]
        color = color+pixel["color"]
        appearance = appearance + AppearanceEnergy(p, key)
        contrast = contrast + \
            Images[key].contrast[oPixel[0], oPixel[1]]
        reference = reference+ReferenceEnergy(p, key)
        sobel = sobel+Images[key].sobel[oPixel[0], oPixel[1]]
        freq = freq+1
    plow = plow/freq
    phigh = phigh/freq
    p = p/freq
    appearance = appearance/freq
    contrast = contrast/freq
    reference = reference/freq
    sobel = sobel/freq
    reference = reference/freq
    color = color/freq
    low = np.linalg.norm(plow-p)
    high = np.linalg.norm(phigh-p)
    geometry = max(low, high)
    unary = geometry + alpha1*appearance+alpha2*contrast+alpha3*reference
    bucket[key] = {"color": color,
                        "unary": unary, "edge": sobel}
    return bucket




def init(images, ri):
    print "inside init"
    global ReprojectedImage;
    global TargetImageSize
    aEnergy = GetApperanceEnergy(images[ReferenceLabel], images)
    for i in range(len(images)):
        image = images[i]
        klDivergenceEnergy = aEnergy[i]
        Images.append(utils.Image(image=image, cameraPose=None, kLDivergenceEnergy=klDivergenceEnergy))
    TargetImageSize = len(ri), len(ri[1])
    ReprojectedImage = [[{} for _ in range(TargetImageSize[1])] for _ in range(TargetImageSize[0])]
    for i,pixelList in enumerate(ri):
        for j,pixelBucket in enumerate(pixelList):
            ReprojectedImage[i][j] = CompressPixelList(pixelBucket)



def EnergyFunction(u, labelu, v=None, labelv=None):
    if labelu == 0 or labelv == 0:
        return 1000000000
    if v is None:
        labelu = labelu-1
        print u
        print u/TargetImageSize[1],u % TargetImageSize[1]
        pixelBucket = ReprojectedImage[u /
                                       TargetImageSize[1]][u % TargetImageSize[1]]
        if labelu in pixelBucket.keys():
            return pixelBucket[labelu]["unary"]
        else:
            return 1000000000
    else:
        labelu = labelu-1
        labelv = labelv-1
        pixelBucketu = ReprojectedImage[u /
                                        TargetImageSize[1]][u % TargetImageSize[1]]
        pixelBucketv = ReprojectedImage[v /
                                        TargetImageSize[1]][v % TargetImageSize[1]]
        if labelu in pixelBucketu.keys() and labelv in pixelBucketv.keys():
            return pixelBucket[labelu]["sobel"]+pixelBucketv[labelv]["sobel"]
        else:
            return 1000000000


def gridGraph(size):
    edgeList = []
    for i in range(size[0]):
        for j in range(size[1]):
            if (j > 0):
                edgeList.append((i*size[1]+j, i*size[1]+j-1, 1))
            if (i > 0):
                edgeList.append((i*size[1]+j, (i-1)*size[1]+j, 1))
    graph = alpha_expansion.Graph(
        numberOfVertices=size[0]*size[1], edgeList=edgeList, undirected=True)
    return graph


def GetImage(assignment):
    global ReprojectedImage;

    image = np.zeros(shape=(len(ReprojectedImage),
                            len(ReprojectedImage[1]), 3), dtype=np.uint8)
    for i, pixelBucketList in enumerate(ReprojectedImage):
        for j, pixelBucket in enumerate(pixelBucketList):
            assign = assignment[i*len(ReprojectedImage[1])+j]
            if assign == 0:
                image[i, j, :] = 0
            else:
                image[i, j, :] = pixelBucket[assign-1]["color"]

    return image;

def alpha_expansion_call(graph):
    assignment = alpha_expansion.alpha_expansion(
        graph=graph, EnergyFunction=EnergyFunction, numberOfLabels=len(Images)+1, iterations=1);

    return assignment;


if __name__ == "__main__":

    graph = gridGraph(size=TargetImageSize)
    assignment = alpha_expansion.alpha_expansion(
        graph=graph, EnergyFunction=EnergyFunction, numberOfLabels=len(Images)+1, iterations=1)
