from time import sleep
from skimage import io, draw, exposure, filters, color, measure, morphology, transform
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import numpy as np


def getDistance(x1, x2, y1, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def drawCircularContours(blackWhite, tolerance):
    contours = measure.find_contours(blackWhite, 0.5, "high")
    blackWhite[:, :] = 0
    for n, contour in enumerate(contours):
        numberOfPointsInContour = len(contour)
        if 50 < numberOfPointsInContour < 500:

            centroidx = np.sum(contour[:, 0]) / numberOfPointsInContour
            centroidy = np.sum(contour[:, 1]) / numberOfPointsInContour
            # print(centroidx, centroidy)

            distancesToCentroid = []
            sumOfDistances = 0

            for point in contour:
                distanceFromPointToCenter = getDistance(point[0], centroidx, point[1], centroidy)
                distancesToCentroid.append(distanceFromPointToCenter)
                sumOfDistances += distanceFromPointToCenter
            avgDistanceToCentroid = sumOfDistances / numberOfPointsInContour

            deviation = 0
            maximalSingleDeviation = 0

            for i in range(numberOfPointsInContour):
                if abs(avgDistanceToCentroid - distancesToCentroid[i]) > maximalSingleDeviation:
                    maximalSingleDeviation = abs(avgDistanceToCentroid - distancesToCentroid[i])
                deviation += abs(avgDistanceToCentroid - distancesToCentroid[i])
            deviation /= numberOfPointsInContour
            # print(deviation)

            if deviation < tolerance and maximalSingleDeviation < 0.4 * avgDistanceToCentroid:
                rr, cc = draw.polygon(contour[:, 1], contour[:, 0])
                blackWhite[cc, rr] = 1
                # plt.plot(centroidy, centroidx, "bo", markersize="1")
            else:
                rr, cc = draw.polygon(contour[:, 1], contour[:, 0])
                blackWhite[cc, rr] = 0


def countResult(image):
    contours = measure.find_contours(image, 0.5, "high")
    sumaoczek = 0
    for n, contour in enumerate(contours):
        numberOfPointsInContour = len(contour)
        if 50 < numberOfPointsInContour < 500:
            sumaoczek += 1
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
    print("Suma oczek:", sumaoczek)
    return str(sumaoczek)

def main():

    for i in range(5):
        dice = io.imread("kostki/dice" + str(i // 10) + str(i % 10) + ".jpg")
        # print(len(dice))

        if (len(dice)) > 1000:
            dice = transform.resize(dice, (1000, int(dice.shape[1] * 1000 / dice.shape[0])))
        #io.imshow(dice)
        #plt.show()

        greyscale = color.rgb2gray(dice)
        greyscale = filters.gaussian(greyscale, 1.25)
        greyscaleEdges = filters.sobel(greyscale)
        greyscaleEdges = exposure.rescale_intensity(greyscaleEdges)
        #io.imshow(greyscaleEdges)
        #plt.show()

        blackWhite = greyscaleEdges > 0.14
        #io.imshow(blackWhite)
        #plt.show()
        blackWhite = morphology.remove_small_objects(blackWhite, 50)
        drawCircularContours(blackWhite, 2.5)
        #io.imshow(blackWhite)
        #plt.show()
        blackWhite = ndi.binary_fill_holes(blackWhite)
        blackWhite = morphology.remove_small_objects(blackWhite, 150)
        countedResult = countResult(blackWhite)
        fullResult = "Suma oczek: " + countedResult

        io.imshow(dice)
        plt.text(dice.shape[1] // 2, 0.1 * len(dice), fullResult, fontsize=16,
                 bbox={'facecolor': 'white', 'alpha': 0.7}, ha='center')
        plt.axis('off')
        plt.show()
        #plt.savefig("k0" + str(i) + ".jpg", bbox_inches='tight')
        plt.clf();
        sleep(1)

if __name__ == '__main__':
    main()