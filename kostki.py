from time import sleep
from skimage import io, draw, exposure, filters, color, measure, morphology, transform
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import numpy as np
import functools


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
    srodkioczek = []
    for n, contour in enumerate(contours):
        numberOfPointsInContour = len(contour)
        if 50 < numberOfPointsInContour < 500:
            sumaoczek += 1
            plt.plot(contour[:, 1], contour[:, 0], linewidth=3)
            """centroidx = np.sum(contour[:, 1]) / len(contour)
            centroidy = np.sum(contour[:, 0]) / len(contour)
            srodkioczek.append([centroidx, centroidy])
            #print(srodkioczek[-1])

    odlegloscimiedzyoczkami = []
    for i in range(len(srodkioczek)):
        for j in range(i + 1, len(srodkioczek)):
            odlegloscimiedzyoczkami.append(getDistance(srodkioczek[i][0], srodkioczek[j][0], srodkioczek[i][1], srodkioczek[j][1]))
    minimalnaodleglosc = 10000
    for odleglosc in odlegloscimiedzyoczkami:
        if odleglosc < minimalnaodleglosc:
            minimalnaodleglosc = odleglosc
    przydzieloneoczka = [False] * len(srodkioczek)
    kostki = []
    for i in range(len(srodkioczek)):
        if not przydzieloneoczka[i]:
            kostki.append(1)
            przydzieloneoczka[i] = True
            for j in range(i + 1, len(srodkioczek)):
                if not przydzieloneoczka[j] and getDistance(srodkioczek[i][0], srodkioczek[j][0], srodkioczek[i][1], srodkioczek[j][1]) < 3 * minimalnaodleglosc:
                    kostki[-1] = kostki[-1] + 1
                    przydzieloneoczka[j] = True
                    if kostki[-1] == 6:
                        break"""
    print("Suma oczek:", sumaoczek)
    #kostki = sorted(kostki)
    #for i in range(len(kostki)):
        #kostki[i] = str(kostki[i])
    return sumaoczek
    #return sumaoczek, ", ".join(kostki)


def processImage(dice):

    if (len(dice)) > 1000:
        dice = transform.resize(dice, (1000, int(dice.shape[1] * 1000 / dice.shape[0])))
    # io.imshow(dice)
    # plt.show()

    greyscale = color.rgb2gray(dice)
    greyscale = filters.gaussian(greyscale, 1.25)
    greyscaleEdges = filters.sobel(greyscale)
    greyscaleEdges = exposure.rescale_intensity(greyscaleEdges)
    # io.imshow(greyscaleEdges)
    # plt.show()

    blackWhite = greyscaleEdges > 0.14
    # io.imshow(blackWhite)
    # plt.show()
    blackWhite = morphology.remove_small_objects(blackWhite, 50)
    drawCircularContours(blackWhite, 2.5)
    # io.imshow(blackWhite)
    # plt.show()
    blackWhite = ndi.binary_fill_holes(blackWhite)
    blackWhite = morphology.remove_small_objects(blackWhite, 150)
    #suma, countedResult = countResult(blackWhite)
    suma = countResult(blackWhite)
    fullResult = "Suma oczek: " + str(suma)
    #kosteczki = "Kostki: " + countedResult
    io.imshow(dice)
    plt.text(dice.shape[1] // 2, 0.1 * len(dice), fullResult, fontsize=16,
             bbox={'facecolor': 'white', 'alpha': 0.7}, ha='center')
    #plt.text(dice.shape[1] // 2, 0.9 * len(dice), kosteczki, fontsize=16,
    #         bbox={'facecolor': 'white', 'alpha': 0.7}, ha='center')
    plt.axis('off')
    plt.show()
    # plt.savefig("k0" + str(i) + ".jpg", bbox_inches='tight')
    plt.clf()
    sleep(0.1)


def main():

    message = "Wpisz nazwe pliku z obrazkiem. Plik musi znajdowac sie w folderze 'kostki'. Wpisz q, jesli chcesz wyjsc\n"
    while True:
        userAnswer = input(message)
        if userAnswer == 'q':
            break
        try:
            dice = io.imread("kostki/" + userAnswer + ".jpg")
        except FileNotFoundError:
            print("Bledna nazwa, sprobuj ponownie")
            continue
        processImage(dice)


if __name__ == '__main__':
    main()