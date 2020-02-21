from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import imutils
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

#gasesc contururi si le initializez
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# verific daca am contururi
if len(cnts) > 0:

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# le ordonez in ordine inversa, ca sa raman cu cele mai mari

    for c in cnts:
# aproximez numarul de varfuri
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

# daca am 4 varfuri, am gasit foaia de examen
        if len(approx) == 4:
            docCnt = approx
            break

paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))

# Otsu's thresholding method pentru binarizare
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# gasesc contururile, initializez
# lista de contururi care corespund raspunsurilor
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# sortez contururile top-to-bottom, apoi initializez
# numarul total de raspunsuri corecte
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

#iterez raspunsurile
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sortez de la stanga la dreapta
    # initializez indexul raspunsului
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # iterez raspunsurile sortate
    for (j, c) in enumerate(cnts):
        # construiesc o masca pentru raspunsul curent
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # aplic masca
        # numar cati pixeli diferiti de zero exista
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)

        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)

    # initializez culoarea si indexul rasp corect
    color = (0, 0, 255)
    k = ANSWER_KEY[q]

    # verific daca raspunsul curent e corect
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1

    # desenez contur rosu/verde
    bx, by, bw, bh = cv2.boundingRect(cnts[k])
    cv2.rectangle(paper, (bx, by), (bx + bw, by + bh), color, 2)

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)




score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


nume=raw_input('Introduceti numele studentului curent: ')
with open("rezultate.txt", "a") as myFile:
    myFile.write((str(score)+ " " + nume) + '\n')
raw_input()

cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
# cv2.imshow("Edged", edged)
# cv2.imshow("Warped", warped)
cv2.waitKey(0)

#python proiect.py --image images/test_01.png