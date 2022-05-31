import easyocr
import cv2


def getTextFromImage(imgPath,gpu=False):
    reader = easyocr.Reader(['en'], gpu=gpu)
    Docs = list()
    documents = list()
    finalDocs = list()
    result = reader.readtext(imgPath)
    Docs.append(result)
    for content in Docs:
        tmpList = list()
        for text in content:
            tmpList.append(text[1])

        documents.append(tmpList)


    for d in documents:
        finalDocs.append(' '.join(i for i in d))

    return finalDocs, Docs

def getBestScore(Docs,threshold = 0):
    confidence = list()
    for content in Docs:
        for text in content:
            confidence.append((text[1],text[2]))

    best = list()
    counter = 0
    for text, score in confidence:
        if score >= threshold:
            counter += 1
            best.append((text,score))
    return (best,counter,len(confidence))

