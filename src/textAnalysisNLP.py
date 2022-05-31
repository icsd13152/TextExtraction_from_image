import textExtraction
import nltk
import re
from nltk.probability import FreqDist
# import spacy
# from gtts import gTTS
# import numpy as np
# from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def getText(imgPath,useGpu=False):
    finalListOfDocs, docsWithStats = textExtraction.getTextFromImage(imgPath,useGpu)
    return finalListOfDocs, docsWithStats

def customTokenize(text):
    pattern = re.compile(r'(?:[A-Z]\.)+|[A-Za-z]+\.|\d+(?:[\./]\d+)|n\'t|\b\w+(?!\'t)|\w+(?:-\w+)*|[!\"#$%&\'()*+,./:;<=>?@[\]^_`{|}~]|-{2}')
    tmpTokens = pattern.findall(text)
    tokens = list()
    for i in tmpTokens:
        i = i.strip()
        i = i.replace('/','\\/')
        tokens.append(i)

    return tokens

def getScoresFromOCR(docsWithStats, threshold = 0):
    bestWords, totalBest,total = textExtraction.getBestScore(docsWithStats,threshold = threshold)
    print("With this threshold found ",totalBest, " words from ",total)
    return bestWords

def getPOStaggs(tokens, method = 'NLTK'):
    posTaggs = list()
    if method == 'NLTK':
       posTaggs = nltk.pos_tag(tokens)
    return posTaggs

def customStemming(token): #(\w+?)(?=ly|es|(?<!s)s|y)
    stems = re.findall(r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$',token)
    stem = stems[0][0]
    return stem

def getStemsFromCustom(tokens):
    stems = list()
    for t in tokens:
        stems.append(customStemming(t))
    return stems

def getTopNmostCommonTokens(tokens,num=30):
    topN_1Custom = FreqDist(tokens).most_common(num)
    return topN_1Custom

def getPercentageOfTokensWithFrequencyN(tokens,num = 1):
    diccustom = FreqDist(tokens)
    filter_tokens = dict([(m, n) for m, n in diccustom.items() if n == num])
    observed = len(filter_tokens)/len(diccustom)
    return observed


#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image to be OCR'd")
# ap.add_argument("-l", "--langs", type=str, default="en",
#                 help="comma separated list of languages to OCR")
# ap.add_argument("-g", "--gpu", type=int, default=-1,
#                 help="whether or not GPU should be used")
# args = vars(ap.parse_args())