import numpy as np
import pickle
from scipy.sparse import csr_matrix, hstack

with open('Dataset_with_shaddah/id_to_letter.pickle', 'rb') as file:
    letterIDs = pickle.load(file)
with open('Dataset_with_shaddah/id_to_diacritic.pickle', 'rb') as file2:
    diacriticIDs = pickle.load(file2)
with open('Dataset_with_shaddah/id_to_word.pickle', 'rb') as file3:
    wordIDs = pickle.load(file3)
with open('Dataset_with_shaddah/word_count.pickle', 'rb') as file4:
    wordCounts = pickle.load(file4)


def to1D(x, y, z, xMax, yMax):
    return (z * xMax * yMax) + (y * xMax) + x


class FeaturesMatrix:
    def __init__(self):
        self.currLetter_tag = {}
        self.prevLetter_currLetter_tag = {}
        self.currLetter_nextLetter_tag = {}
        self.prevTag_tag = {}
        self.prevTag2_prevTag_tag = {}
        self.prevTag_currLetter_tag = {}
        self.currWord_tag = {}
        self.prevWord_currWord_tag = {}
        self.lettersNum = len(letterIDs)
        self.diacriticsNum = len(diacriticIDs)

        frequentWords = [id for id in wordIDs.keys() if wordCounts[wordIDs[id]] > 5]
        self.wordsNum = len(frequentWords)
        self.frequent_to_original_ID = {}
        self.original_to_frequent_ID = {}
        i = 0
        for id in frequentWords:
            self.frequent_to_original_ID[i] = id
            self.original_to_frequent_ID[id] = i

        data = np.array([1])
        row = np.array([0])

        # Create <current letter, tag> feature vector
        for letter in letterIDs.keys():
            for diacritic in diacriticIDs.keys():
                col = np.array([self.diacriticsNum * letter + diacritic])
                self.currLetter_tag[(letter, diacritic)] = \
                    csr_matrix((data, (row, col)), shape=(1, self.lettersNum*self.diacriticsNum), dtype=np.int8)

        # Create <previous letter, current letter, tag> feature
        for prevLetter in letterIDs.keys():
            for currLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    col = np.array([to1D(diacritic, currLetter, prevLetter, self.diacriticsNum, self.lettersNum)])
                    self.prevLetter_currLetter_tag[(prevLetter, currLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(1, self.lettersNum**2 * self.diacriticsNum),
                                   dtype=np.int8)

        # Create <current letter, next letter, tag> feature
        for currLetter in letterIDs.keys():
            for nextLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    col = np.array([to1D(diacritic, nextLetter, currLetter, self.diacriticsNum, self.lettersNum)])
                    self.currLetter_nextLetter_tag[(currLetter, nextLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(1, self.lettersNum**2 * self.diacriticsNum),
                                   dtype=np.int8)

        # Create <previous tag, tag> feature
        for prevTag in diacriticIDs.keys():
            for diacritic in diacriticIDs.keys():
                col = np.array([prevTag * self.diacriticsNum + diacritic])
                self.prevTag_tag[(prevTag, diacritic)] = \
                    csr_matrix((data, (row, col)), shape=(1, self.diacriticsNum**2), dtype=np.int8)

        # Create <previous tag 2, previous tag, tag> feature
        for prevTag2 in diacriticIDs.keys():
            for prevTag in diacriticIDs.keys():
                for diacritic in diacriticIDs.keys():
                    col = np.array([to1D(diacritic, prevTag, prevTag2, self.diacriticsNum, self.diacriticsNum)])
                    self.prevTag2_prevTag_tag[(prevTag2, prevTag, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(1, self.diacriticsNum**3),
                                   dtype=np.int8)

        # Create <previous tag, current letter, tag> feature
        for prevTag in diacriticIDs.keys():
            for currLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    col = np.array([to1D(diacritic, currLetter, prevTag, self.diacriticsNum, self.lettersNum)])
                    self.prevTag_currLetter_tag[(prevTag, currLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(1, self.diacriticsNum**2 * self.lettersNum),
                                   dtype=np.int8)

        # Create <current word, tag>
        for currWord in self.frequent_to_original_ID.keys():
            for diacritic in diacriticIDs.keys():
                col = np.array([currWord * self.diacriticsNum + diacritic])
                self.currWord_tag[(currWord, diacritic)] = \
                    csr_matrix((data, (row, col)), shape=(1, self.wordsNum*self.diacriticsNum), dtype=np.int8)

        # Create <previous word, current word, tag> feature
        for prevWord in self.frequent_to_original_ID.keys():
            for currWord in self.frequent_to_original_ID.keys():
                for diacritic in diacriticIDs.keys():
                    col = np.array([to1D(diacritic, currWord, prevWord, self.diacriticsNum, self.wordsNum)])
                    self.prevWord_currWord_tag[(prevWord, currWord, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(1, self.wordsNum**2 * self.diacriticsNum),
                                   dtype=np.int8)


def globalFeatures(self, history):
    indicators = []
    cols = []
    rows = []

    # Check if first letter in word
    if history.position_in_word == 0:
        indicators.append(1)
        cols.append(0)
        rows.append(0)

    # Check if last letter in word
    if history.nextLetter is None:
        indicators.append(1)
        cols.append(1)
        rows.append(0)

    # Check if first word
    if len(history.prev_word) == 0:
        indicators.append(1)
        cols.append(2)
        rows.append(0)

    # Check if last word
    if len(history.next_word) == 0:
        indicators.append(1)
        cols.append(3)
        rows.append(0)

    data = np.array(indicators)
    row = np.array(rows)
    col = np.array(cols)
    return csr_matrix((data, (row, col)), shape=(1, 4), dtype=np.int8)


def generateFeatures(history, tag):
    pass


# TESTING
test = FeaturesMatrix()
matrx1 = test.currLetter_tag[(1, 2)]
matrx2 = test.currLetter_tag[(15, 2)]
print(matrx1.toarray())
print(matrx2.toarray())
matrxTOT = hstack([matrx1, matrx2], format="csr")
print(matrxTOT.toarray())

