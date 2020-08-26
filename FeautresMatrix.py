import numpy as np
import pickle
from scipy.sparse import csr_matrix

with open('Dataset_with_shaddah/id_to_letter.pickle', 'rb') as file:
    letterIDs = pickle.load(file)
with open('Dataset_with_shaddah/id_to_diacritic.pickle', 'rb') as file2:
    diacriticIDs = pickle.load(file2)


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

        data = np.array([1])

        # Create <current letter, tag> feature vector
        for letter in letterIDs.keys():
            for diacritic in diacriticIDs.keys():
                row = np.array([letter])
                col = np.array([diacritic])
                self.currLetter_tag[(letter, diacritic)] = \
                    csr_matrix((data, (row, col)), shape=(self.lettersNum, self.diacriticsNum), dtype=np.int8)

        # Create <previous letter, current letter, tag> feature
        for prevLetter in letterIDs.keys():
            for currLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    row = np.array([self.lettersNum * currLetter + prevLetter])
                    col = np.array([diacritic])
                    self.prevLetter_currLetter_tag[(prevLetter, currLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(self.lettersNum**2, self.diacriticsNum),
                                   dtype=np.int8)

        # Create <current letter, next letter, tag> feature
        for currLetter in letterIDs.keys():
            for nextLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    row = np.array([self.lettersNum * nextLetter + currLetter])
                    col = np.array([diacritic])
                    self.currLetter_nextLetter_tag[(currLetter, nextLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(self.lettersNum**2, self.diacriticsNum),
                                   dtype=np.int8)

        # Create <previous tag, tag> feature
        for prevTag in diacriticIDs.keys():
            for diacritic in diacriticIDs.keys():
                row = np.array([prevTag])
                col = np.array([diacritic])
                self.prevTag_tag[(prevTag, diacritic)] = \
                    csr_matrix((data, (row, col)), shape=(self.diacriticsNum, self.diacriticsNum), dtype=np.int8)

        # Create <previous tag 2, previous tag, tag> feature
        for prevTag2 in diacriticIDs.keys():
            for prevTag in diacriticIDs.keys():
                for diacritic in diacriticIDs.keys():
                    row = np.array([self.diacriticsNum * prevTag + prevTag2])
                    col = np.array([diacritic])
                    self.prevTag2_prevTag_tag[(prevTag2, prevTag, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(self.diacriticsNum**2, self.diacriticsNum),
                                   dtype=np.int8)

        # Create <previous tag, current letter, tag> feature
        for prevTag in diacriticIDs.keys():
            for currLetter in letterIDs.keys():
                for diacritic in diacriticIDs.keys():
                    row = np.array([self.diacriticsNum * currLetter + prevTag])
                    col = np.array([diacritic])
                    self.prevTag2_prevTag_tag[(prevTag, currLetter, diacritic)] = \
                        csr_matrix((data, (row, col)),
                                   shape=(self.diacriticsNum*self.lettersNum, self.diacriticsNum),
                                   dtype=np.int8)

        # TODO: Create <current word, tag>

        # TODO: Create <previous word, current word, tag>


# TESTING
# test = FeaturesMatrix()
# matrx = test.prevLetter_currLetter_tag[(1, 1, 2)]
# print(matrx.toarray())
