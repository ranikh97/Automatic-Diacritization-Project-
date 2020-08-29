import numpy as np
import pickle
from scipy.sparse import csr_matrix, hstack

from History import History

with open('Dataset_with_shaddah/id_to_letter_memm.pickle', 'rb') as file:
    letterIDs = pickle.load(file)
with open('Dataset_with_shaddah/id_to_diacritic_memm.pickle', 'rb') as file2:
    diacriticIDs = pickle.load(file2)
with open('Dataset_with_shaddah/id_to_word_memm.pickle', 'rb') as file3:
    wordIDs = pickle.load(file3)
with open('Dataset_with_shaddah/word_count.pickle', 'rb') as file4:
    wordCounts = pickle.load(file4)
with open('Dataset_with_shaddah/word_to_id_memm.pickle', 'rb') as file5:
    wordToID = pickle.load(file5)
with open('Dataset_with_shaddah/letter_to_id_memm.pickle', 'rb') as file6:
    letterToID = pickle.load(file6)


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

        self.frequentWords = [id for id in wordIDs.keys() if wordCounts[wordIDs[id]] > 5]
        self.wordsNum = len(self.frequentWords)
        self.frequent_to_original_ID = {}
        self.original_to_frequent_ID = {}
        i = 0
        for id in self.frequentWords:
            self.frequent_to_original_ID[i] = id
            self.original_to_frequent_ID[id] = i
            i += 1

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

    def get_currLetter_tag(self, letter, tag):
        return self.currLetter_tag[(letter, tag)]

    def get_prevLetter_currLetter_tag(self, prevLetter, currLetter, tag):
        return self.prevLetter_currLetter_tag[(prevLetter, currLetter, tag)]

    def get_currLetter_nextLetter_tag(self, currLetter, nextLetter, tag):
        return self.currLetter_nextLetter_tag[(currLetter, nextLetter, tag)]

    def get_prevTag_tag(self, prevTag, tag):
        return self.prevTag_tag[(prevTag, tag)]

    def get_prevTag2_prevTag_tag(self, prevTag2, prevTag, tag):
        return self.prevTag2_prevTag_tag[(prevTag2, prevTag, tag)]

    def get_prevTag_currLetter_tag(self, prevTag, currLetter, tag):
        return self.prevTag_currLetter_tag[(prevTag, currLetter, tag)]

    def get_currWord_tag(self, currWord, tag):
        if currWord not in self.frequentWords:
            return csr_matrix((1, self.wordsNum*self.diacriticsNum), dtype=np.int8)
        return self.currWord_tag[(currWord, tag)]

    def get_prevWord_currWord_tag(self, prevWord, currWord, tag):
        if currWord not in self.frequentWords:
            return csr_matrix((1, self.wordsNum**2 * self.diacriticsNum), dtype=np.int8)
        return self.prevWord_currWord_tag[(prevWord, currWord, tag)]


def calcGlobalFeatures(history):
    indicators = []
    cols = []
    rows = []

    # Check if first letter in word
    if history.get_position_in_word() == 0:
        indicators.append(1)
        cols.append(0)
        rows.append(0)

    # Check if last letter in word
    if history.get_next_letter() in [letterToID[" "], letterToID["*"]]:
        indicators.append(1)
        cols.append(1)
        rows.append(0)

    # Check if first word
    if history.get_prev_word() == wordToID["*"]:
        indicators.append(1)
        cols.append(2)
        rows.append(0)

    # Check if last word
    if history.get_next_word() == wordToID["*"]:
        indicators.append(1)
        cols.append(3)
        rows.append(0)

    data = np.array(indicators)
    row = np.array(rows)
    col = np.array(cols)
    return csr_matrix((data, (row, col)), shape=(1, 4), dtype=np.int8)


def generateFeatures(features: FeaturesMatrix, history: History, tag):
    features_list = []
    features_list.append(features.get_currLetter_tag(history.get_current_letter(),
                                                     tag))
    features_list.append(features.get_prevLetter_currLetter_tag(history.get_prev_letter(),
                                                                history.get_current_letter(),
                                                                tag))
    features_list.append(features.get_currLetter_nextLetter_tag(history.get_current_letter(),
                                                                history.get_next_letter(),
                                                                tag))
    features_list.append(features.get_prevTag_tag(history.get_prev_label1(),
                                                  tag))
    features_list.append(features.get_prevTag2_prevTag_tag(history.get_prev_label2(),
                                                           history.get_prev_label1(),
                                                           tag))
    features_list.append(features.get_prevTag_currLetter_tag(history.get_prev_label1(),
                                                             history.get_current_letter(),
                                                             tag))
    features_list.append(features.get_currWord_tag(history.get_curr_word(),
                                                   tag))
    features_list.append(features.get_prevWord_currWord_tag(history.get_prev_word(),
                                                            history.get_curr_word(),
                                                            tag))
    features_list.append(calcGlobalFeatures(history))

    return hstack(features_list, format="csr")


# TESTING
sentence = [letterToID['و'], letterToID['ل'], letterToID['و'], letterToID[' '],
            letterToID['ت'], letterToID['ر'], letterToID['ك'], letterToID[' '],
            letterToID['خ'], letterToID['ش'], letterToID['ع']]
histTest = History(2, 5, sentence, 4)

test = FeaturesMatrix()

print(generateFeatures(test, histTest, 3))
