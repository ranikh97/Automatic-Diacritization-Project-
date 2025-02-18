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
data = np.array([1])
row = np.array([0])
min_frequency = 5

frequentWords = [id for id in wordIDs.keys() if wordCounts[wordIDs[id]] >= min_frequency]
frequent_to_original_ID = {}
original_to_frequent_ID = {}
i = 0
for word_id in frequentWords:
    frequent_to_original_ID[i] = word_id
    original_to_frequent_ID[word_id] = i
    i += 1


def to1D(x, y, z, xMax, yMax):
    return (z * xMax * yMax) + (y * xMax) + x


# class FeaturesMatrix:
#     def __init__(self):
#         self.currLetter_tag = {}
#         self.prevLetter_currLetter_tag = {}
#         self.currLetter_nextLetter_tag = {}
#         self.prevTag_tag = {}
#         self.prevTag2_prevTag_tag = {}
#         self.prevTag_currLetter_tag = {}
#         self.currWord_tag = {}
#         self.prevWord_currWord_tag = {}
#
#         self.lettersNum = len(letterIDs)
#         self.diacriticsNum = len(diacriticIDs)
#
#         self.frequentWords = [id for id in wordIDs.keys() if wordCounts[wordIDs[id]] > 5]
#         self.wordsNum = len(self.frequentWords)
#         self.frequent_to_original_ID = {}
#         self.original_to_frequent_ID = {}
#         i = 0
#         for id in self.frequentWords:
#             self.frequent_to_original_ID[i] = id
#             self.original_to_frequent_ID[id] = i
#             i += 1
#
#         data = np.array([1])
#         row = np.array([0])
#
#         # Create <current letter, tag> feature vector
#         for letter in letterIDs.keys():
#             for diacritic in diacriticIDs.keys():
#                 col = np.array([self.diacriticsNum * letter + diacritic])
#                 self.currLetter_tag[(letter, diacritic)] = \
#                     csr_matrix((data, (row, col)), shape=(1, self.lettersNum*self.diacriticsNum), dtype=np.int8)
#
#         # Create <previous letter, current letter, tag> feature
#         for prevLetter in letterIDs.keys():
#             for currLetter in letterIDs.keys():
#                 for diacritic in diacriticIDs.keys():
#                     col = np.array([to1D(diacritic, currLetter, prevLetter, self.diacriticsNum, self.lettersNum)])
#                     self.prevLetter_currLetter_tag[(prevLetter, currLetter, diacritic)] = \
#                         csr_matrix((data, (row, col)),
#                                    shape=(1, self.lettersNum**2 * self.diacriticsNum),
#                                    dtype=np.int8)
#
#         # Create <current letter, next letter, tag> feature
#         for currLetter in letterIDs.keys():
#             for nextLetter in letterIDs.keys():
#                 for diacritic in diacriticIDs.keys():
#                     col = np.array([to1D(diacritic, nextLetter, currLetter, self.diacriticsNum, self.lettersNum)])
#                     self.currLetter_nextLetter_tag[(currLetter, nextLetter, diacritic)] = \
#                         csr_matrix((data, (row, col)),
#                                    shape=(1, self.lettersNum**2 * self.diacriticsNum),
#                                    dtype=np.int8)
#
#         # Create <previous tag, tag> feature
#         for prevTag in diacriticIDs.keys():
#             for diacritic in diacriticIDs.keys():
#                 col = np.array([prevTag * self.diacriticsNum + diacritic])
#                 self.prevTag_tag[(prevTag, diacritic)] = \
#                     csr_matrix((data, (row, col)), shape=(1, self.diacriticsNum**2), dtype=np.int8)
#
#         # Create <previous tag 2, previous tag, tag> feature
#         for prevTag2 in diacriticIDs.keys():
#             for prevTag in diacriticIDs.keys():
#                 for diacritic in diacriticIDs.keys():
#                     col = np.array([to1D(diacritic, prevTag, prevTag2, self.diacriticsNum, self.diacriticsNum)])
#                     self.prevTag2_prevTag_tag[(prevTag2, prevTag, diacritic)] = \
#                         csr_matrix((data, (row, col)),
#                                    shape=(1, self.diacriticsNum**3),
#                                    dtype=np.int8)
#
#         # Create <previous tag, current letter, tag> feature
#         for prevTag in diacriticIDs.keys():
#             for currLetter in letterIDs.keys():
#                 for diacritic in diacriticIDs.keys():
#                     col = np.array([to1D(diacritic, currLetter, prevTag, self.diacriticsNum, self.lettersNum)])
#                     self.prevTag_currLetter_tag[(prevTag, currLetter, diacritic)] = \
#                         csr_matrix((data, (row, col)),
#                                    shape=(1, self.diacriticsNum**2 * self.lettersNum),
#                                    dtype=np.int8)
#
#         # Create <current word, tag>
#         for currWord in self.frequent_to_original_ID.keys():
#             for diacritic in diacriticIDs.keys():
#                 col = np.array([currWord * self.diacriticsNum + diacritic])
#                 self.currWord_tag[(currWord, diacritic)] = \
#                     csr_matrix((data, (row, col)), shape=(1, self.wordsNum*self.diacriticsNum), dtype=np.int8)
#
#         # Create <previous word, current word, tag> feature
#         for prevWord in self.frequent_to_original_ID.keys():
#             for currWord in self.frequent_to_original_ID.keys():
#                 for diacritic in diacriticIDs.keys():
#                     col = np.array([to1D(diacritic, currWord, prevWord, self.diacriticsNum, self.wordsNum)])
#                     self.prevWord_currWord_tag[(prevWord, currWord, diacritic)] = \
#                         csr_matrix((data, (row, col)),
#                                    shape=(1, self.wordsNum**2 * self.diacriticsNum),
#                                    dtype=np.int8)

def get_currLetter_tag(letter, tag, consider_tag):
    # col = np.array([len(diacriticIDs) * letter + tag])
    # return csr_matrix((data, (row, col)), shape=(1, len(letterIDs) * len(diacriticIDs)), dtype=np.int8)
    if consider_tag:
        arr = np.zeros(len(letterIDs) * len(diacriticIDs))
        col = [len(diacriticIDs) * letter + tag]
        arr[col] = 1
        return arr
    else:
        arr = np.zeros(len(letterIDs))
        col = [letter]
        arr[col] = 1
        return arr


def get_prevLetter_currLetter_tag(prevLetter, currLetter, tag, consider_tag):
    if consider_tag:
        arr = np.zeros(len(letterIDs)**2 * len(diacriticIDs))
        col = to1D(tag, currLetter, prevLetter, len(diacriticIDs), len(letterIDs))
        arr[col] = 1
        return arr
        # col = np.array([to1D(tag, currLetter, prevLetter, len(diacriticIDs), len(letterIDs))])
        # return csr_matrix((data, (row, col)), shape=(1, len(letterIDs)**2 * len(diacriticIDs)), dtype=np.int8)
    else:
        arr = np.zeros(len(letterIDs)**2)
        col = [len(letterIDs) * prevLetter + currLetter]
        arr[col] = 1
        return arr


def get_currLetter_nextLetter_tag(currLetter, nextLetter, tag, consider_tag):
    if consider_tag:
        # col = np.array([to1D(tag, nextLetter, currLetter, len(diacriticIDs), len(letterIDs))])
        # return csr_matrix((data, (row, col)), shape=(1, len(letterIDs)**2 * len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(letterIDs)**2 * len(diacriticIDs))
        col = to1D(tag, nextLetter, currLetter, len(diacriticIDs), len(letterIDs))
        arr[col] = 1
        return arr
    else:
        arr = np.zeros(len(letterIDs)**2)
        col = [len(letterIDs) * currLetter + nextLetter]
        arr[col] = 1
        return arr


def get_prevTag_tag(prevTag, tag, consider_tag):
    if consider_tag:
        # col = np.array([prevTag * len(diacriticIDs) + tag])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)**2), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs)**2)
        col = prevTag * len(diacriticIDs) + tag
        arr[col] = 1
        return arr
    else:
        arr = np.zeros(len(diacriticIDs))
        col = [prevTag]
        arr[col] = 1
        return arr


def get_prevTag2_prevTag_tag(prevTag2, prevTag, tag, consider_tag):
    if consider_tag:
        # col = np.array([to1D(tag, prevTag, prevTag2, len(diacriticIDs), len(diacriticIDs))])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)**3), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs)**3)
        col = to1D(tag, prevTag, prevTag2, len(diacriticIDs), len(diacriticIDs))
        arr[col] = 1
        return arr
    else:
        arr = np.zeros(len(diacriticIDs) ** 2)
        col = [len(diacriticIDs) * prevTag2 + prevTag]
        arr[col] = 1
        return arr


def get_prevTag_currLetter_tag(prevTag, currLetter, tag, consider_tag):
    if consider_tag:
        # col = np.array([to1D(tag, currLetter, prevTag, len(diacriticIDs), len(letterIDs))])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)**2 * len(letterIDs)), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs)**2 * len(letterIDs))
        col = to1D(tag, currLetter, prevTag, len(diacriticIDs), len(letterIDs))
        arr[col] = 1
        return arr
    else:
        arr = np.zeros(len(diacriticIDs) * len(letterIDs))
        col = [len(letterIDs) * prevTag + currLetter]
        arr[col] = 1
        return arr


def get_currWord_tag(currWord, tag, consider_tag):
    if wordCounts[wordIDs[currWord]] < min_frequency:
        if consider_tag:
            # return csr_matrix((1, len(frequentWords) * len(diacriticIDs)), dtype=np.int8)
            return np.zeros(len(frequentWords) * len(diacriticIDs))
        else:
            return np.zeros(len(frequentWords))
    else:
        if consider_tag:
            # col = np.array([currWord * len(diacriticIDs) + tag])
            # return csr_matrix((data, (row, col)), shape=(1, len(frequentWords) * len(diacriticIDs)), dtype=np.int8)
            arr = np.zeros(len(frequentWords) * len(diacriticIDs))
            col = original_to_frequent_ID[currWord] * len(diacriticIDs) + tag
            arr[col] = 1
            return arr
        else:
            arr = np.zeros(len(frequentWords))
            col = [original_to_frequent_ID[currWord]]
            arr[col] = 1
            return arr


def get_prevWord_currWord_tag(prevWord, currWord, tag):
    if wordCounts[wordIDs[currWord]] < min_frequency or wordCounts[wordIDs[prevWord]] < min_frequency:
        # return csr_matrix((1, len(frequentWords)**2 * len(diacriticIDs)), dtype=np.int8)
        return np.zeros(len(frequentWords)**2 * len(diacriticIDs))
    else:
        # col = np.array([to1D(tag, currWord, prevWord, len(diacriticIDs), len(frequentWords))])
        # return csr_matrix((data, (row, col)), shape=(1, len(frequentWords)**2 * len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(frequentWords)**2 * len(diacriticIDs))
        col = to1D(tag, original_to_frequent_ID[currWord], prevWord, len(diacriticIDs), len(frequentWords))
        arr[col] = 1
        return arr


def get_if_first_letter(history, tag):
    if history.get_position_in_word() == 0:
        # col = np.array([tag])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs))
        arr[tag] = 1
        return arr
    else:
        # return csr_matrix((1, len(diacriticIDs)), dtype=np.int8)
        return np.zeros(len(diacriticIDs))


def get_if_last_letter(history, tag):
    if history.get_next_letter() in [letterToID[" "], letterToID["*"]]:
        # col = np.array([tag])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs))
        arr[tag] = 1
        return arr
    else:
        # return csr_matrix((1, len(diacriticIDs)), dtype=np.int8)
        return np.zeros(len(diacriticIDs))


def get_if_first_word(history, tag):
    if history.get_prev_word() == wordToID["*"]:
        # col = np.array([tag])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs))
        arr[tag] = 1
        return arr
    else:
        # return csr_matrix((1, len(diacriticIDs)), dtype=np.int8)
        return np.zeros(len(diacriticIDs))


def get_if_last_word(history, tag):
    if history.get_next_word() == wordToID["*"]:
        # col = np.array([tag])
        # return csr_matrix((data, (row, col)), shape=(1, len(diacriticIDs)), dtype=np.int8)
        arr = np.zeros(len(diacriticIDs))
        arr[tag] = 1
        return arr
    else:
        # return csr_matrix((1, len(diacriticIDs)), dtype=np.int8)
        return np.zeros(len(diacriticIDs))


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


def generateFeatures(history: History, tag, consider_tag: bool):
    features_list = []
    features_list.append(get_currLetter_tag(history.get_current_letter(),
                                            tag, consider_tag))
    features_list.append(get_prevLetter_currLetter_tag(history.get_prev_letter(),
                                                       history.get_current_letter(),
                                                       tag, consider_tag))
    features_list.append(get_currLetter_nextLetter_tag(history.get_current_letter(),
                                                       history.get_next_letter(),
                                                       tag, consider_tag))
    features_list.append(get_prevTag_tag(history.get_prev_label1(),
                                         tag, consider_tag))
    features_list.append(get_prevTag2_prevTag_tag(history.get_prev_label2(),
                                                  history.get_prev_label1(),
                                                  tag, consider_tag))
    features_list.append(get_prevTag_currLetter_tag(history.get_prev_label1(),
                                                    history.get_current_letter(),
                                                    tag, consider_tag))
    features_list.append(get_currWord_tag(history.get_curr_word(),
                                          tag, consider_tag))
    # features_list.append(get_prevWord_currWord_tag(history.get_prev_word(),
    #                                                history.get_curr_word(),
    #                                                tag))
    # features_list.append(calcGlobalFeatures(history))
    features_list.append(get_if_first_letter(history, tag))
    features_list.append(get_if_last_letter(history, tag))
    features_list.append(get_if_first_word(history, tag))
    features_list.append(get_if_last_word(history, tag))

    # return hstack(features_list, format="csr")
    return np.concatenate(tuple(features_list))


# TESTING
# sentence = [0, 1, 4, 2, 16, 5, 8, 20, 2, 20, 16, 2, 32, 16, 31, 2, 4, 10, 2, 11, 1, 36, 6, 11, 8, 2, 1, 20, 30, 2, 0, 11, 29, 17, 2, 4, 4, 10, 2, 14, 11, 32, 8, 18, 11, 2, 4, 10, 2, 11, 1, 34, 29, 11, 14, 19, 2, 8, 25, 16, 2, 11, 1, 1, 18, 2, 5, 10, 18, 4, 2, 14, 1, 20, 30, 2, 7, 4, 7, 5, 7, 2, 14, 9, 2, 0, 10, 29, 0, 18, 2, 0, 11, 1, 1, 18, 2, 12, 5, 1, 4, 2, 22, 0, 1, 18, 2, 0, 22, 11, 1, 2, 4, 11, 1, 9, 2, 18, 0, 2, 3, 11, 35, 23, 2, 10, 27, 14, 7, 18, 2, 21, 1, 13, 2, 4, 11, 1, 9, 2, 33, 1, 15]
# histTest = History(16, 16, sentence, 0)
#
# res = generateFeatures(histTest, 17, True)
# print(res)
