import pickle

# Global constants
skipStars = 0   # NOT USED

with open('Dataset_with_shaddah/letter_to_id_memm.pickle', 'rb') as file:
    letterMap = pickle.load(file)
with open('Dataset_with_shaddah/word_to_id_memm.pickle', 'rb') as file2:
    wordMap = pickle.load(file2)
with open('Dataset_with_shaddah/id_to_letter_memm.pickle', 'rb') as file3:
    id_to_letter = pickle.load(file3)


class History:
    def __init__(self, prevLabel1, prevLabel2, sentence, position):
        self.prevLabel1 = prevLabel1
        self.prevLabel2 = prevLabel2
        self.sentence = sentence
        self.position = position + skipStars
        self.current_word, self.prev_word, self.next_word, self.position_in_word = self.__parse_words()
        self.curr_letter = sentence[self.position]
        if position == 0:
            self.prevLetter = letterMap['*']
        else:
            self.prevLetter = sentence[position-1]

        if position == len(sentence) - 1:
            self.nextLetter = letterMap['*']
        else:
            self.nextLetter = sentence[position+1]

    def __parse_words(self):
        prev_space_position = None
        word_begin_position = skipStars

        i = skipStars
        if self.sentence[skipStars] == letterMap[' ']:
            i = skipStars + 1

        while i < self.position:
            if self.sentence[i] == letterMap[' ']:
                prev_space_position = word_begin_position
                word_begin_position = i
            i += 1

        if word_begin_position == skipStars and self.sentence[skipStars] != letterMap[' ']:
            word_begin_position = skipStars - 1
            position_in_word = self.position - skipStars
        else:
            position_in_word = self.position - word_begin_position - 1

        word_end_position = len(self.sentence) - 1
        while i < len(self.sentence):
            if self.sentence[i] == letterMap[' ']:
                word_end_position = i
                break
            i += 1

        next_word_end_position = i
        if i == len(self.sentence):
            word_end_position = len(self.sentence)
            next_word = wordMap["*"]
        else:
            i += 1
            while i < len(self.sentence):
                if self.sentence[i] == letterMap[' ']:
                    next_word_end_position = i
                    break
                i += 1
            if i == len(self.sentence):
                next_word_end_position = len(self.sentence)
            next_word = self.sentence[word_end_position+1:next_word_end_position]

        current_word = self.sentence[word_begin_position+1:word_end_position]

        if word_begin_position == skipStars-1:
            prev_word = wordMap["*"]
        else:
            prev_word = self.sentence[prev_space_position+1:word_begin_position]

        if current_word != wordMap["*"]:
            translated_word_list = [id_to_letter[id] for id in current_word]
            translated_word = ''.join(translated_word_list)
            current_word_id = wordMap[translated_word]
        else:
            current_word_id = current_word

        if prev_word != wordMap["*"]:
            translated_word_list = [id_to_letter[id] for id in prev_word]
            translated_word = ''.join(translated_word_list)
            prev_word_id = wordMap[translated_word]
        else:
            prev_word_id = prev_word

        if next_word != wordMap["*"]:
            translated_word_list = [id_to_letter[id] for id in next_word]
            translated_word = ''.join(translated_word_list)
            next_word_id = wordMap[translated_word]
        else:
            next_word_id = next_word

        return current_word_id, prev_word_id, next_word_id, position_in_word

    def get_prev_letter(self):
        return self.prevLetter

    def get_next_letter(self):
        return self.nextLetter

    def get_prev_word(self):
        return self.prev_word

    def get_curr_word(self):
        return self.current_word

    def get_next_word(self):
        return self.next_word

    def get_sentence(self):
        return self.sentence

    def get_position(self):
        return self.position

    def get_position_in_word(self):
        return self.position_in_word

    def get_current_letter(self):
        return self.curr_letter

    def get_prev_label1(self):
        return self.prevLabel1

    def get_prev_label2(self):
        return self.prevLabel2


# TESTING
# sentence = [11, 12, 3, 2, 1, 10, 21, 2, 23, 18]
# histTest = History('a', 'u', sentence, 9)
# print("Next Letter: ", histTest.get_next_letter())
# print("Prev Letter: ", histTest.get_prev_letter())
# print("Curr Word: ", histTest.get_curr_word())
# print("Next Word: ", histTest.get_next_word())
# print("Prev Word: ", histTest.get_prev_word())
# print("Position in Word: ", histTest.get_position_in_word())
