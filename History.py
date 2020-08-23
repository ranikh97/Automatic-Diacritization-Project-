class History:
    def __init__(self, prevLetter1, prevLetter2, nextLetter1, nextLetter2, prevWord, sentence, position):
        self.prevLetter1 = prevLetter1
        self.prevLetter2 = prevLetter2
        self.nextLetter1 = nextLetter1
        self.nextLetter2 = nextLetter2
        self.prevWord = prevWord
        self.sentence = sentence
        self.position = position

    def get_prev_letter1(self):
        return self.prevLetter1

    def get_prev_letter2(self):
        return self.prevLetter2

    def get_next_letter1(self):
        return self.prevLetter1

    def get_next_letter2(self):
        return self.prevLetter2

    def get_prev_word(self):
        return self.prevWord

    def get_sentence(self):
        return self.sentence

    def get_position(self):
        return self.position