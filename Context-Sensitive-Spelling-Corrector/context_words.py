from collections import defaultdict


class contextwords:
    def __init__(self, training_set, vocabulary, confusing_set, uninformative_words, k, min_occurrences):
        self.training_set = training_set
        self.vocabulary = vocabulary
        self.confusing_set = confusing_set
        self.stop_words = uninformative_words
        self.k = k
        self.min_occurrences = min_occurrences
        self.context_word_prob = defaultdict(float)

        """
        For all the confusing word in training_set, storing the "context_word" with in distance of 3 from confusing word
        in a dictionary with tuple (context_word, confusing word) as index and its frequency as its value
        """
        for line in self.training_set:
            for i in xrange(len(line)):
                word = line[i]
                if self.confusing_word_check(word, self.confusing_set):
                    # j is index of word with in distance of 3 from "word = line[i]"
                    for j in xrange(i - self.k, i):
                        if j not in xrange(len(line)):
                            break
                        context_word = line[j]
                        self.context_word_prob[(context_word, word)] += 1
                    for j in xrange(i + 1, i + self.k + 1):
                        if j not in xrange(len(line)):
                            break
                        context_word = line[j]
                        self.context_word_prob[(context_word, word)] += 1

        unwanted_context_words = []
        for bigram, count in self.context_word_prob.iteritems():
            word_freq = self.vocabulary[bigram[1]]  # frequency of confusing word in self.vocabulary
            """
            There is no need to use every word in k distance to discriminate among the words in the confusion set
            If we do not have enough training data for a context_word to accurately estimate p(c|wi) for all wi, then we
            simply disregard context_word
            """
            if count < self.min_occurrences or (word_freq - count) < self.min_occurrences:
                unwanted_context_words.append(bigram)
            else:
                bigram_freq = 0
                for line in self.training_set:
                    if bigram[0] in line and bigram[1] in line:
                        bigram_freq += 1
                self.context_word_prob[bigram] = ((len(self.training_set) / float(bigram_freq)) * count) / float(word_freq)
            """
            Another reason to ignore a context word is if it does not help discriminate among words in the confusion set
            By ignoring uninformative words like I, me, the, is, of, we eliminate a source of noise in our
            discrimination procedure
            """
            if bigram[0] in self.stop_words:
                if bigram not in unwanted_context_words:
                    unwanted_context_words.append(bigram)
        """
        Prune context words that have insufficient data and are uninformative discriminators
        """
        for bigram in unwanted_context_words:
            del self.context_word_prob[bigram]

    """
    This function matches the passed word to all confusion sets
    and returns the confusing_set if match is found otherwise returns None
    """
    def confusing_word_check(self, word, confusing_sets_pruned):
        for confusing_set in confusing_sets_pruned:
            if word in confusing_set:
                return confusing_set
        return None

    # TESTING
    def test(self, testset):
        testset_corrected = []
        for line in testset:
            corrected_line = []
            for i in xrange(len(line)):
                predicted_word = self.word_prediction(i, line)
                corrected_line.append(predicted_word)
            testset_corrected.append(corrected_line)
        return testset_corrected

    def word_prediction(self, i, line):
        word = line[i]
        confusing_set = self.confusing_word_check(word, self.confusing_set)
        if confusing_set == None:
            return word
        else:
            confusion_proba_dict = defaultdict(float)
            word_freq = 0
            for val in self.vocabulary.values():
                word_freq += val  # Total no. of words in vocabulary
            for confusing_word in confusing_set:
                confusion_proba_dict[confusing_word] = float(self.vocabulary[confusing_word]) / word_freq

                for j in xrange(i - self.k, i):
                    if j not in xrange(len(line)):
                        break
                    context_word = line[j]
                    bigram = (context_word, confusing_word)
                    if bigram in self.context_word_prob:
                        confusion_proba_dict[confusing_word] *= self.context_word_prob[bigram]
                for j in xrange(i + 1, i + self.k + 1):
                    if j not in xrange(len(line)):
                        break
                    context_word = line[j]
                    bigram = (context_word, confusing_word)
                    if bigram in self.context_word_prob:
                        confusion_proba_dict[confusing_word] *= self.context_word_prob[bigram]
            # Storing the most probable sentence
            best_prob = None
            best_word = None
            for confusing_word, prob in confusion_proba_dict.iteritems():
                if prob > best_prob:
                    best_prob = prob
                    best_word = confusing_word
            return best_word
