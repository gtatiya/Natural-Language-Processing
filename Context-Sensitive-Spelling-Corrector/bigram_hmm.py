from collections import defaultdict

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"

class BigramHMM:
    def __init__(self, training_set, confusing_sets):
        self.training_set = training_set
        self.confusing_sets = confusing_sets
        self.unigram_counts = defaultdict(int)
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)
        self.dictionary = defaultdict(lambda:list())

        # self.unigram_counts is a dictionary with POS as index and POS frequency as its value
        for line in training_set:
            for word in line:
                self.unigram_counts[word[1]] += 1

        # self.transitions contains P(ti|ti-1) i.e. C(ti-1,ti)/C(ti-1)
        for line in training_set:
            for i in xrange(len(line) - 1):
                bigram = (line[i][1], line[i + 1][1])
                self.transitions[bigram] += 1
        for bigram, count in self.transitions.iteritems():
            self.transitions[bigram] = float(count) / float(self.unigram_counts[bigram[0]])

        # self.emissions that contains P(wi|ti) i.e. C(wi,ti)/C(ti)
        for line in training_set:
            for bigram in line:
                self.emissions[bigram] += 1
        for bigram, count in self.emissions.iteritems():
            self.emissions[bigram] = float(count) / float(self.unigram_counts[bigram[1]])

        # self.dictionary is a dictionary that maps a word to the set of possible tags
        for line in training_set:
            for word in line:
                if word[1] not in self.dictionary[word[0]]:
                    self.dictionary[word[0]].append(word[1])

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
                word = line[i]
                confusing_set = self.confusing_word_check(word, self.confusing_sets)
                confusing_set_prob = {}
                if confusing_set == None:
                    corrected_line.append(word)
                else:
                    for confusing_word in confusing_set:
                        probable_sent = line
                        probable_sent[i] = confusing_word
                        confusing_set_prob[confusing_word] = self.viterbi(probable_sent)
                    # Storing the most probable sentence
                    best_prob = None
                    best_word = None
                    for confusing_word, prob in confusing_set_prob.iteritems():
                        if prob[0] > best_prob:
                            best_word = confusing_word
                            best_prob = prob[0]
                    corrected_line.append(best_word)
            testset_corrected.append(corrected_line)
        return testset_corrected

    def viterbi(self, line):
        T = len(line)
        viterbi = []
        backpointer = []
        for i in xrange(T):
            viterbi.append({})
            backpointer.append({})

        viterbi[0][start_token] = 1.0
        backpointer[0][start_token] = None

        for i in xrange(1, T):
            word = line[i]
            for pos in self.dictionary[word]:
                max_last_pos = None
                max_prob = None
                for last_pos, last_prob in viterbi[i - 1].iteritems():
                    path_prob = self.transitions[(last_pos, pos)] * self.emissions[(word, pos)] * last_prob
                    if (max_prob or path_prob != -float('inf')) and path_prob > max_prob:
                        max_prob = path_prob
                        max_last_pos = last_pos
                # Storing the POS with maximum probability
                viterbi[i][pos] = max_prob
                backpointer[i][pos] = max_last_pos

        most_prob = viterbi[T - 1][end_token]  # probability of '</S>'
        best_sent = self.back_tracing(line, backpointer)
        return (most_prob, best_sent)

    def back_tracing(self, line, backpointer, pos=end_token, idx=None):
        if idx == 0:
            return [pos]
        if idx == None:
            idx = len(line) - 1
        last_pos = backpointer[idx][pos]
        last_best_sent = self.back_tracing(line, backpointer, last_pos, idx - 1)
        return last_best_sent + [pos]
