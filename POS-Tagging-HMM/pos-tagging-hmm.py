import sys
from collections import defaultdict
from math import log, exp

from nltk.corpus import treebank
from nltk.tag.util import untag  # Untags a tagged sentence. 

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Remove trace tokens and tags from the treebank as these are not necessary.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

# Introducing unknown tokens <UNK> and adding sentence boundary symbols <S></S> at the beginning and
# end of every sentence and return sentence by sentence list (dataset_prep) & 1D list of all the tokens (dataset_prep_1d)
def PreprocessText(dataset, vocabulary):
    dataset_prep = []
    dataset_prep_1d = []
    for sentence in dataset:
        dataset_prep_1d.append((start_token, start_token)) #start and end symbols are always tagged as themselves
        for i in xrange(len(sentence)):
            if sentence[i][0] not in vocabulary:
                word_POS = sentence[i]
                sentence[i] = unknown_token, word_POS[1]
            dataset_prep_1d.append(sentence[i])
        dataset_prep_1d.append((end_token, end_token))
        sentence.append((end_token, end_token))
        sentence.insert(0,(start_token, start_token))
        dataset_prep.append(sentence)
    return dataset_prep, dataset_prep_1d

class BigramHMM:
    def __init__(self):
        """ Implement:
        self.transitions, the A matrix of the HMM: a_{ij} = P(t_j | t_i)
        self.emissions, the B matrix of the HMM: b_{ii} = P(w_i | t_i)
        self.dictionary, a dictionary that maps a word to the set of possible tags
        """
        self.dictionary = defaultdict(set)  # maps every word to a set of its candidate tags in the training set
        self.dataset_size = int()
        self.transitions = defaultdict(float)
        self.emissions = defaultdict(float)

    def Train(self, training_set):
        """ 
        1. Estimate the A matrix a_{ij} = P(t_j | t_i)
        2. Estimate the B matrix b_{ii} = P(w_i | t_i)
        3. Compute the tag dictionary 
        """
        c_word_tag = defaultdict(int)
        c_tag = defaultdict(int)
        c_tag_bigram = defaultdict(int)
        self.dataset_size = len(training_set)

        for i in xrange(len(training_set) - 1):
            c_word_tag[training_set[i]] += 1
            c_tag[training_set[i][1]] += 1
            c_tag_bigram[(training_set[i][1], training_set[i + 1][1])] += 1
            self.dictionary[training_set[i][0]].add(training_set[i][1])
        c_word_tag[training_set[-1]] += 1
        c_tag[training_set[-1][1]] += 1
        self.dictionary[end_token].add(end_token)

        # "self.transitions" contains P(ti|ti-1) i.e. C(ti-1,ti)/C(ti-1)
        for tag_tag in c_tag_bigram:
            self.transitions[tag_tag] = log(float(c_tag_bigram[tag_tag]) / c_tag[tag_tag[0]])

        # "self.emissions" contains P(Wi|ti) i.e. C(Wi,ti)/C(ti)
        for word_tag in c_word_tag:
            self.emissions[word_tag] = log(float(c_word_tag[word_tag]) / c_tag[word_tag[1]])

    def ComputePercentAmbiguous(self, data_set):
        """ Compute the percentage of tokens in data_set that have more than one tag according to self.dictionary. """
        """
        FEEDBACK 3.2
        You should not be including the start/end tags in the calculation as they
        artificially decrease the ambiguity, which is misleading
        """
        c_ambiguity = 0

        for i in xrange(len(data_set)):
            if len(self.dictionary[data_set[i][0]]) > 1:
                c_ambiguity += 1
        ambiguity = float(c_ambiguity) * 100 / self.dataset_size

        print "Number of tags for the unknown token in the dictionary:", len(self.dictionary['<UNK>'])
        print "List of tags for the unknown token in the dictionary:", self.dictionary['<UNK>']

        return ambiguity
        
    def JointProbability(self, sent):
        """ Compute the joint probability of the words and tags of a tagged sentence. """
        jp = 0.0
        for i in xrange(len(sent) - 1):
            jp += self.emissions[sent[i]] + self.transitions[(sent[i][1], sent[i + 1][1])]
        return exp(jp)
        
    def Viterbi(self, sent):
        """ Find the probability and identity of the most likely tag sequence given the sentence. """
        viterbi = defaultdict(dict)
        backpointer = defaultdict(dict)
        sent_tag = []
        pos_list = [end_token]
        viterbi['0'] = 1.0

        # Initialization step
        # This loop will run for all the tags of each first word (sent[1][0])(word next to <S>) in dictionary
        for tag in self.dictionary[sent[1][0]]:
            # if any sentance in our trained data starts with a word that has same tag as "state"
            if (start_token, tag) in self.transitions:
                viterbi[str(1)][tag] = self.transitions[(start_token, tag)] + self.emissions[(sent[1][0], tag)]
            else:
                viterbi[str(1)][tag] = -float('inf')
            backpointer[str(1)][tag] = start_token

        # Recursion step
        # This loop will run for rest of the tuples (word, pos) after first tuple in "sent"
        for i in xrange(2, len(sent)):
            # This loop will run for all the tags of each word (sent[idx][0]) in dictionary
            for tag in self.dictionary[sent[i][0]]:
                maximum_value = -float("inf")
                maximum_loc = []
                # This loop will run for all the tags in previous word (sent[idx-1][0]) in dictionary
                for prev_tag in self.dictionary[sent[i - 1][0]]:
                    # if any sentance in our trained data has (privious tag, current tag) or (pre_state, state) of given word
                    if (prev_tag, tag) in self.transitions:
                        t = viterbi[str(i - 1)][prev_tag] + self.transitions[(prev_tag, tag)]
                    else:
                        t = -float('inf')
                    if t >= maximum_value:
                        maximum_value = t
                        maximum_loc = prev_tag

                viterbi[str(i)][tag] = maximum_value + self.emissions[(sent[i][0], tag)]
                backpointer[str(i)][tag] = maximum_loc

        t = end_token
        for i in xrange(1, len(sent)):
            t = backpointer[str(len(sent) - i)][t]
            pos_list.append(t)

        for tup in sent:
            sent_tag.append((tup[0], pos_list.pop()))

        #print "viterbi:", viterbi
        #print "backpointer:", backpointer
        #print "sent_tagged", sent_tag

        return sent_tag

    def Test(self, test_set):
        """ Use Viterbi and predict the most likely tag sequence for every sentence. Return a re-tagged test_set. """
        test_set_retagged = []
        # This loop will call Viterbi method and pass each sentence (word, POS) in "test_set" one by one,
        # and save the returned retagged pos in "test_set_retagged"
        a = 0
        for sent in test_set:
            test_set_retagged.append(self.Viterbi(sent))
        return test_set_retagged

def MostCommonClassBaseline(training_set, test_set):
    """ Implement the most common class baseline for POS tagging. Return the test set tagged according to this baseline. """
    pos_counts_dict = defaultdict(dict)
    max_pos_dict = dict()
    test_common_tags = []

    # Dictionary "pos_counts_dict" stores a dictionary for each word that stores counts of each pos of the word
    # This loop runs for each sentence (word, pos) in "training_set"
    for sentence in training_set:
        # This loop runs for each tuple (word, pos) in sentence
        for word_pos in sentence:
            # if word (word_pos[0]) not in "pos_counts_dict"
            if word_pos[0] not in pos_counts_dict:
                pos_counts_dict[word_pos[0]] = defaultdict(int)
            # increment for each tuple (word, pos) in sentence
            pos_counts_dict[word_pos[0]][word_pos[1]] += 1

    # Find most frequent tag associated to each word and store it in "max_pos_dict"
    # This loop runs for each word in "pos_counts_dict"
    for word in pos_counts_dict:
        count = 0
        tag = str()

        # This loop runs for each tag of the word
        for pos in pos_counts_dict[word]:
            if pos_counts_dict[word][pos] > count:
                count = pos_counts_dict[word][pos]
                tag = pos
        max_pos_dict[word] = tag

    # Match tag in "max_pos_dict" for each word of "test_set" and store in "test_common_tags"

    # This loop runs for each sentence (word, pos) in "test_set"
    for sentence in test_set:
        temp_sentence = []
        # This loop runs for no. of tuples (word, pos) in sentence
        for i in xrange(len(sentence)):
            # if word is in "pos_counts_dict" then store tuple (word, max count) in "temp_sentence"
            if sentence[i][0] not in pos_counts_dict:
                print "Word not in training_set:", tup[0]
            else:
                temp_sentence.append((sentence[i][0], max_pos_dict[sentence[i][0]]))
        test_common_tags.append(temp_sentence)

    return test_common_tags

def ComputeAccuracy(test_set, test_set_predicted):
    """ Using the gold standard tags in test_set, compute the sentence and tagging accuracy of test_set_predicted. """
    total_tokens = 0
    error_counts = 0
    error_counts_line = 0

    # This loop runs for no. of sentences in "test_set"
    for idx in xrange(len(test_set)):
        # if sentences doesn't match exactly, count errors for line (error_counts_line) and tokens (error_counts)
        if test_set[idx] != test_set_predicted[idx]:
            error_counts_line += 1
            # This loop runs for no. of tuples (word, pos) in sentence
            for jdx in xrange(len(test_set[idx])):
                total_tokens += 1
                if test_set[idx][jdx] != test_set_predicted[idx][jdx]:
                    error_counts += 1
        else:
            total_tokens += len(test_set[idx])

    sentence_accuracy = 100 - float(error_counts_line) * 100 / len(test_set)
    tagging_accuracy = 100 - float(error_counts) * 100 / (total_tokens - 2 * len(test_set)) # excluding sentence boundary tokens

    print "Sentence Accuracy: %.2f%%" % sentence_accuracy
    print "Tagging Accuracy: %.2f%%" % tagging_accuracy

def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens.
    training_set = treebank_tagged_sents[:3000]  # This is the train-test split that we will use. 
    test_set = treebank_tagged_sents[3000:]
    
    """ Transform the data sets by eliminating unknown words and adding sentence boundary tokens.
    """
    # Extracted the vocabulary form the training_set
    dictionary = defaultdict(int)
    vocabulary = set()
    for sentence in training_set:
        for word_POS in sentence:
            dictionary[word_POS[0]] += 1
    for word in dictionary:
        if dictionary[word] >= 2:  # Treat every word occurring not more than once in training set as an unknown token
            vocabulary.add(word)
    vocabulary.add(unknown_token)  # adding sentence boundary tokens
    vocabulary.add(start_token)
    vocabulary.add(end_token)

    training_set_prep, training_set_prep_1d = PreprocessText(training_set, vocabulary)
    test_set_prep, test_set_prep_1d = PreprocessText(test_set, vocabulary)
    
    """ Print the first sentence of each data set.
    """
    print "Answer 1: Print out the first sentence of each Preprocessed dataset"
    print " ".join(untag(training_set_prep[0]))  # See nltk.tag.util module.
    print " ".join(untag(test_set_prep[0]))

    """ Estimate Bigram HMM from the training set, report level of ambiguity.
    """
    # Answer 3.1.1: Train BigramHMM and estimating A and B matrices and
    # dictionary to maps every word to a set of its candidate tags
    bigram_hmm = BigramHMM()
    bigram_hmm.Train(training_set_prep_1d)

    print "Answer 3.1.2: Calculate the percentage ambiguity"
    print "Percent tag ambiguity in training set is %.2f%%." %bigram_hmm.ComputePercentAmbiguous(training_set_prep_1d)

    print "Answer 3.1.3: Calculate the Joint Probability"
    print "Joint probability of the first sentence is %s." %bigram_hmm.JointProbability(training_set_prep[0])
    
    """ Implement the most common class baseline. Report accuracy of the predicted tags.
    """
    print "Answer 2: Implement the most common class algorithm and calculate accuracy"
    test_set_predicted_baseline = MostCommonClassBaseline(training_set_prep, test_set_prep)
    print "--- Most common class baseline accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_baseline)    

    """ Use the Bigram HMM to predict tags for the test set. Report accuracy of the predicted tags.
    """
    print "Answer 4: Implement Viterbi and Test method of BigramHMM and compute accuracy"
    test_set_predicted_bigram_hmm = bigram_hmm.Test(test_set_prep)
    print "--- Bigram HMM accuracy ---"
    ComputeAccuracy(test_set_prep, test_set_predicted_bigram_hmm)    

if __name__ == "__main__": 
    main()
    