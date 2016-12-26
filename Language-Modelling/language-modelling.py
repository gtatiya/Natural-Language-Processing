import sys
from collections import defaultdict
from math import log, exp
from nltk.corpus import brown

unknown_token = "<UNK>"  # unknown word token.
start_token = "<S>"  # sentence boundary token.
end_token = "</S>"  # sentence boundary token.

""" Implement any helper functions here, e.g., for text preprocessing.
"""

# Introducing unknown tokens <UNK> and adding sentence boundary symbols <S></S> at the beginning and end of every
# sentence, and return sentence by sentence list (dataset_prep) & 1D list of all the tokens (dataset_prep_1d)
def PreprocessText(dataset, vocabulary):
    dataset_prep = []
    dataset_prep_1d = []
    for sentence in dataset:
        dataset_prep_1d.append(start_token)
        for i in xrange(len(sentence)):
            if sentence[i] not in vocabulary:
                sentence[i] = unknown_token
            dataset_prep_1d.append(sentence[i])
        dataset_prep_1d.append(end_token)
        sentence.append(end_token)
        sentence.insert(0,start_token)
        dataset_prep.append(sentence)
    return dataset_prep, dataset_prep_1d

class BigramLM:
    def __init__(self, vocabulary = set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.log_probs = {}

        self.bigram_prob = defaultdict(float)
        self.dataset_size = int()
    """ Implement the functions EstimateBigrams, CheckDistribution, Perplexity and any 
    other function you might need here.
    """

    # Count tokens for unigram_counts and bigram_counts
    def count_tokens(self, dataset):
        self.dataset_size = len(dataset)
        for i in xrange(len(dataset) - 1):
            self.unigram_counts[dataset[i]] += 1
            self.bigram_counts[(dataset[i], dataset[i + 1])] += 1

        self.unigram_counts[dataset[-1]] += 1

        del self.unigram_counts[end_token] # deleting counts for <S>
        del self.bigram_counts[end_token, start_token] # deleting counts for (</S>, <S>)

    # Calculating bigrams probability for given smoothing method. Argument *args is used for held_out_set in Deleted interpolation
    def EstimateBigrams(self, smoothing, *args):
        if smoothing == 'no smoothing':
            for (a,b) in self.bigram_counts:
                self.bigram_prob[(a,b)] = float(self.bigram_counts[(a,b)])/self.unigram_counts[a]

        elif smoothing == 'Laplace smoothing':
            """
            FEEDBACK: 1.4
            The point of laplace smoothing is to take care of bigrams that were not seen in training.
            So, the laplace probability for unseen bigrams should be 1/(count a + vocab size)
            """
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = (float(self.bigram_counts[(a, b)]) + 1) / (self.unigram_counts[a] + len(self.vocabulary))

        elif smoothing == 'Simple linear interpolation':
            for (a, b) in self.bigram_counts:
                self.bigram_prob[(a, b)] = 0.5 * (float(self.bigram_counts[(a, b)]) / self.unigram_counts[a]) + 0.5 * (float(self.unigram_counts[b]) / (self.dataset_size))

        elif smoothing == 'Deleted interpolation':
            held_out_set_prep_1d = args[0]
            held_out_size = len(args[0])

            held_out_unigram_counts = defaultdict(int)
            held_out_bigram_counts = defaultdict(int)

            # Count tokens for unigram_counts and bigram_counts for held_out_set
            for i in xrange(len(held_out_set_prep_1d) - 1):
                held_out_unigram_counts[held_out_set_prep_1d[i]] += 1
                held_out_bigram_counts[(held_out_set_prep_1d[i], held_out_set_prep_1d[i + 1])] += 1
            held_out_unigram_counts[held_out_set_prep_1d[-1]] += 1
            del held_out_unigram_counts[end_token]  # deleting counts for <S>
            del held_out_bigram_counts[end_token, start_token]  # deleting counts for (</S>, <S>)

            # Estimating the interpolation weights (SLP Figure 5.19) using the held out corpus for unigram and bigram
            lambda1 = 0.0
            lambda2 = 0.0
            for (a, b) in held_out_bigram_counts:
                if held_out_unigram_counts[a] == 1:
                    value2 = 0
                else:
                    value2 = float(held_out_bigram_counts[(a, b)] - 1) / (held_out_unigram_counts[a] - 1)
                value1 = float(held_out_unigram_counts[b] - 1) / (held_out_size - 1)

                # Taking the maximum value
                if value1 > value2:
                    lambda1 += held_out_bigram_counts[(a, b)]
                else:
                    lambda2 += held_out_bigram_counts[(a, b)]

            # Normalizing
            t = lambda1 + lambda2
            self.lambda1 = float(lambda1) / t
            self.lambda2 = float(lambda2) / t

            for (a, b) in held_out_bigram_counts:
                self.bigram_prob[(a, b)] = self.lambda2*float(held_out_bigram_counts[(a,b)])/held_out_unigram_counts[a] + self.lambda1*float(held_out_unigram_counts[b])/ held_out_size

    # Checking the validity of your bigram estimates from EstimateBigrams
    def CheckDistribution(self):
        # ret contains the sum of all probabilities for every valid unigram
        ret = defaultdict(int)
        for (first, _), val in self.bigram_prob.iteritems():
            ret[first] += val

        not_valid = 0
        for unigram, proba in ret.iteritems():
            proba = round(proba)
            if proba != 1.0:
                print "Distribution is not valid for unigram:", unigram
                not_valid = 1
                break

        if not_valid == 1:
            print("Distribution is not valid for your bigram estimates")
        else:
            print("Distribution is valid for your bigram estimates")


    # Computing the perplexity given a test corpus
    """
    FEEDBACK: 1.2.3
    If a bigram is not seen in training,
    you should return -infinity for the log perplexity (log(0) = -inf)
    """
    def Perplexity(self, testset):
        perplexity = 0
        N = 0
        for i in xrange(len(testset) - 1):
            if (testset[i], testset[i + 1]) != (end_token, start_token):
                if ((testset[i], testset[i + 1]) in self.bigram_prob):
                    N += 1
                    perplexity += log(1 / self.bigram_prob[(testset[i], testset[i + 1])])
        perplexity = exp(perplexity / N)
        return perplexity

def main():
    training_set = brown.sents()[:50000]
    held_out_set = brown.sents()[-6000:-3000]
    test_set = brown.sents()[-3000:]

    """ Transform the data sets by eliminating unknown words and adding sentence boundary 
    tokens.
    """
    #Extracted the vocabulary form the training_set
    dictionary = defaultdict(int)
    vocabulary = set()
    for sentence in training_set:
        for word in sentence:
            dictionary[word] += 1
    for word in dictionary:
        if dictionary[word] >= 2:  # Treat every word occurring not more than once in the training set as an unknown token
            vocabulary.add(word)
        vocabulary.add(unknown_token) # adding sentence boundary tokens
    vocabulary.add(start_token)
    vocabulary.add(end_token)

    training_set_prep, training_set_prep_1d = PreprocessText(training_set, vocabulary)
    held_out_set_prep, held_out_set_prep_1d = PreprocessText(held_out_set, vocabulary)
    test_set_prep, test_set_prep_1d = PreprocessText(test_set, vocabulary)

    """ Print the first sentence of each data set.
    """
    print "Answer 1.1: Print out the first sentence of each Preprocessed dataset"
    print training_set_prep[0]
    print held_out_set_prep[0]
    print test_set_prep[0]

    """ Estimate a bigram_lm object, check its distribution, compute its perplexity.
    """
    # "Answer 1.2.1 and 1.3: Estimating bigram MLE probabilities"
    bigram_lm = BigramLM(vocabulary)
    bigram_lm.count_tokens(training_set_prep_1d)
    bigram_lm.EstimateBigrams('no smoothing')

    print "Answer 1.2.2 and 1.3: Checking the validity of your bigram estimates from EstimateBigrams"
    print "CheckDistribution Result:", bigram_lm.CheckDistribution()

    print "Answer 1.2.3 and 1.3: Computing the perplexity given a test corpus"
    print "Testing set perplexity:", bigram_lm.Perplexity(test_set_prep_1d)

    """ Print out perplexity after Laplace smoothing.
    """
    print "Answer 1.4: Estimating bigram probabilities with Laplace smoothing and compute its perplexity"
    bigram_lm.EstimateBigrams('Laplace smoothing')
    print "Testing set perplexity with Laplace smoothing:", bigram_lm.Perplexity(test_set_prep_1d)

    """ Print out perplexity after simple linear interpolation (SLI) with lambda = 0.5.
    """
    print "Answer 1.5: Estimating bigram probabilities with Simple linear interpolation and compute its perplexity"
    bigram_lm.EstimateBigrams('Simple linear interpolation')
    print "Testing set perplexity with SLI, lambda = 0.5:", bigram_lm.Perplexity(test_set_prep_1d)

    """ Estimate interpolation weights using the deleted interpolation algorithm on the 
    held out set and print out.
    """
    print "Answer 1.6: Computing interpolation weights, bigram probabilities on held_out_set with Deleted interpolation and compute its perplexity"
    bigram_lm.EstimateBigrams('Deleted interpolation', held_out_set_prep_1d)
    print "Lambda 1 = ", bigram_lm.lambda1
    print "Lambda 2 = ", bigram_lm.lambda2

    """ Print out perplexity after simple linear interpolation (SLI) with the estimated
    interpolation weights.
    """
    print "Perplexity with Deleted interpolation:", bigram_lm.Perplexity(test_set_prep_1d)

if __name__ == "__main__": 
    main()