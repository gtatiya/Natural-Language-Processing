import nltk
from nltk.corpus import treebank, stopwords
from nltk.tag.util import untag
from collections import defaultdict
from random import randint
import matplotlib.pyplot as plt

from bigram_hmm import BigramHMM
from context_words import contextwords

unknown_token = "<UNK>"
start_token = "<S>"
end_token = "</S>"

"""
Remove trace tokens and tags from the treebank as these are not necessary.
tuples (word, POS) with POS "-NONE-" are removed.
There are 3914 sentences in treebank.
"""
def TreebankNoTraces():
    return [[x for x in sent if x[1] != "-NONE-"] for sent in treebank.tagged_sents()]

"""
This function matches the passed word to all confusion sets
and returns the confusing_set if match is found otherwise returns None
"""
def confusing_word_check(word, confusing_sets_pruned):
    for confusing_set in confusing_sets_pruned:
        if word in confusing_set:
            return confusing_set
    return None

"""
This function generate spelling errors in test_set by randomly replacing the actual word with one of
confusing_sets_pruned words, if there is a word in test_set that is also in confusing_sets_pruned.
"""
def error_generation(test_set, confusing_sets_pruned):
    erroneous_test_set = []
    for line in test_set:
        erroneous_line = []
        for word in line:
            confusing_set = confusing_word_check(word[0], confusing_sets_pruned)
            if confusing_set == None:
                erroneous_line.append(word)
            else:
                random_index = randint(0, len(confusing_set) - 1)
                erroneous_line.append((confusing_set[random_index], word[1]))
        erroneous_test_set.append(erroneous_line)
    return erroneous_test_set

"""
This function introduce unknown tokens <UNK> for every word occurring not more than once in dataset
and add sentence boundary symbols <S></S> at the beginning and end of every sentence
"""
def PreprocessText(dataset, vocabulary):
    dataset_prep = []
    for sentence in dataset:
        processed_sent = []
        for i in xrange(len(sentence)):
            if sentence[i][0] not in vocabulary or vocabulary[sentence[i][0]] <= 1:
                processed_sent.append((unknown_token, sentence[i][1]))
            else:
                processed_sent.append(sentence[i])
        processed_sent.append((end_token, end_token))  # start and end symbols are tagged as themselves
        processed_sent.insert(0, (start_token, start_token))
        dataset_prep.append(processed_sent)
    return dataset_prep

"""
This function removes tag of each word of the corpus
"""
def untag_dataset(corpus):
    untagged_corpus = []
    for sent in corpus:
        untagged_sent = untag(sent)
        untagged_corpus.append(untagged_sent)
    return untagged_corpus

"""
This function takes test_set_prep_erroneous, vocabulary and pruned_confusion_sets as arguments.
For each word found in test_set_prep_erroneous and pruned_confusion_sets, it selects the most frequent word among corresponding
pruned_confusion_sets using vocabulary dictionary and replaces with the founded word to built a testset_corrected.
"""
def Baseline(test_set, vocabulary, confusion_sets):
    testset_corrected = []
    for line in test_set:
        corrected_line = []
        for word in line:
            confusing_set = confusing_word_check(word, confusion_sets)
            if confusing_set == None:
                corrected_line.append(word)
            else:
                best_word = str()
                max_freq = 0
                for confusing_word in confusing_set:
                    if (vocabulary[confusing_word] > max_freq) and (confusing_word in vocabulary):
                        max_freq = vocabulary[confusing_word]
                        best_word = confusing_word
                corrected_line.append(best_word)
        testset_corrected.append(corrected_line)
    return testset_corrected

"""
This function predicts the accuracy of our prediction
"""
def ComputeAccuracy(test_set_prep_untag, test_set_prep_erroneous_untag, testset_corrected):
    corrections = 0
    errors = 0

    for i in xrange(len(test_set_prep_untag)):
        for j in xrange(1, (len(test_set_prep_untag[i]) - 1)):
            if test_set_prep_untag[i][j] != test_set_prep_erroneous_untag[i][j]:
                errors += 1
                if test_set_prep_untag[i][j] == testset_corrected[i][j]:
                    corrections += 1

    accuracy = (float(corrections) * 100) / errors

    #print "Number of wrong words:", errors
    #print "Number of words corrected:", corrections
    print "Percent error corrected in test set is %.2f%%." % accuracy

    return accuracy


def main():
    treebank_tagged_sents = TreebankNoTraces()  # Remove trace tokens
    train_set = treebank_tagged_sents
    test_set = treebank_tagged_sents[:3000]

    vocabulary = defaultdict(int)
    for sent in train_set:
        for word in sent:
            vocabulary[word[0]] += 1  # vocabulary is a dictionary with words as index and word frequency as its value

    """
    Reading the file and store a 2D list (confused_words):  [len][confusing words]
    confused_words
    """
    confused_words = []
    file = open("OftenConfusedWords.txt", "r")
    for line in file:
        line = line.replace("\n", "")
        confused_words.append(line.split(", "))

    """
    Removing all the words from confused_words that are not in vocabulary and saving a new common confusion sets
    """
    confusing_sets_pruned = []
    for c_set in confused_words:
        pruned_set = []
        for word in c_set:
            if word in vocabulary and vocabulary[word] > 1:  # taking words occurring more than once
                pruned_set.append(word)
        if pruned_set not in confusing_sets_pruned and len(pruned_set) > 0:
            confusing_sets_pruned.append(pruned_set)
    print "Number of confusing sets:", len(confusing_sets_pruned)


    test_set_erroneous = error_generation(test_set, confusing_sets_pruned)

    train_set_prep = PreprocessText(train_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)
    test_set_prep_erroneous = PreprocessText(test_set_erroneous, vocabulary)

    train_set_prep_untag = untag_dataset(train_set_prep)
    test_set_prep_untag = untag_dataset(test_set_prep)
    test_set_prep_erroneous_untag = untag_dataset(test_set_prep_erroneous)

    accuracy_values = []

    testset_corrected1 = Baseline(test_set_prep_erroneous_untag, vocabulary, confusing_sets_pruned)
    print "=" * 50
    print "The Baseline Accuracy:"
    accuracy = ComputeAccuracy(test_set_prep_untag, test_set_prep_erroneous_untag, testset_corrected1)
    accuracy_values.append(accuracy)

    """
    Storing a 1D list of uninformative_words
    """
    uninformative_words = stopwords.words('english')
    uninformative_words.extend(['<S>', '</S>', '<UNK>', '.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{',
              '}'])

    k = 3  # k is word window of the target word
    minimum_occurrences = 10

    context_words = contextwords(train_set_prep_untag, vocabulary, confusing_sets_pruned, uninformative_words, k,
                                 minimum_occurrences)
    testset_corrected2 = context_words.test(test_set_prep_erroneous_untag)
    print "=" * 50
    print "The Context words Accuracy:"
    accuracy = ComputeAccuracy(test_set_prep_untag, test_set_prep_erroneous_untag, testset_corrected2)
    accuracy_values.append(accuracy)

    bigram_hmm = BigramHMM(train_set_prep, confusing_sets_pruned)
    testset_corrected3 = bigram_hmm.test(test_set_prep_erroneous_untag)
    print "=" * 50
    print "The Bigram Hidden Markov Model Accuracy:"
    accuracy = ComputeAccuracy(test_set_prep_untag, test_set_prep_erroneous_untag, testset_corrected3)
    accuracy_values.append(accuracy)

    label = ['Baseline', 'Context words', 'Bigram HMM']
    width = 1 / 1.5
    plt.ylim(0, 100)
    plt.bar(xrange(1, len(label) + 1), accuracy_values, width, color="blue", align='center')
    plt.xticks(xrange(1, len(label) + 1), label)
    plt.ylabel("Accuracy %", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()