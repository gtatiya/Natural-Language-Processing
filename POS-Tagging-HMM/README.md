# Part of Speech Tagging with Hidden Markov model

In this project, I implemented POS Tagging with Hidden Markov models using the Treebank corpora. The Treebank corpora is a 3914 sentences data set in nltk package in Python that provide a syntactic parse for each sentence. 
I have trained my model using the first 3K sentences of the corpus, treebank.tagged_sents[:3000]. I calculated Transition probability matrix A and Observation likelihood (emission probability) matrix B to train my model. 
Then, implement the Viterbi method of Bigram Hidden Markov model for POS tagging a test sentence using the A and B matrices, and predicted POS tags for last 3k sentences of the corpus, treebank.tagged_sents[3000:].
I computed the accuracy of Most Common Class Baseline and Hidden Markov model. By using Bigram Hidden Markov Model (HMM), the sentence accuracy is improved by 9.96% and the tagging accuracy is improved by 4.82% as compared to most common class baseline accuracy.

<b>Most common class baseline accuracy</b><br>
Sentence Accuracy: 7.00%<br>
Tagging Accuracy: 85.21%<br>
<br>
<b>Bigram HMM accuracy</b><br>
Sentence Accuracy: 16.96%<br>
Tagging Accuracy: 90.03%