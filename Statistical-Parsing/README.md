# Statistical parsing

In this project, I built a Probabilistic Context-Free Grammars (PCFG) parser using the Treebank corpora. The Treebank corpora is a 3914 sentences data set in nltk package in Python that provide a syntactic parse for each sentence. 
I pre-processed the Treebank corpora and used its first 3K sentences to learn a PCFG using the function induce_pcfg provided by NLTK’s grammar module.
I implemented the probabilistic CKY algorithm for parsing a test sentence using your learned PCFG.
