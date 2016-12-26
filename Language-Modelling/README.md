# Language Modelling

In this project, I have built a bigram language model for the Brown corpus. Brown corpus is a 57K-sentence data set in nltk package in Python.
In have trained my model using the first 50K sentences of the corpus, brown.sents()[:50000]. I calculated bigrams probability using Maximum Likelihood Estimate, Laplace smoothing and Simple linear interpolation with weights of (0.5, 0.5), and calculated the perplexity for each on the test set. Perplexity is the most common intrinsic evaluation metric in language modelling. The better model is the one that assigns the test data a higher probability. So, lower perplexity represents the better language model.
The test set is the last 3K ?sentences of the corpus, brown.sents()[-3000:]
I have also implemented Deleted Interpolation Algorithm on held out set, brown.sents()[-6000:-3000], and calculated its perplexity.

Dataset | Perplexity
--- | ---
Testing set perplexity | 55.6767553189
Testing set perplexity with Laplace smoothing | 614.656135069
Testing set perplexity with SLI, lambda = 0.5 | 84.6359546331
Held out set Deleted interpolation: Lambda 1 = 0.308362081621 Lambda 2 = 0.691637918379 | 59.4456071233
