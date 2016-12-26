import sys, re
import nltk
from nltk.corpus import treebank
from collections import defaultdict
from nltk import induce_pcfg
from nltk.grammar import Nonterminal
from nltk.tree import Tree
from math import exp, pow, log

unknown_token = "<UNK>"  # unknown word token.

""" Removes all function tags e.g., turns NP-SBJ into NP.
"""         
def RemoveFunctionTags(tree):
    for subtree in tree.subtrees():  # for all nodes of the tree
        # if it's a preterminal node with the label "-NONE-", then skip for now
        if subtree.height() == 2 and subtree.label() == "-NONE-": continue
        nt = subtree.label()  # get the nonterminal that labels the node
        labels = re.split("[-=]", nt)  # try to split the label at "-" or "="
        if len(labels) > 1:  # if the label was split in two e.g., ["NP", "SBJ"]
            subtree.set_label(labels[0])  # only keep the first bit, e.g. "NP"

""" Return true if node is a trace node.
"""         
def IsTraceNode(node):
    # return true if the node is a preterminal node and has the label "-NONE-"
    return node.height() == 2 and len(node) == 1 and node.label() == "-NONE-"

""" Deletes any trace node children and returns true if all children were deleted.
"""
def RemoveTraces(node):
    if node.height() == 2:  # if the node is a preterminal node
        return False  # already a preterminal, cannot have a trace node child.
    i = 0
    while i < len(node):  # iterate over the children, node[i]
        # if the child is a trace node or it is a node whose children were deleted
        if IsTraceNode(node[i]) or RemoveTraces(node[i]): 
            del node[i]  # then delete the child
        else: i += 1
    return len(node) == 0  # return true if all children were deleted
    
""" Preprocessing of the Penn treebank.
"""
def TreebankNoTraces():
    tb = []
    for t in treebank.parsed_sents():
        if t.label() != "S": continue
        RemoveFunctionTags(t)
        RemoveTraces(t)
        t.collapse_unary(collapsePOS = True, collapseRoot = True)
        t.chomsky_normal_form()
        tb.append(t)
    return tb
        
""" Enumerate all preterminal nodes of the tree.
""" 
def PreterminalNodes(tree):
    for subtree in tree.subtrees():
        if subtree.height() == 2:
            yield subtree
    
""" Print the tree in one line no matter how big it is
    e.g., (VP (VB Book) (NP (DT that) (NN flight)))
"""         
def PrintTree(tree):
    if tree.height() == 2: return "(%s %s)" %(tree.label(), tree[0])
    return "(%s %s)" %(tree.label(), " ".join([PrintTree(x) for x in tree]))

""" Preprocessing text treating every word that occurs not more than
    once as an unknown token
"""
def PreprocessText(dataset, vocabulary):
    for tree in dataset:
        for node in PreterminalNodes(tree):
            if node[0] not in vocabulary:
                node[0] = unknown_token
    return dataset


def Recursive_BuildTree(cky_table, sent, i, j, nonterminal):
    if j - i == 1:
        Tree_end = Tree(nonterminal.symbol(), [sent[i]])
    else:
        (k, B, C) = cky_table[(i, j)][nonterminal]
        Tree_end = Tree(nonterminal.symbol(), [Recursive_BuildTree(cky_table, sent, i, k, B), Recursive_BuildTree(cky_table, sent, k, j, C)])
    return Tree_end


def file_generate(file_test, file_gold, bucket, invertedGrammar):
    for tree in bucket:
        predict_tree = invertedGrammar.BuildTree(invertedGrammar.Parse(tree.leaves())[1], tree.leaves())
        tree.un_chomsky_normal_form()
        if predict_tree != None:
            predict_tree.un_chomsky_normal_form()
            file_test.write(PrintTree(predict_tree) + '\n')
            file_gold.write(PrintTree(tree) + '\n')
        else:
            file_test.write('\n')

    file_test.close()
    file_gold.close()

class InvertedGrammar:
    def __init__(self, pcfg):
        #""" Implement:
        self._pcfg = pcfg
        self._r2l = defaultdict(list)  # maps RHSs to list of LHSs
        self._r2l_lex = defaultdict(list)  # maps lexical items to list of LHSs
        self.BuildIndex()  # populates self._r2l and self._r2l_lex according to pcfg
        #"""

    def PrintIndex(self, filename):
        f = open(filename, "w")
        for rhs, prods in self._r2l.iteritems():
            f.write("%s\n" %str(rhs))
            for prod in prods:
                f.write("\t%s\n" % str(prod))
            f.write("---\n")
        for rhs, prods in self._r2l_lex.iteritems():
            f.write("%s\n" % str(rhs))
            for prod in prods:
                f.write("\t%s\n" % str(prod))
            f.write("---\n")
        f.close()
        
    def BuildIndex(self):
        """ Build an inverted index of your grammar that maps right hand sides of all 
        productions to their left hands sides.
        """
        for production in self._pcfg.productions():
            if production.is_lexical():  # "is_lexical()" Return True if the right-hand contain at least one terminal token
                self._r2l_lex[production.rhs()].append(production)
            else:
                self._r2l[production.rhs()].append(production)
        self.PrintIndex('index')
            
    def Parse(self, sent):
        """ Implement the CKY algorithm for PCFGs, populating the dynamic programming 
        table with log probabilities of every constituent spanning a sub-span of a given 
        test sentence (i, j) and storing the appropriate back-pointers. 
        """
        table = defaultdict(dict)
        back_pointer = defaultdict(dict)
        for j in xrange(1, len(sent) + 1):
            for A in self._r2l_lex[tuple([sent[j - 1]])]:
                table[(j - 1, j)][A.lhs()] = log(A.prob())
                '''table' is a dictionary that stores dictionary for each '(j-1, j)' that stores log proba.
                for each non-terminal 'A.lhs()' of the word
                '''
            if j >= 2:
                for i in xrange(j - 2, -1, -1):
                    for k in xrange(i + 1, j):
                        for B in table[(i, k)]:
                            for C in table[(k, j)]:
                                for A in self._r2l[(B, C)]:
                                    a_log_prob = log(A.prob()) + table[(i, k)][B] + table[(k, j)][C]
                                    if A.lhs() not in table[(i, j)]:
                                        back_pointer[(i, j)][A.lhs()] = (k, B, C)
                                        table[(i, j)][A.lhs()] = a_log_prob
                                    else:
                                        if table[(i, j)][A.lhs()] < a_log_prob:
                                            back_pointer[(i, j)][A.lhs()] = (k, B, C)
                                            table[(i, j)][A.lhs()] = a_log_prob
        return table, back_pointer
        
    @staticmethod
    def BuildTree(cky_table, sent):
        """ Build a tree by following the back-pointers starting from the largest span 
        (0, len(sent)) and recursing from larger spans (i, j) to smaller sub-spans 
        (i, k), (k, j) and eventually bottoming out at the preterminal level (i, i+1).
        """
        if Nonterminal('S') not in cky_table[(0, len(sent))]:
            print 'Parsing error: not start with S'
            return None
        else:
            return Recursive_BuildTree(cky_table, sent, 0, len(sent), Nonterminal('S'))

def main():
    treebank_parsed_sents = TreebankNoTraces()
    training_set = treebank_parsed_sents[:3000]
    test_set = treebank_parsed_sents[3000:]
    
    """ Transform the data sets by eliminating unknown words.
    """
    # Extracted the vocabulary form the training_set
    dic = defaultdict(int)
    vocabulary = set()
    for sent in training_set:
        for word in sent.leaves():
            if word in dic:
                vocabulary.add(word)  # "word" is added to "vocabulary" if it is in "dic." So, "word" has occured more than once
            dic[word] += 1

    training_set_prep = PreprocessText(training_set, vocabulary)
    test_set_prep = PreprocessText(test_set, vocabulary)

    """ Print the first sentence of each data set.
    """
    print "Answer 1: Print out the first trees of each Preprocessed dataset"
    print "training_set_prep: ", PrintTree(training_set_prep[0])
    print "test_set_prep: ", PrintTree(test_set_prep[0])
    
    """ Implement your solutions to problems 2-4.
    """
    print "Answer 2: Training"
    productions = []

    for tree in training_set_prep:
        productions += tree.productions()

    S = Nonterminal("S")
    grammar = induce_pcfg(S, productions)

    NP_list = []
    for production in grammar.productions():
        if str(production.lhs()) == 'NP':
            NP_list.append(production)

    print "Number of productions for the NP nonterminal: " + str(len(NP_list))

    NP_list.sort(key=lambda x: x.prob(), reverse=True)  # "key=lambda x: x.prob()" is called for each item in "NP_list" to sort
    print "Most probable 10 productions for the NP nonterminal:"
    print NP_list[:10]

    print "Answer 3: Testing"
    print "Answer 3.1: Implement BuildIndex method of InvertedGrammar class"
    invertedGrammar = InvertedGrammar(grammar)

    test_sentence = "Terms were n't disclosed .".split()
    print "Answer 3.2: Implement Parse method of InvertedGrammar class"
    table, tree = invertedGrammar.Parse(test_sentence)
    print "Log probability of nonterminal S for 5-token sentence: ", table[(0, len(test_sentence))][Nonterminal('S')]

    print "Answer 3.3: Implement BuildTree method of InvertedGrammar class"
    print "Parse tree for the sentence:\n", invertedGrammar.BuildTree(tree, test_sentence)

    print "Answer 3.4: Number of sentences in bucket:"
    bucket1 = []
    bucket2 = []
    bucket3 = []
    bucket4 = []
    bucket5 = []

    for tree in test_set_prep:
        len_sent = len(tree.leaves())
        if len_sent > 0 and len_sent < 10:
            bucket1.append(tree)
        elif len_sent >= 10 and len_sent < 20:
            bucket2.append(tree)
        elif len_sent >= 20 and len_sent < 30:
            bucket3.append(tree)
        elif len_sent >= 30 and len_sent < 40:
            bucket4.append(tree)
        elif len_sent >= 40:
            bucket5.append(tree)

    print "Bucket1:", len(bucket1)
    print "Bucket2:", len(bucket2)
    print "Bucket3:", len(bucket3)
    print "Bucket4:", len(bucket4)
    print "Bucket5:", len(bucket5)

    # "Answer 3.5: Generating test_* and gold_* files:"

    bucket = []
    bucket.append(bucket1)
    bucket.append(bucket2)
    bucket.append(bucket3)
    bucket.append(bucket4)
    bucket.append(bucket5)

    for i in xrange(1, len(bucket) + 1):
        file_test = open('test_' + str(i), "w")
        file_gold = open('gold_' + str(i), "w")
        print "Processing Bucket:", i
        file_generate(file_test, file_gold, bucket[i - 1], invertedGrammar)

"""
3.4/3.5:
Expected:
Bucket1: Bracketing FMeasure    80.95
             Average crossing        0.27
Bucket2: Bracketing FMeasure    80.61
             Average crossing        1.05
Bucket3: Bracketing FMeasure    71.38
             Average crossing        3.03
Bucket4: Bracketing FMeasure    62.19
             Average crossing        6.66
Bucket5: Bracketing FMeasure    61.50
             Average crossing        8.41
"""

if __name__ == "__main__":
    main()
