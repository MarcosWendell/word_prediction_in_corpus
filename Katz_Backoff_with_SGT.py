import json
import time
import numpy as np
from random import shuffle
from copy import deepcopy
from collections import Counter, defaultdict, OrderedDict

train_path = '/Users/giovannicassani/Desktop/CL/LanguageModel/training_corpus.json'
test_path = '/Users/giovannicassani/Desktop/CL/LanguageModel/test_corpus.json'

class Corpus(object):
    
    """
    This class creates a corpus object read off a .json file consisting of a list of lists,
    where each inner list is a sentence encoded as a list of strings.
    """
    
    def __init__(self, path, n, t, bos_eos=True, vocab=None):
        
        """
        A Corpus object has the following attributes:
         - vocab: set or None (default). If a set is passed, words in the input file not 
                         found in the set are replaced with the UNK string
         - path: str, the path to the .json file used to build the corpus object
         - t: int, words with frequency count < t are replaced with the UNK string
         - ngram_size: int, 2 for bigrams, 3 for trigrams, and so on.
         - bos_eos: bool, default to True. If False, bos and eos symbols are not 
                     prepended and appended to sentences.
         - sentences: list of lists, containing the input sentences after lowercasing and 
                         splitting at the white space
         - frequencies: Counter, mapping tokens to their frequency count in the corpus
        """
        
        self.vocab = vocab        
        self.path = path
        self.ngram_size = n
        self.t = t
        self.bos_eos = bos_eos
        
        self.sentences = self.read()
        # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
    
        self.frequencies = self.freq_distr()
        # output --> Counter('the': 485099, 'of': 301877, 'i': 286549, ...)
        # the numbers are made up, they aren't the actual frequency counts
        
        if self.t or self.vocab:
            # input --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'park', '.'], ...]
            self.sentences = self.filter_words()
            # output --> [['i', 'am', 'home' '.'], ['you', 'went', 'to', 'the', 'UNK', '.'], ...]
            # supposing that park wasn't frequent enough or was outside of the training 
            # vocabulary, it gets replaced by the UNK string
            
        if self.bos_eos:
            # input --> [['i', 'am', 'home' '.'], ...]
            self.sentences = self.add_bos_eos()
            # output --> [['bos', 'i', 'am', 'home', '.', 'eos'], ...]
                    
    def read(self):
        
        """
        Reads the sentences off the .json file, replaces quotes, lowercases strings and splits 
        at the white space. Returns a list of lists.
        """
        
        if self.path.endswith('.json'):
            sentences = json.load(open(self.path, 'r'))                
        else:   
            sentences = []
            with open(self.path, 'r', encoding='latin-1') as f:
                for line in f:
                    print(line[:20])
                    # first strip away newline symbols and the like, then replace ' and " with the empty 
                    # string and get rid of possible remaining trailing spaces 
                    line = line.strip().translate({ord(i): None for i in '"\'\\'}).strip(' ')
                    # lowercase and split at the white space (the corpus has ben previously tokenized)
                    sentences.append(line.lower().split(' '))
        
        return sentences
    
    def freq_distr(self):
        
        """
        Creates a counter mapping tokens to frequency counts
        
        count = Counter()
        for sentence in self.sentences:
            for word in sentence:
                count[w] += 1
        """
    
        return Counter([word for sentence in self.sentences for word in sentence])
        
    
    def filter_words(self):
        
        """
        Replaces illegal tokens with the UNK string. A token is illegal if its frequency count
        is lower than the given threshold and/or if it falls outside the specified vocabulary.
        The two filters can be both active at the same time but don't have to be. To exclude the 
        frequency filter, set t=0 in the class call.
        """
                
        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                if self.t and self.vocab:
                    # check that the word is frequent enough and occurs in the vocabulary
                    filtered_sentence.append(
                        word if self.frequencies[word] > self.t and word in self.vocab else 'UNK'
                    )
                else:
                    if self.t:
                        # check that the word is frequent enough
                        filtered_sentence.append(word if self.frequencies[word] > self.t else 'UNK')
                    else:
                        # check if the word occurs in the vocabulary
                        filtered_sentence.append(word if word in self.vocab else 'UNK')
                        
            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)
    
        return filtered_sentences
    
    def add_bos_eos(self):
        
        """
        Adds the necessary number of BOS symbols and one EOS symbol.
        
        In a bigram model, you need one bos and one eos; in a trigram model you need two bos and one eos, 
        and so on...
        """
        
        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = ['#bos#']*(self.ngram_size-1) + sentence + ['#eos#']
            padded_sentences.append(padded_sentence)
    
        return padded_sentences

class LM(object):
    
    """
    Creates a language model object which can be trained and tested.
    The language model has the following attributes:
     - vocab: set of strings
     - lam: float, indicating the constant to add to transition counts to smooth them (default to 1)
     - ngram_size: int, the size of the ngrams
    """
    
    def __init__(self, n, vocab=None, smooth='Laplace', lam=1):
        
        self.vocab = vocab
        self.lam = lam
        self.ngram_size = n
        
    def get_ngram(self, sentence, i):
        
        """        
        Takes in a list of string and an index, and returns the history and current 
        token of the appropriate size: the current token is the one at the provided 
        index, while the history consists of the n-1 previous tokens. If the ngram 
        size is 1, only the current token is returned.
        
        Example:
        input sentence: ['bos', 'i', 'am', 'home', 'eos']
        target index: 2
        ngram size: 3
        
        ngram = ['bos', 'i', 'am']  
        #from index 2-(3-1) = 0 to index i (the +1 is just because of how Python slices lists) 
        
        history = ('bos', 'i')
        target = 'am'
        return (('bos', 'i'), 'am')
        """
        
        if self.ngram_size == 1:
            return sentence[i]
        else:
            ngram = sentence[i-(self.ngram_size-1):i+1]
            history = tuple(ngram[:-1])
            target = ngram[-1]
            return (history, target)
                    
    def update_counts(self, corpus):
        
        """        
        Creates a transition matrix with counts in the form of a default dict mapping history
        states to current states to the co-occurrence count (unless the ngram size is 1, in which
        case the transition matrix is a simple counter mapping tokens to frequencies. 
        The ngram size of the corpus object has to be the same as the language model ngram size.
        The input corpus (passed by providing the corpus object) is processed by extracting ngrams
        of the chosen size and updating transition counts.
        
        This method creates three attributes for the language model object:
         - counts: dict, described above
         - vocab: set, containing all the tokens in the corpus
         - vocab_size: int, indicating the number of tokens in the vocabulary
        """
        
        if self.ngram_size != corpus.ngram_size:
            raise ValueError("The corpus was pre-processed considering an ngram size of {} while the "
                             "language model was created with an ngram size of {}. \n"
                             "Please choose the same ngram size for pre-processing the corpus and fitting "
                             "the model.".format(corpus.ngram_size, self.ngram_size))
        
        self.counts = defaultdict(dict) if self.ngram_size > 1 else Counter()
        for sentence in corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    self.counts[ngram] += 1
                else:
                    # it's faster to try to do something and catch an exception than to use an if statement to check
                    # whether a condition is met beforehand. The if is checked everytime, the exception is only catched
                    # the first time, after that everything runs smoothly
                    try:
                        self.counts[ngram[0]][ngram[1]] += 1
                    except KeyError:
                        self.counts[ngram[0]][ngram[1]] = 1
        
        # first loop through the sentences in the corpus, than loop through each word in a sentence
        self.vocab = {word for sentence in corpus.sentences for word in sentence}
        self.vocab_size = len(self.vocab)
    
    def get_unigram_probability(self, ngram):
        
        """        
        Compute the probability of a given unigram in the estimated language model using
        Laplace smoothing (add k).
        """
        
        tot = sum(list(self.counts.values())) + (self.vocab_size*self.lam)
        try:
            ngram_count = self.counts[ngram] + self.lam
        except KeyError:
            ngram_count = self.lam
            print(ngram_count, tot)
        
        return ngram_count/tot
    
    def get_bigram_probability(self, history, target):
        
        """        
        Compute the probability of a given bigram in the estimated language model using
        Katz Backoff algorithm.
        
        The Katz Backoff algorithm work as follow for a bigram model:
        
        P_katz_2(history, target) = { 
            P_gt[r] / obsBi,      if r > 0
            alpha(history) * P_mle[target],  if r == 0   
                    }
                    
        Where P_gt is the bigram count (r) in the corpus subtracted by the mass probability for that
        especificy count in the corpus; P_mle is the MLE for the target in the unigram model; and alpha
        is the normalizing constant governing how the remaining probability mass should be distributed
        to the unseen N-grams.
        
        - history: previous observed word in the corpus
        - target: word that we want to predict the probability of occurance
        """
        
        # Calculate the total bigrams count for a especific history word
        self.obsBi = np.sum(list(dict(self.counts[history]).values()))        
        
        # If it is an observed bigram: P_katz_2 = P_gt[r] / obsBi
        try:
            r = self.counts[history][target]
            prob = self.probabilities[r] / self.obsBi

        # If it is an unobserved bigram, then check unigrams: P_katz_2 = alpha * P_mle[target]
        except KeyError:
            
            # alpha = (1 - sum( P_gt[count(history, w)] ) / obsBi) / (1 - sum( P_mle[w] ) / obsUni )
            # w represents each observed word given the especific history
            disc_uni = disc_bi = 0.0
            for key, value in dict(self.counts[history]).items():
                disc_bi += self.probabilities[value] 
                disc_uni += self.unicounts[key]

            alpha = (1 - disc_bi / self.obsBi) / (1 - disc_uni / self.obsUni)

            r = self.unicounts[target] 
            prob = alpha * r / self.obsUni
        
        return prob
    
    def get_ngram_probability(self, history, target):
        
        """        
        Compute the conditional probability of the target token given the history, using 
        Laplace smoothing (add k).
        """
                
        try:
            ngram_tot = np.sum(list(self.counts[history].values())) + (self.vocab_size*self.lam)
            try:
                transition_count = self.counts[history][target] + self.lam
            except KeyError:
                transition_count = self.lam
        except KeyError:
            transition_count = self.lam
            ngram_tot = self.vocab_size*self.lam
            
        return transition_count/ngram_tot
    
    def perplexity(self, test_corpus):
        
        """
        Uses the estimated language model to process a corpus and computes the perplexity 
        of the language model over the corpus.
        """
        
        probs = []
        for sentence in test_corpus.sentences:
            for idx in range(self.ngram_size-1, len(sentence)):
                ngram = self.get_ngram(sentence, idx)
                if self.ngram_size == 1:
                    probs.append(self.get_unigram_probability(ngram))
                else:
                    probs.append(self.get_bigram_probability(ngram[0], ngram[1]))
        
        entropy = np.log2(probs)
        # this assertion makes sure that you retrieved valid probabilities, whose log must be <= 0
        assert all(entropy <= 0)
        
        avg_entropy = -1 * (sum(entropy) / len(entropy))
        
        return pow(2.0, avg_entropy)

class SGT(object): 
    
    """
    Calculates the Simple Good-Turing (SGT) discount method of a given ngram model.
    
    - ngram_size: int, the size of the ngrams 
    - counts: dictionary, containing the ngram frequency counts
    - k: int, Katz constant (which is suggested to be = 5)
    """
    
    def __init__(self, model):
        self.ngram_size = model.ngram_size
        self.counts = model.counts
        self.k = 5
        
        r, n, p0 = self.nCounts()
        
        z = self.Z_transform(r, n)            
        
        self.a, self.b = self.linear_regression(r, z)
        
        rStar = np.array(r, dtype=float)
        for i in range(0, len(r)):
            rStar[i] = self.GT_discount(r, i) if (i < self.k) else r[i] 
            
        self.prob = self.find_prob(p0, r, n, rStar)
    
    def nCounts(self):
        
        """
        Creates a counts of counts dictionary that maps the number of occurances
        of a given ngram frequency in the corpus.
        
        N[r] = sum(all ngram that occurs r times)
        
        returns:
        - r_times: array, containing all the possible frequencies found in the ngram model
        - n_ngram: array, containing the number of ngrams occurring r times
        - p0: int, initial probability of unobserved items
        """
        
        # Create a Nc dictionary based on the possible counts values
        nc = dict()
        for value in self.counts.values():
            for r in dict(value).values():
                try:
                    nc[r] += 1
                except KeyError:
                    nc[r] = 1

        sorted_nc = dict(sorted(nc.items()))
        r_times = np.array(list(sorted_nc.keys()), dtype=int)
        n_ngrams = np.array(list(sorted_nc.values()), dtype=int)
        p0 = n_ngrams[0] / np.inner(r_times, n_ngrams)
                            
        return r_times, n_ngrams, p0

    def Z_transform(self, r, n):
        
        """
        To handle the case that most of the Nr are zero for large r, which is a 
        necessary value when calculating r∗. Instead of depending on the raw count Nr
        of each non-zero frequency, these non-zero values are averaged and replaced by:
            Zr = Nr / 0.5 * (t−q)
        where t and q are, respectively, the successor and the predecessor
        of a given r value that were observed in the ngram model.
        for the first r index, q = 0 and for the last index, t = 2*r - q.
        
        returns:
        - Z: array, containing the estimated Z values given count of counts arrays (r, n)
        """
        
        Z = dict()

        # First:  Zr = Nr/0.5*(t-q) with t = 2 and q = 0
        Z[r[0]] = 2 * n[0] / r[1]
        
        # Last: Zr = Nr/(r-q) with t = (2*r - q) and q = r-1 
        Z[r[-1]] = n[-1] / (r[-1] - r[-2]) 

        # General case: Nr/0.5*(t-q)
        for idx in range(1, len(r) - 1):
            Z[r[idx]] = 2 * n[idx] / (r[idx+1] - r[idx-1])
        
        Z = dict(sorted(Z.items()))
        
        return np.array(list(Z.values()), dtype=float)

    def linear_regression(self, r, Z):
        
        """
        Linear regression method to find the appropriated parameters (a, b)
        for the equation:
            log(Zr) = a + b * log(r)
        """
        
        # Working with logarithm values due to the risk of underflow
        l = len(r)
        logr = np.log(r)
        logZ = np.log(Z)
            
        # Find the mean values for log(z):y and log(r):x
        meanX = np.sum(logr)/l
        meanY = np.sum(logZ)/l
        
        # Iterate through the arrays to find the values' deviation from the mean
        xy = x2 = 0
        for i in range(0, l):
            xy += (logr[i] - meanX) * (logZ[i] - meanY)
            x2 += (logr[i] - meanX)**2
    
        # Find the logistic regression parameters
        b = xy/x2
        a = meanY - b*meanX
                
        return a, b
    
    def Z_smoothed(self, i):
        
        """
        Aply Zr[i] = exp(a + b*log(r[i])) to find the smoothed values for Nr(now Zr)
        """
        
        return np.exp(self.a + self.b * np.log(i))
    
    def GT_discount(self, r, i):
        
        """
        Calculate new ngram frequency count (r*) estimate for lower values (< k) of r
        so these values can lose some probability mass that went to the unobserved ngrams.        
        """
        
        # r* = ((r+1) * Zr+1/Zr - r * (k+1) * Zk+1/Z0) / (1 - (k+1) * Zk+1/Z0)
        
        rStar = (((r[i] + 1) * self.Z_smoothed(r[i]+1) / self.Z_smoothed(r[i])) - 
                 (r[i] * (r[self.k] + 1) * self.Z_smoothed(r[self.k]+1) / self.Z_smoothed(r[0])))
        
        rStar /= 1 - (r[self.k] + 1) * self.Z_smoothed(r[self.k]+1) / self.Z_smoothed(r[0])
        
        return rStar

    def find_prob(self, p0, r, n, rStar):
        
        """
        Find SGT discount probabilities and already subtract them from the
        original r count to optimize the Katz Backoff computation
        
        p = (1 - p0) * (rStar * n) / total
        
        (rStar * n) = counts of an especific r value
        total = sum of all counts in the model
        
        returns:
        - p: dictionary, containing the original r frequencies subtracted by the SGT discount probability
        """
        
        p = Counter()

        total = np.inner(n, rStar)
        for idx in range(0, len(n)):
            p[r[idx]] = (1 - p0) * rStar[idx] * n[idx] / total
            p[r[idx]] = r[idx] - p[r[idx]]
                
        return p

class Fold(object):
    
    """
    Creates a fold from the original corpus
    """
    
    def freq_distr(self):    
        return Counter([word for sentence in self.sentences for word in sentence])
    
    def filter_words(self):
        filtered_sentences = []
        for sentence in self.sentences:
            filtered_sentence = []
            for word in sentence:
                # check if the word occurs in the vocabulary
                filtered_sentence.append(word if word in self.vocab else 'UNK')
                
            if len(filtered_sentence) > 1:
                # make sure that the sentence contains more than 1 token
                filtered_sentences.append(filtered_sentence)
    
        return filtered_sentences

def split_corpus(corpus, n):
    
    """
    Split the corpus into n folds to separate the training from the test set using CV 
    """
    
    step = round(len(corpus.sentences)/n)
    
    nfolds = []
    for i in range(0, n):
        new = Fold()
        new.sentences = corpus.sentences[i*step : (i+1)*step - 1]
        new.ngram_size = corpus.ngram_size
        new.frequencies = new.freq_distr()
        nfolds.append(new)
        
    return nfolds

# example code to run a bigram model with Katz backoff algorithm and Good-Turing discounting.

uni_corpus = Corpus(train_path, 1, 10, bos_eos=True, vocab=None)
bi_corpus = Corpus(train_path, 2, 10, bos_eos=True, vocab=None)

uni_model = LM(1)
bi_model = LM(2)

uni_model.update_counts(uni_corpus)
bi_model.update_counts(bi_corpus)

# Calculate Simple Goood-Turing discount
bi_sgt = SGT(bi_model)

# Incorporate SGT probabilities in the bigram model
bi_model.probabilities = bi_sgt.prob

# N-gram preparation: calculate previously the total observed unigrams and attach unigram counts to the bigram model
bi_model.obsUni = np.sum(list(uni_model.counts.values()))
bi_model.unicounts = uni_model.counts

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, 2, 10, bos_eos=True, vocab=bi_model.vocab)
print(bi_model.perplexity(test_corpus))


# Best case scenario: use the train_corpus as test_corpus to analise max likelihood
"""
n = 2
corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model = LM(n, lam=0.001)
bigram_model.update_counts(corpus)

bigram_model.perplexity(corpus)
"""

# example code to run a unigram model with add 0.001 smoothing. Tokens with a frequency count lower than 10
# are replaced with the UNK string
"""
n = 1
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
unigram_model = LM(n, lam=0.001)
unigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=unigram_model.vocab)
unigram_model.perplexity(test_corpus)
"""


# In[ ]:


"""
# example code to run a bigram model with add 0.001 smoothing. The same frequency threshold is applied.
n = 2
train_corpus = Corpus(train_path, 10, n, bos_eos=True, vocab=None)
bigram_model = LM(n, lam=0.001)
bigram_model.update_counts(train_corpus)

# to ensure consistency, the test corpus is filtered using the vocabulary of the trained language model
test_corpus = Corpus(test_path, None, n, bos_eos=True, vocab=bigram_model.vocab)
bigram_model.perplexity(test_corpus)
"""

# example code to run a bigram model with add 0.001 smoothing using CV
"""
n = 2
corpus = Corpus(train_path, 10, bos_eos=True, vocab=None)

if corpus.bos_eos:
    corpus.sentences = corpus.add_bos_eos(2)

shuffle(corpus.sentences)
nfolds = split_corpus(corpus, 10)

bigram_model = LM(n, lam=0.001)
for i in range(0, 9):
    bigram_model.update_counts(nfolds[i])
    
# Check vocabulary in the test_corpus
nfolds[9].vocab = bigram_model.vocab
nfolds[9].sentences = nfolds[9].filter_words()

print(bigram_model.perplexity(nfolds[9]))
"""