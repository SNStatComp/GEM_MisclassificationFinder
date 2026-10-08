"""
Created on Wed Feb 20 16:29:20 2019

@author: HKMN

the following self-written functions are in here:
    - bm25
    - bm25_transform
"""
import numpy as np
import math
from six import iteritems
#from scipy import sparse
from scipy.sparse import dok_matrix

########################################################################################

def bm25(corpus, min_df = 1, vocab = None):
    ''' Function to do okapi bm25 weighting for words in a corpus.
     Input: corpus as list/pandas series with already tokenized words.
     Optional: Minimum number of documents a word should appear in,
     set of word that builds vocabulary
     Output: sparse matrix (csr) with documents in the rows and words in the columns,
     dict with vocabulary of corpus and index for columns
    '''

    PARAM_K1 = 1.5
    PARAM_B = 0.75
    EPSILON = 0.25

    corpus_size = len(corpus)
    avgdl = 0
    doc_freqs = []
    idf = {}
    doc_len = []
    # Compute document frequency first so that filtering out low frequency words is easier
    nd = {}  # word -> number of documents with word
    for document in corpus:
        for word in set(document):
            if word not in nd:
                nd[word] = 0
            nd[word] += 1
    # Filter out words with document frequency lower than min_df and words that don't appear in vocab
    if vocab is not None:
        nd = {word:freq for word, freq in nd.items() if (freq >= min_df) & (word in vocab)}
    if (min_df>1) & (vocab is None):
        nd = {word:freq for word, freq in nd.items() if freq >= min_df}
    # nd now containes all words that are going to appear in the end result and nothing more!

    num_doc = 0
    for document in corpus:
        len_doc = 0 # For every document, I need to count the number of tokens in it (but only those tokens, that belong to my total vocabulary)
        frequencies = {}
        for word in document:
            if word in nd.keys():  # Test whether words are even supposed to be included
                len_doc += 1
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
        doc_freqs.append(frequencies)
        doc_len.append(len_doc)
        num_doc += len_doc

    # create dict with index values for words
    counter = 0
    index_dict = {}
    for word in nd:
        index_dict[word]=counter
        counter+=1

    avgdl = float(num_doc) / corpus_size  # average length of documents

    idf_sum = 0
    # collect words with negative idf to set them a special epsilon value.
    # idf can be negative if word is contained in more than half of documents

    for word, freq in iteritems(nd):
        in_df = math.log(corpus_size - freq + 0.5) - math.log(freq + 0.5)
        idf[word] = in_df
        idf_sum += in_df
    average_idf = float(idf_sum) / len(idf)

    eps = EPSILON * average_idf
    for word in idf:
        if idf[word] < eps:
            idf[word] = eps


    # Get a weight for every word in every document
    index = 0
    bm25 = {}
    for document in doc_freqs:
        weighted = {}
        for word in document:
            weight_word = (idf[word] * document[word] * (PARAM_K1 + 1)) / (document[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * doc_len[index] / avgdl))
            weighted[index_dict[word]] = weight_word

        bm25[index] = weighted
        index += 1

    # sparse_bm25 = sparse.dok_matrix((corpus_size, len(index_dict)), dtype = np.float64)
    
    sparse_bm25 = dok_matrix((corpus_size, len(index_dict)), dtype = np.float64)
    for doc_id, feats in bm25.items():
        for word in feats:
            sparse_bm25[doc_id, word] = feats[word]
    sparse_bm25 = sparse_bm25.tocsr()

    return sparse_bm25, index_dict, idf, avgdl

########################################################################################

def bm25_transform(corpus, vocab, idf, avgdl):
    ''' Transform test set the same way as training set
    Input: List/pandas series of test data, dictionary of vocabulary with column numbers as values,
    dict of inverse document frequency for every column/word, average document length in training set
    Output: sparse weighted term-document matrix with documents in rows and terms in columns
    '''
    PARAM_K1 = 1.5
    PARAM_B = 0.75

    doc_freqs = []
    doc_len = []
    num_doc = 0
    for document in corpus:
        len_doc = 0 # For every document, I need to count the number of tokens in it (but only those tokens, that belong to my total vocabulary)
        frequencies = {}
        for word in document:
            if word in vocab.keys():  # Test whether words are even supposed to be included
                len_doc += 1
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
        doc_freqs.append(frequencies)
        doc_len.append(len_doc)
        num_doc += len_doc

    # Get a weight for every word in every document

    index = 0
    bm25 = {}
    for document in doc_freqs:
        weighted = {}
        for word in document:
            weight_word = (idf[word] * document[word] * (PARAM_K1 + 1)) / (document[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * doc_len[index] / avgdl))
            weighted[vocab[word]] = weight_word

        bm25[index] = weighted
        index += 1

    # sparse_bm25 = sparse.dok_matrix((len(corpus), len(vocab)), dtype = np.float64)
    sparse_bm25 = dok_matrix((len(corpus), len(vocab)), dtype = np.float64)
    for doc_id, feats in bm25.items():
        for word in feats:
            sparse_bm25[doc_id, word] = feats[word]
    sparse_bm25 = sparse_bm25.tocsr()

    return sparse_bm25
