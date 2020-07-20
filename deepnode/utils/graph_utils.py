from itertools import combinations
import torch
import numpy as np
from math import log
import scipy.sparse as sp

def make_window(sentences, window_size) :
    # In this case, sentences = train_data[][1]
    windows = []
    for sentence in sentences :
        sentence_length = len(sentence)
        if sentence_length <= window_size :
            windows.append(sentence)
        else :
            for j in range(sentence_length - window_size + 1) :
                window = sentence[j:j+window_size]
                windows.append(window)
    return windows

def count_word(windows, word) :
    count = 0
    for window in windows :
        if word in window :
            count += 1

    return count

def count_word_freq(vocab, windows) :
    word_freq = {}
    for word in vocab :
        if word not in word_freq :
            word_freq[word] = count_word(windows, word)

    return word_freq

def count_pair_freq(windows) :
    pair_freq = dict()
    for i, window in enumerate(windows) :
        combination = list(combinations(window, 2))
        for comb in combination :
            if (comb[0], comb[1]) in pair_freq :
                pair_freq[(comb[0], comb[1])] += 1
            elif (comb[1], comb[0]) in pair_freq :
                pair_freq[(comb[1], comb[0])] += 1
            else :
                pair_freq[(comb[0], comb[1])] = 1
    return pair_freq

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



class Graph_creation(object):
    
    def __init__(self):
        pass
    
    def build_vocab(self,sentences):

        # build vocab
        word_freq = {}
        word_set = set()
        for doc_words in sentences:
            words = doc_words.split()
            for word in words:
                word_set.add(word)
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        vocab = list(word_set)
        vocab_size = len(vocab)
        word2int = {j : i for i,j in enumerate(vocab)}

        return {'vocab' : vocab, 
                'vocab_size' : vocab_size, 
                'word_freq' : word_freq, 
                'word_2_int' : word2int}


    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def word_doc_freq(self,sentences):
        word_doc_list = {}
        for i in range(len(sentences)):
            doc_words = sentences[i]
            words = doc_words.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_doc_list:
                    doc_list = word_doc_list[word]
                    doc_list.append(i) 
                    word_doc_list[word] = doc_list
                else:
                    word_doc_list[word] = [i]
                appeared.add(word)

        word_doc_freq = {}
        for word, doc_list in word_doc_list.items():
            word_doc_freq[word] = len(doc_list)
        return {'word_doc_list' : word_doc_list, 'word_doc_freq' : word_doc_freq}




    def word_windows(self, sentences, window_size):
        # word co-occurence with context windows
        windows = []

        for doc_words in sentences:
            words = doc_words.split()
            length = len(words)
            if length <= window_size:
                windows.append(words)
            else:
                # print(length, length - window_size + 1)
                for j in range(length - window_size + 1):
                    window = words[j: j + window_size]
                    windows.append(window)
                    # print(window)
        return windows


    def word_window_freq(self, windows):

        word_window_freq = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in word_window_freq:
                    word_window_freq[window[i]] += 1
                else:
                    word_window_freq[window[i]] = 1
                appeared.add(window[i])

        return word_window_freq


    # In[ ]:




    def word_pair_count(self, word_window_list, word_id_map):
        word_pair_count = {}
        for window in word_window_list:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]

                    if word_i_id == word_j_id:

                        continue
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id)

                    if word_pair_str in word_pair_count:

                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        return word_pair_count



    def word_pair_count_demo(self, word_window_list, word_id_map):
        word_pair_count = {}
        for window in word_window_list:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]

                    if word_i_id == word_j_id:

                        continue
                    word_pair_str = str(word_i) + ',' + str(word_j)

                    if word_pair_str in word_pair_count:

                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j) + ',' + str(word_i)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1

        return word_pair_count


    def calculate_pmi(self, word_windows, word_pair_count, 
                      word_window_freq, vocab, train_size):
        row = []
        col = []
        weight = []

        # pmi as weights
        num_window = len(word_windows)

        for key in word_pair_count:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])

            count = word_pair_count[key]

            word_freq_i = word_window_freq[vocab[i]]
            word_freq_j = word_window_freq[vocab[j]]


            pmi = log((1.0 * count / num_window) /
                      (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue

            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

        return {'row' : row, 'col' : col, 'weight' : weight }


    def heterogeneous_graph(self, sentences, word_id_map, 
                            train_size, pmi, 
                            vocab, test_size,word_doc_freq):


        vocab_size = len(vocab)
        row = pmi['row']
        col = pmi['col']
        weight = pmi['weight']

        # doc word frequency
        doc_word_freq = {}

        for doc_id in range(len(sentences)):
            doc_words = sentences[doc_id]
            words = doc_words.split()
            for word in words:
                word_id = word_id_map[word]
                doc_word_str = str(doc_id) + ',' + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1

        for i in range(len(sentences)):
            doc_words = sentences[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = doc_word_freq[key]
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size)
                col.append(train_size + j)
                idf = log(1.0 * len(sentences) /
                          word_doc_freq[vocab[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)


        node_size = train_size + vocab_size + test_size
        adj = sp.csr_matrix(
        (weight, (row, col)), shape=(node_size, node_size))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(node_size))

        return adj