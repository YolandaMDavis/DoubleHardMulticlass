"""
Code adapted from https://github.com/uvavision/Double-Hard-Debias/blob/master/eval.py
"""

import os
import numpy as np


def evaluate_analogies(word2vec_dict):
    # Get all word embeddings into a matrix
    vectors = []
    for word in word2vec_dict:
        vectors.append(word2vec_dict[word])
    wv = np.array(vectors)

    # Normalize embeddings
    W = np.zeros(wv.shape)
    d = (np.sum(wv ** 2, 1) ** (0.5))
    W = (wv.T / d).T

    # Create word to index dictionary
    vocab = {word: i for i, word in enumerate(word2vec_dict)}

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
    ]
    prefix = os.path.join('data', 'analogies')

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0;  # count correct semantic questions
    correct_syn = 0;  # count correct syntactic questions
    correct_tot = 0  # count correct questions
    count_sem = 0;  # count all semantic questions
    count_syn = 0;  # count all syntactic questions
    count_tot = 0  # count all questions
    full_count = 0  # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        if data:
            indices = np.array([[vocab[word] for word in row] for row in data])
            ind1, ind2, ind3, ind4 = indices.T

            predictions = np.zeros((len(indices),))
            num_iter = int(np.ceil(len(indices) / float(split_size)))
            for j in range(num_iter):
                subset = np.arange(j * split_size, min((j + 1) * split_size, len(ind1)))

                pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                            + W[ind3[subset], :])
                # cosine similarity if input W has been normalized
                dist = np.dot(W, pred_vec.T)

                for k in range(len(subset)):
                    dist[ind1[subset[k]], k] = -np.Inf
                    dist[ind2[subset[k]], k] = -np.Inf
                    dist[ind3[subset[k]], k] = -np.Inf

                # predicted word index
                predictions[subset] = np.argmax(dist, 0).flatten()

            val = (ind4 == predictions)  # correct predictions
            count_tot = count_tot + len(ind1)
            correct_tot = correct_tot + sum(val)
            if i < 5:
                count_sem = count_sem + len(ind1)
                correct_sem = correct_sem + sum(val)
            else:
                count_syn = count_syn + len(ind1)
                correct_syn = correct_syn + sum(val)

            print("%s:" % filenames[i])
            print('ACCURACY TOP1: %.2f%% (%d/%d)' %
                  (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
          (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
          (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
          (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
