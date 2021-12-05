"""
Original Code Adapted from https://github.com/TManzini/DebiasMulticlassWordEmbedding/blob/master/Debiasing/biasOps.py
(Except where directly indicated from another source)
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def normalize(word_vectors):
    for k, v in word_vectors.items():
        word_vectors[k] = v / np.linalg.norm(v)

def calculate_main_pca_components(word_vectors):
    """From https://github.com/uvavision/Double-Hard-Debias/blob/master/GloVe_Debias.ipynb"""
    vectors = word_vectors.vectors
    wv_mean = np.mean(np.array(vectors), axis=0)
    wv_hat = vectors - wv_mean
    main_pca = PCA()
    main_pca.fit(wv_hat)
    return main_pca

def neutralize_and_equalize_with_frequency_removal(vocab, words, eq_sets, bias_subspace, embedding_dim, principal_component):
    """
    Function to support double hard debias technique, ensureing to remove frequence direction before
    executing netrualize and equalize function

    vocab - dictionary mapping words to embeddings
    words - words to neutralize
    eq_sets - set of equality sets
    bias_subspace - subspace of bias from identify_bias_subspace
    embedding_dim - dimensions of the word embeddings
    """

    if bias_subspace.ndim == 1:
        bias_subspace = np.expand_dims(bias_subspace, 0)
    elif bias_subspace.ndim != 2:
        raise ValueError("bias subspace should be either a matrix or vector")

    full_set = set(list(words) + [word for eq_words in eq_sets for word in eq_words])
    freq_vocab = vocab.copy()

    for word in full_set:
        vector = freq_vocab[word]
        projection = np.dot(np.dot(np.transpose(principal_component), vector), principal_component)
        freq_vocab[word] = vector - projection

    return neutralize_and_equalize(freq_vocab, words, eq_sets, bias_subspace, embedding_dim)


def identify_bias_subspace(vocab, def_sets, subspace_dim, embedding_dim):
    """
    Similar to bolukbasi's implementation at
    https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py

    vocab - dictionary mapping words to embeddings
    def_sets - sets of words that represent extremes? of the subspace
            we're interested in (e.g. man-woman, boy-girl, etc. for binary gender)
    subspace_dim - number of vectors defining the subspace
    embedding_dim - dimensions of the word embeddings
    """
    # calculate means of defining sets
    means = {}
    for k, v in def_sets.items():
        wSet = []
        for w in v:
            try:
                wSet.append(vocab[w])
            except KeyError as e:
                pass
        set_vectors = np.array(wSet)
        means[k] = np.mean(set_vectors, axis=0)

    # calculate vectors to perform PCA
    matrix = []
    for k, v in def_sets.items():
        wSet = []
        for w in v:
            try:
                wSet.append(vocab[w])
            except KeyError as e:
                pass
        set_vectors = np.array(wSet)
        diffs = set_vectors - means[k]
        matrix.append(diffs)

    matrix = np.concatenate(matrix)

    pca = PCA(n_components=subspace_dim)
    pca.fit(matrix)

    return pca.components_

def project_onto_subspace(vector, subspace):
    v_b = np.zeros_like(vector)
    for component in subspace:
        v_b += np.dot(vector.transpose(), component) * component
    return v_b

def calculateDirectBias(vocab, neutral_words, bias_subspace, c=1):
    directBiasMeasure = 0
    for word in neutral_words:
        vec = vocab[word]
        directBiasMeasure += np.linalg.norm(cosine_similarity(vec, bias_subspace))**c
    directBiasMeasure *= 1.0/len(neutral_words)
    return directBiasMeasure

def neutralize_and_equalize(vocab, words, eq_sets, bias_subspace, embedding_dim):
    """
    vocab - dictionary mapping words to embeddings
    words - words to neutralize
    eq_sets - set of equality sets
    bias_subspace - subspace of bias from identify_bias_subspace
    embedding_dim - dimensions of the word embeddings
    """

    if bias_subspace.ndim == 1:
        bias_subspace = np.expand_dims(bias_subspace, 0)
    elif bias_subspace.ndim != 2:
        raise ValueError("bias subspace should be either a matrix or vector")

    new_vocab = vocab.copy()
    for w in words:
        # get projection onto bias subspace
        if w in vocab:
            v = vocab[w]
            v_b = project_onto_subspace(v, bias_subspace)

            new_v = (v - v_b) / np.linalg.norm(v - v_b)
            #print np.linalg.norm(new_v)
            # update embedding
            new_vocab[w] = new_v

    normalize(new_vocab)

    for eq_set in eq_sets:
        mean = np.zeros((embedding_dim,))

        #Make sure the elements in the eq sets are valid
        cleanEqSet = []
        for w in eq_set:
            try:
                _ = new_vocab[w]
                cleanEqSet.append(w)
            except KeyError as e:
                pass

        for w in cleanEqSet:
            mean += new_vocab[w]
        mean /= float(len(cleanEqSet))

        mean_b = project_onto_subspace(mean, bias_subspace)
        upsilon = mean - mean_b

        for w in cleanEqSet:
            v = new_vocab[w]
            v_b = project_onto_subspace(v, bias_subspace)

            frac = (v_b - mean_b) / np.linalg.norm(v_b - mean_b)
            new_v = upsilon + np.sqrt(1 - np.sum(np.square(upsilon))) * frac

            new_vocab[w] = new_v

    return new_vocab
