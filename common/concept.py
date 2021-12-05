import glob
from sklearn.utils import Bunch
from sklearn.cluster import AgglomerativeClustering, KMeans
from six import iteritems
import os
import numpy as np

def evaluate_categorization(word_vectors, X, y, method='kmeans', seed=None):
    """
    Evaluate embeddings on categorization task.
    Parameters
    ----------
    w: dict
      Embeddings to test.
    X: vector, shape: (n_samples, )
      Vector of words.
    y: vector, shape: (n_samples, )
      Vector of cluster assignments.
    """

    # Get all word embeddings into a matrix
    vectors = []
    for word in word_vectors:
        vectors.append(word_vectors[word])

    # Mean of all embeddings
    mean_vector = np.mean(vectors, axis=0, keepdims=True)

    w = word_vectors

    new_x = []
    new_y = []
    exist_cnt = 0
    for idx, word in enumerate(X.flatten()):
        if word in w:
            new_x.append(X[idx])
            new_y.append(y[idx])
            exist_cnt += 1

    # Number of words in BLESS that also exists in our vocabulary
    print('exist {} in {}'.format(exist_cnt, len(X)))

    X = np.array(new_x)
    y = np.array(new_y)

    # Put all the words that were in both BLESS and our vocab into a matrix
    words = np.vstack([w.get(word, mean_vector) for word in X.flatten()])
    ids = np.random.RandomState(seed).choice(range(len(X)), len(X), replace=False)

    # Evaluate clustering on several hyperparameters of AgglomerativeClustering and
    # KMeans
    best_purity = 0

    if method == "all" or method == "agglomerative":
        best_purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                       affinity="euclidean",
                                                                       linkage="ward").fit_predict(words[ids]))
        for affinity in ["cosine", "euclidean"]:
            for linkage in ["average", "complete"]:
                purity = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                          affinity=affinity,
                                                                          linkage=linkage).fit_predict(words[ids]))
                best_purity = max(best_purity, purity)

    if method == "all" or method == "kmeans":
        purity = calculate_purity(y[ids], KMeans(random_state=seed, n_init=10, n_clusters=len(set(y))).
                                  fit_predict(words[ids]))
        best_purity = max(purity, best_purity)

    return best_purity


def calculate_purity(y_true, y_pred, verbose=False):
    """
    Calculate purity for given true and predicted cluster labels.
    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels
    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.
    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)
    true_clusters = np.zeros(
        shape=(len(set(y_true)), len(y_true)))  # creates sparse array (categories, words both in bench and vocab)
    pred_clusters = np.zeros_like(true_clusters)  # creates sparse array (categories, words both in bench and vocab)
    for id, cl in enumerate(set(y_true)):
        if verbose:
            print("true:", id)
        true_clusters[id] = (y_true == cl).astype("int")  # Everwhere the label is of a certain class, put a 1
    for id, cl in enumerate(set(y_pred)):
        if verbose:
            print("pred:", id)
        pred_clusters[id] = (y_pred == cl).astype("int")  # Everwhere the label is in a certain cluster, put a 1

    # For each clust in the prediction, find the true cluster that has the MOST overlap
    # Sum up the number of words that overlap between the pred and true cluster that has the most overlap
    # Divide this by (the number of words both in bench and vocab)
    M = pred_clusters.dot(true_clusters.T)
    return 1. / len(y_true) * np.sum(np.max(M, axis=1))


def evaluate_cate(wv_dict, benchmarks, method='all', seed=None):
    categorization_tasks = benchmarks
    categorization_results = {}

    # Calculate results using helper function
    for name, data in iteritems(categorization_tasks):
        print("Sample data from {}, num of samples: {} : \"{}\" is assigned class {}".format(
            name, len(data.X), data.X[0], data.y[0]))
        categorization_results[name] = evaluate_categorization(wv_dict, data.X, data.y, method=method, seed=None)
        print("Cluster purity on {} {}".format(name, categorization_results[name]))


def create_bunches(bench_paths):
    output = {}
    for path in bench_paths:
        files = glob.glob(os.path.join(path, "*.txt"))
        name = os.path.basename(path)
        if name == 'EN-BATTIG':
            sep = ","
        else:
            sep = " "

        X = []
        y = []
        names = []
        for cluster_id, file_name in enumerate(files):
            with open(file_name) as f:
                lines = f.read().splitlines()[:]
                X += [l.split(sep) for l in lines]
                y += [os.path.basename(file_name).split(".")[0]] * len(lines)
        output[name] = Bunch(X=np.array(X, dtype="object"), y=np.array(y).astype("object"))

        if sep == ",":
            data = output[name]
            output[name] = Bunch(X=data.X[:, 0], y=data.y, freq=data.X[:, 1], frequency=data.X[:, 2], rank=data.X[:, 3],
                                 rfreq=data.X[:, 4])

    return output