import numpy as np
#from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

"""Calculate KNN purity score
   function is taken from theislab/cpa/metrics
"""


def knn_purity(data, labels: np.ndarray, n_neighbors=30):
    """Computes KNN Purity for ``data`` given the labels.
        Parameters
        ----------
        data:
            Numpy ndarray of data
        labels
            Numpy ndarray of labels
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.
    """
    n_samp = data.shape[0]
    if n_neighbors > n_samp:
        if n_samp <= 11:
            n_neighbors = n_samp
        else:
            n_neighbors = n_samp - 10
    labels = LabelEncoder().fit_transform(labels.ravel())

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]
    neighbors_labels = np.vectorize(lambda i: labels[i])(indices)

    # per cell purity scores
    scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
    res = [
        np.mean(scores[labels == i]) for i in np.unique(labels)
    ]  # per label purity

    return np.mean(res)