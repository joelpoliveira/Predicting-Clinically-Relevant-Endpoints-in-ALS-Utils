# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#                   Multi-Label TomekLinks                      #
#   Adapted from: https://doi.org/10.1016/j.neucom.2019.11.076  #
#                                                               #
#              Implementation Author: Joel Oliveira             #
#                Email: fc59942 @ alunos.fc.ul.pt               #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..shared import BaseResampler, IRLbl, mean_IRLbl


class MLTL(BaseResampler):
    """
    Implementation of the Multi-Label Tomek Links (MLTL) algorithm with compliance to the imbalanced-learn API.

    Parameters
    ----------
    threshold: float, default=0.3
        Threshold to determine if two samples are Tomek Links.

    Attributes
    ----------
    n_labels_: int
        Number of labels in the dataset.
    nn_: NearestNeighbors  
        nearest neighbors for each data sample in the input dataset.
    mapper_: dict
        Dictionary to map the index values of the dataset to their respective position in the index array.
    """
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def _hamming_dist(self, y1, y2):
        return np.sum(y1 != y2, axis=1) / self.n_labels_

    def _undersample(self, X, y):
        """
        Collects indices that are Tomek Links. This is, nearest samples with an average hamming distance 
        to it's neighbors greater than the pre-defined threshold.

        Parameters
        ----------
        X: pd.DataFrame
            Features of the dataset.
        y: pd.DataFrame
            Labels of the dataset.

        Returns
        -------
        tl: list
            List of indices of samples that are Tomek Links.
        """
        neighbors = self.nn_.kneighbors(return_distance=False)
        meanIR = mean_IRLbl(y)

        maj_bag_indices = []
        for label, ir in IRLbl(y).items():
            if ir < meanIR:
                maj_bag_indices.extend(X[y[label] == 0].index.tolist())

        maj_bag_indices = np.array(maj_bag_indices)
        maj_bag_indices = np.unique(maj_bag_indices)  # Ensure unique indices
        
        i_indices = np.array([self.mapper_[idx] for idx in maj_bag_indices])
        nearest_neighbors = neighbors[i_indices, 0]

        y_i = y.iloc[i_indices].to_numpy()
        y_nearest = y.iloc[nearest_neighbors].to_numpy()

        hamming_distances = self._hamming_dist(y_i, y_nearest)

        tl = maj_bag_indices[hamming_distances >= self.threshold]

        return tl.tolist()

    def _fit_resample(self, X, y):
        """
        Implementation of the MLTL algorithm.

        Parameters
        ----------
        X: pd.DataFrame
            Features of the dataset.
        y: pd.DataFrame
            Labels of the dataset.

        Returns
        -------
        X_resampled: pd.DataFrame
            Features of the resampled dataset.
        y_resampled: pd.DataFrame
            Labels of the resampled dataset.
        """
        self.n_labels_ = y.shape[1]
        self.nn_ = NearestNeighbors().fit(X)
        self.mapper_ = {idx: i for i, idx in enumerate(X.index)}

        tl = self._undersample(X, y)
        X = X.drop(index=tl)
        y = y.drop(index=tl)
        return X.reset_index(drop=True), y.reset_index(drop=True)

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)
