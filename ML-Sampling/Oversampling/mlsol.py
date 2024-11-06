# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#          Multi-label Sampling Based on Local Imbalance        #
#   Adapted from: https://doi.org/10.1007/978-3-030-46147-8_11  #
#                                                               #
#              Implementation Author: Joel Oliveira             #
#                Email: fc59942 @ alunos.fc.ul.pt               #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from numpy.random import Generator, MT19937, SeedSequence

from ..shared import BaseResampler

class MLSOL(BaseResampler):
    """
    Implementation of the Multi-Label Synthetic Over-sampling based on Local Label Distribution (ML-SOL) algorithm with compliance to the sklearn Framework.

    Parameters:
    -----------
    percentage: float, default=0.5
        Percentage of samples to generate.
    n_neighbors: int, default=5
        Number of neighbors to consider for the generation of samples.
    random_state: int, default=None
        Random seed for reproducibility.

    Attributes:
    -----------
    random_generator_: numpy.random.Generator
        Random generator for reproducibility.
    nn_: sklearn.neighbors.NearestNeighbors
        NearestNeighbors object to find the nearest neighbors of each instance.
    nn_mapper_: dict
        Dictionary to map the index of the samples to the index of the samples in the dataset.
    """
    def __init__(self, percentage=0.5, n_neighbors=5, random_state=None):
        self.percentage = percentage
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _get_C_matrix(self, neighbors, y):
        neighbor_labels = y.values[neighbors]
        y_matrix = y.values[:, np.newaxis, :]
        differences = (y_matrix != neighbor_labels).sum(axis=1) / self.n_neighbors
        return differences

    def _get_w_vector(self, y, C):
        min_classes = np.array([
            y[l].value_counts().index[1] for l in y
        ])
        mismatch_counts = C*(y.values == min_classes)*(C<1)
        denominator = (C * (y.values == min_classes) * (C < 1)).sum(axis=0)
        w = (mismatch_counts / denominator).sum(axis=1)
        return w

    def _init_types(self, neighbors, y, C):
        T = np.zeros_like(y.values)
        T[(C < 0.3) & (y.values != 0)] = 1
        T[(C >= 0.3) & (C < 0.7) & (y.values != 0)] = 2
        T[(C >= 0.7) & (C < 1) & (y.values != 0)] = 3
        T[(C >= 1) & (y.values != 0)] = 4

        has_changed = True
        while has_changed:
            has_changed = False
            mask = (T == 3)
            for i, row in enumerate(mask):
                if np.any(row):
                    current_neighbors = neighbors[i, 1:]
                    neighbor_types = T[current_neighbors]
                    change_mask = (neighbor_types == 1) | (neighbor_types == 2)
                    if np.any(change_mask):
                        T[i][row] = 2
                        has_changed = True
        return T

    def _generate_samples(self, dtypes, xs_samples, ys_samples, Ts, xr_samples, yr_samples, Tr):
        num_samples = xs_samples.shape[0]
        num_features = xs_samples.shape[1]

        new_x = xs_samples.copy()
        
        is_category = (dtypes == "category").to_numpy()
        not_category = ~is_category

        if np.any(is_category):
            probs = self.random_generator_.random(num_samples)
            filter_cols = np.repeat(is_category[np.newaxis, :], num_samples, axis=0)

            filter_rows = np.repeat((probs<0.5)[:, np.newaxis], num_features, axis=1)
            new_x[filter_rows&filter_cols] = xs_samples[filter_rows&filter_cols]
            
            filter_rows = np.repeat((probs>=0.5)[:, np.newaxis], num_features, axis=1) 
            new_x[filter_rows&filter_cols] = xr_samples[filter_rows&filter_cols]

        if np.any(not_category):
            weights = self.random_generator_.random((num_samples, 1))
            new_x[:, not_category] = xs_samples[:, not_category] + weights * (xr_samples[:, not_category] - xs_samples[:, not_category])

        ds = np.linalg.norm(new_x - xs_samples, axis=1)
        dr = np.linalg.norm(new_x - xr_samples, axis=1)
        
        cd = ds
        cd[(ds-dr)==0] = 0 
        cd=np.divide(cd, (ds - dr), where=(ds-dr)!=0)
        #cd[np.isnan(cd)]=0

        new_y = ys_samples.copy()
        for i in range(new_y.shape[0]):
            for j in range(new_y.shape[1]):
                if ys_samples[i,j] != yr_samples[i,j]:
                    if Ts[i, j] == 0:
                        xs_samples[i], xr_samples[i] = xr_samples[i], xs_samples[i]
                        ys_samples[i], yr_samples[i] = yr_samples[i], ys_samples[i]
                        Ts[i], Tr[i] = Tr[i], Ts[i]
                        cd[i] = 1 - cd[i]
                    
                    theta = 0
                    if Ts[i,j] == 1:
                        theta = 0.5
                    elif Ts[i,j] == 2:
                        theta = 0.75
                    elif Ts[i,j] == 3:
                        theta = 1 + 1e-5
                    elif Ts[i,j] == 4:
                        theta = -1e-5

                    new_y[i,j] = ys_samples[i,j] if cd[i] <= theta else yr_samples[i,j]
        return new_x, new_y

    def _fit_resample(self, X, y):
        """
        Implementation of the ML-SOL algorithm to generate samples for the minority labelsets.

        Parameters:
        -----------
        X: pd.DataFrame
            Features of the dataset.
        y: pd.DataFrame
            Multi-labels of the dataset.

        Returns:
        --------
        X: pd.DataFrame
            New features with generated samples.
        y: pd.DataFrame
            New multi-labels with generated samples.
        """
        self.random_generator_ = Generator(MT19937(SeedSequence(self.random_state)))
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)

        n_samples = int(self.percentage * len(X))
        neighbors = self.nn_.kneighbors(n_neighbors=self.n_neighbors, return_distance=False)

        C = self._get_C_matrix(neighbors, y)
        w = self._get_w_vector(y, C)
        T = self._init_types(neighbors, y, C)

        seed_inds = self.random_generator_.choice(
            range(X.shape[0]), n_samples, p=w/w.sum()
        )
        neighbor_ref = self.random_generator_.choice(range(self.n_neighbors), n_samples)
        neighbor_indices = neighbors[seed_inds, neighbor_ref]
        
        new_X_samples, new_y_samples = self._generate_samples(
            X.dtypes,
            X.values[seed_inds], y.values[seed_inds], T[seed_inds],
            X.values[neighbor_indices], y.values[neighbor_indices], T[neighbor_indices]
        )
        
        new_X_samples = pd.DataFrame(
            new_X_samples,
            columns = X.columns
        ).astype(X.dtypes)
        new_y_samples = pd.DataFrame(
            new_y_samples,
            columns = y.columns
        ).astype(y.dtypes)

        X = pd.concat((X, new_X_samples)).reset_index(drop=True).astype(X.dtypes)
        y = pd.concat((y, new_y_samples)).reset_index(drop=True)

        return X, y

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)
