# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#                   Multi-Label SMOTE                           #
#   Adapted from: https://doi.org/10.1016/j.knosys.2015.07.019  #
#                                                               #
#              Implementation Author: Joel Oliveira             #
#                Email: fc59942 @ alunos.fc.ul.pt               #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
import pandas as pd
from scipy.stats import mode
from numpy.random import Generator, MT19937, SeedSequence
from sklearn.neighbors import NearestNeighbors

from ..shared import BaseResampler, IRLbl, mean_IRLbl

class MLSMOTE(BaseResampler):
    """
    Implementation of the Multi-Label Synthetic Minority Over-sampling Technique (ML-SMOTE) algorithm with compliance to the sklearn Framework.
    
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
        NearestNeighbors object to find the nearest neighbors.
    nn_mapper_: dict
        Dictionary to map the index of the samples to the index of the samples in the dataset.
    
    Note:
    ---------
    This implementation suffered changes in the amount of samples to be generated. In the original paper there isn't a parameter such as 
    'how much samples to generate'. The generation step is performed until a certain condition is met. In this implementation, the user
    specifies the percentage of samples to generate, which is a more practical approach for the user.
    Another change was made, due to performance. The bag of samples from which to sample is iteratively updated, removing the samples
    that are no longer minority samples. In this implementation, to make the most of the numpy and pandas libraries, the samples are
    generated in a single step, which may lead to a performance hit in large datasets. 
    """
    def __init__(self, percentage=0.5, n_neighbors=5,random_state=None):
        self.n_neighbors = n_neighbors
        self.percentage = percentage
        self.random_state = random_state

    def _generate_samples(self, dtypes, samples_X, samples_y, refs_X, nn_X, nn_y):
        """
        Sample generation step of the ML-SMOTE algorithm. The samples are generated based on the nearest neighbors of the minority samples.

        Parameters:
        -----------
        dtypes: pd.Series
            Data types of the features.
        samples_X: np.ndarray
            Features of the minority samples.
        samples_y: np.ndarray
            Multi-labels of the minority samples.
        refs_X: np.ndarray  
            Features of the reference sample for each sample in 'samples_X'.
        nn_X: np.ndarray
            Features of the nearest neighbors of the minority samples.
        nn_y: np.ndarray
            Multi-labels of the nearest neighbors of the minority samples.
        """
        num_samples = samples_X.shape[0]
        new_X = samples_X.copy()
        
        is_category = (dtypes == 'category').to_numpy()
        not_category = ~is_category

        # Handle categorical features
        if np.any(is_category):
            mode_vals = mode(nn_X[:, :, is_category], axis=1).mode
            new_X[:, is_category] = mode_vals.reshape(num_samples, -1)
        
        # Handle continuous features
        if np.any(not_category):
            diff = refs_X[:, not_category] - samples_X[:, not_category]
            offsets = diff * self.random_generator_.random((num_samples, diff.shape[1]))
            new_X[:, not_category] = samples_X[:, not_category] + offsets
        
        # Create new labels
        new_y = ((samples_y + nn_y.sum(axis=1)) > (self.n_neighbors / 2)).astype(int)
        
        return new_X, new_y
    
    def _fit_resample(self, X, y):
        """
        Implementation of the ML-SMOTE algorithm to generate samples for the minority labelsets.

        Parameters:
        -----------
        X: pd.DataFrame
            Features of the dataset.
        y: pd.DataFrame 

        Returns:
        --------   
        X: pd.DataFrame
            New features with generated samples.
        y: pd.DataFrame
            New multi-labels with generated samples.
        """
        self.random_generator_ = Generator(MT19937(SeedSequence(self.random_state)))
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        self.nn_mapper_ = {v:k for k,v in enumerate(X.index)}

        n_samples = int(self.percentage * len(X))
        meanIR = mean_IRLbl(y)
        neighbors = self.nn_.kneighbors(
            n_neighbors=self.n_neighbors, 
            return_distance=False
        )

        # create minority bags with index reference for minority instances
        min_bag = set()
        for label in y.columns:
            irlbl_scores = IRLbl(y)
            if irlbl_scores[label] > meanIR:
                min_label_value = y[label].unique()[y[label].value_counts().argmin()]
                min_bag |= set(X[y[label]==min_label_value].index)

        # map minority bag index to array position
        min_bag = np.array([self.nn_mapper_[idx] for idx in min_bag])
        min_bag_neighbours = neighbors[min_bag]
        
        # create weights based on inverse frequency of labelsets
        counts = y.value_counts()
        weights = np.array(
            [ counts.loc[
                tuple(labelset)
            ] for labelset in y.iloc[min_bag].values]
        )
        weights = 1/(1 + weights)
        weights /= weights.sum()
        
        # select random seeds for generation
        bag_indices = self.random_generator_.choice(range(len(min_bag)), n_samples, p=weights)
        actual_indices = min_bag[bag_indices]
        # select random reference neighbors
        neighbor_ref = self.random_generator_.choice(range(self.n_neighbors), n_samples)
        neighbor_indices = min_bag_neighbours[bag_indices, neighbor_ref]
        
        # generate samples
        samples_X = X.values[actual_indices]
        samples_y = y.values[actual_indices]
        refs_X = X.values[neighbor_indices]
        nn_X = X.values[min_bag_neighbours[bag_indices]]
        nn_y = y.values[min_bag_neighbours[bag_indices]]
        
        new_X_samples, new_y_samples = self._generate_samples(X.dtypes, samples_X, samples_y, refs_X, nn_X, nn_y)
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
        return X,y

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)