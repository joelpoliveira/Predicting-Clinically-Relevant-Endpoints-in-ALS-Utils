# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#               Label Power-set Random Oversampling             #
#   Adapted from: https://doi.org/10.1016/j.neucom.2014.08.091  #
#                                                               #
#              Implementation Author: Joel Oliveira             #
#                Email: fc59942 @ alunos.fc.ul.pt               #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
import pandas as pd
from numpy.random import Generator, MT19937, SeedSequence

from ..shared import BaseResampler

class LPROS(BaseResampler):
    """
    Implementation of the Label-Powerset Random Oversampling (LPROS) algorithm with compliance to the imblearn API.

    Parameters:
    -----------
    percentage: float, default=0.5
        Percentage of samples to generate.
    random_state: int, default=None
        Random seed for reproducibility.

    Attributes:
    -----------
    random_generator_: numpy.random.Generator
        Random generator for reproducibility.
    """
    def __init__(self, percentage=0.5, random_state=None):
        self.percentage = percentage
        self.random_state = random_state

    def _distribute(self, current_bag, bags, remainder):
        """
        Function in which the remainder of samples to generate is distributed among the remaining bags.
        {current_bag} is removed from the set of {bags}, the remainder of samples to be generated are
        distributed among the remaining bags according to their proportion in the dataset and the bags
        remaining are returned.

        Parameters:
        -----------
        current_bag: pd.Series
            Bag for which samples were generated in the respective iteration
        bags: dict
            Dictionary with the remaining bags (including current_bag) and their counts.
        remainder: int
            Number of samples to distribute among the remaining bags.

        Returns:
        --------
        bags: dict
            Dictionary with the remaining bags and their counts after the distribution of samples.
        """
        bag_without_current = dict(
            filter(lambda x: x[0] != current_bag, bags.items())
        )

        label_counts = np.array(list(bag_without_current.values()))
        distributions =  label_counts / label_counts.sum() 

        for i, bag in enumerate(bag_without_current):
            n_to_add = int(distributions[i] * remainder)
            bag_without_current[bag] += n_to_add

        return bag_without_current

    def _fit_resample(self, X, y):
        """
        Implementation of the LPROS algorithm to generate samples for the minority labelsets.

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
        # number of samples to generate
        n_samples = int(self.percentage * len(X))
        # bag of labelset counts
        label_bags = y.value_counts().to_dict()
        # average lebal-set frequency
        mean_size =  sum(label_bags.values()) / len(label_bags)
        # bag with minority sized labelsets
        min_bags = dict(filter(lambda x: x[1] < mean_size, label_bags.items()))
        # average number of samples to generate per bag
        mean_red = n_samples / len(min_bags)
        # sort min bags to generate samples for the bigger bag first
        min_bags = dict(
            sorted( 
                min_bags.items(), 
                key=lambda x: x[1],
                reverse=True
            )
        )

        for bag in min_bags:
            # number of samples to generate for the current bag
            n_to_add = int(min(mean_size - min_bags[bag], mean_red))
            # remainder of samples to generate, to be distributed for the remaining bags
            remainder = mean_red - n_to_add
            
            bag_idx = X[(y==bag).all(axis=1)].index
            idx_to_add = self.random_generator_.choice(bag_idx, n_to_add, replace=True)
            
            X = pd.concat((X, X.loc[idx_to_add]))
            y = pd.concat((y, y.loc[idx_to_add]))

            min_bags = self._distribute(bag, min_bags, remainder)

        return X, y
    
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)