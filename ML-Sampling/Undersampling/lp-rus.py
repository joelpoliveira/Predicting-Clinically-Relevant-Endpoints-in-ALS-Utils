# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                               #
#               Label Power-set Random Undersampling            #
#   Adapted from: https://doi.org/10.1016/j.neucom.2014.08.091  #
#                                                               #
#              Implementation Author: Joel Oliveira             #
#                Email: fc59942 @ alunos.fc.ul.pt               #
#                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
from numpy.random import Generator, MT19937, SeedSequence

from ..shared import BaseResampler

class LPRUS(BaseResampler):
    """
    Implementation of the Label Powerset Random Under-Sampling (LPRUS) algorithm with compliance to the imbalanced-learn API.

    Parameters
    ----------
    percentage: float, default=0.5
        Percentage of samples to be removed from the dataset.
    random_state: int, default=None
        Seed for the random number generator.

    Attributes
    ----------
    random_generator_: Generator
        Random number generator used to select samples to be removed.
    """
    def __init__(self, percentage=0.5, random_state=None):
        self.percentage = percentage
        self.random_state = random_state

    def _distribute(self, current_bag, bags, remainder):
        """
        Function in which the remainder of samples to delete is distributed among the remaining bags.
        {current_bag} is removed from the set of {bags}, the remainder of samples to be deleted are
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
        Implementation of the LPRUS algorithm.

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
        self.random_generator_ = Generator(MT19937(SeedSequence(self.random_state)))
        # number of samples to delete
        n_samples = int(self.percentage * len(X))
        # bag of labelset counts
        label_bags = y.value_counts().to_dict()
        # average size of labelset
        mean_size = 1/len(label_bags) * sum(label_bags.values())
        # bag with majority sized labelsets
        maj_bags = dict(filter(lambda x: x[1] > mean_size, label_bags.items()))
        # average number of samples to delete per bag
        mean_red = n_samples / len(maj_bags)
        # sort maj bags to delete samples from the smallest bag first
        maj_bags = dict(
            sorted( 
                maj_bags.items(), 
                key=lambda x: x[1],
            )
        )

        for bag in maj_bags:
            # number of samples to delete from the current bag
            n_to_delete = int(min(maj_bags[bag] - mean_size, mean_red))
            # remainder after deleting samples, to be distributed among the other bags
            remainder = mean_red - n_to_delete

            bag_idx = X[(y==bag).all(axis=1)].index
            idx_to_delete = np.random.choice(bag_idx, n_to_delete, replace=False)
            
            X = X.drop(index=idx_to_delete)
            y = y.drop(index=idx_to_delete)

            maj_bags = self._distribute(bag, maj_bags, remainder)

        return X, y
    
    def fit_resample(self, X, y):
        return self._fit_resample(X, y)