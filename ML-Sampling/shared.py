from abc import abstractmethod
from imblearn.base import SamplerMixin, OneToOneFeatureMixin

def IRLbl(y):
    """
    Imbalance Ratio per Label
    
    Parameters:
    -----------
    y: pd.DataFrame
        Multi-label binary matrix
    
    Returns:
    --------
    irlbl: pd.Series
        Imbalance Ratio per Label
    """
    label_sum = y.sum(axis=0)
    irlbl = label_sum.astype(float)
    irlbl[label_sum != 0] = irlbl.max() / irlbl[label_sum != 0]
    return irlbl


def mean_IRLbl(y):
    """
    Mean Imbalance Ratio per Label
    
    Parameters:
    -----------
    y: pd.DataFrame
        Multi-label binary matrix
    
    Returns:
    --------
    mean_irlbl: float
        Mean Imbalance Ratio per Label
    """
    return IRLbl(y).mean(None)


class BaseResampler(SamplerMixin, OneToOneFeatureMixin):
    """
    Base class for custom resamplers compatible with imblearn Pipeline.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def _fit_resample(self, X, y):
        pass

    @abstractmethod
    def fit_resample(self, X, y):
        pass

    def fit(self, X, y):
        return self
    
