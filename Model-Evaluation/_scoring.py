import warnings
import numpy as np

from functools import partial
from contextlib import suppress
from collections import Counter
from traceback import format_exc
from sklearn.metrics._scorer import (
    _BaseScorer,
    _cached_call
)
from sklearn.utils.multiclass import type_of_target


def _bin_score(estimator, X_test, y_test, scorer, error_score="raise"):
    """
    COPY OF sklearn.model_selection._validation._score to perform labelwise binary evaluation 
      on multi-label data
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _Labelwise_Multilabel_MultimetricScorer(scorers=scorer, raise_exc=(error_score == "raise"))

    try:
        scores = scorer(estimator, X_test, y_test) 
    except Exception:
        if isinstance(scorer, _Labelwise_Multilabel_MultimetricScorer):
            # If `_MultimetricScorer` raises exception, the `error_score`
            # parameter is equal to "raise".
            raise
        else:
            if error_score == "raise":
                raise
            else:
                scores = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )

    # Check non-raised error messages in `_MultimetricScorer`
    if isinstance(scorer, _Labelwise_Multilabel_MultimetricScorer):
        exception_messages = [
            (name, str_e) for name, str_e in scores.items() if isinstance(str_e, str)
        ]
        if exception_messages:
            # error_score != "raise"
            for name, str_e in exception_messages:
                scores[name] = error_score
                warnings.warn(
                    "Scoring failed. The score on this train-test partition for "
                    f"these parameters will be set to {error_score}. Details: \n"
                    f"{str_e}",
                    UserWarning,
                )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, dict):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, dict):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores

class _Labelwise_Multilabel_MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.

    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).

    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.

    raise_exc : bool, default=True
        Whether to raise the exception in `__call__` or not. If set to `False`
        a formatted string of the exception details is passed as result of
        the failing scorer.
    """

    def __init__(self, *, scorers, raise_exc=True):
        self._scorers = scorers
        self._raise_exc = raise_exc

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)

        for name, scorer in self._scorers.items():
            try:
                if isinstance(scorer, _BaseScorer):
                    score = scorer._score(cached_call, estimator, *args, **kwargs)

                else:
                    score = scorer(estimator, *args, **kwargs)
                scores[name] = score
            except Exception as e:
                if self._raise_exc:
                    raise e
                else:
                    scores[name] = format_exc()
        return scores
    
    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial.

        Caching may be beneficial when one of these conditions holds:
          - `_ProbaScorer` will be called twice.
          - `_PredictScorer` will be called twice.
          - `_ThresholdScorer` will be called twice.
          - `_ThresholdScorer` and `_PredictScorer` are called and
             estimator is a regressor.
          - `_ThresholdScorer` and `_ProbaScorer` are called and
             estimator does not have a `decision_function` attribute.

        """
        if len(self._scorers) == 1:  # Only one scorer
            return False

        counter = Counter([type(v) for v in self._scorers.values()])

        if any(
            counter[known_type] > 1
            for known_type in [_Labelwise_Multilabel_PredictScorer, _Labelwise_Multilabel_ThreholdScorer]
        ):
            return True

        #if counter[_Labelwise_Multilabel_ThreholdScorer]:
        #    if is_regressor
        return False
    




class _Labelwise_Multilabel_PredictScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return {
                f"label_{i}": self._sign * self._score_func(
                    y_true.iloc[:, i], y_pred[:, i], 
                    sample_weight=sample_weight, 
                    **self._kwargs
                ) for i in range(y_true.shape[1])
            }
        else:
            return {
                f"label_{i}": self._sign * self._score_func(
                    y_true.iloc[:, i], y_pred[:, i], 
                    **self._kwargs
                ) for i in range(y_true.shape[1])
            }

class _Labelwise_Multilabel_ThreholdScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        try:
            y_pred = method_caller(estimator, "decision_function", X)

            if isinstance(y_pred, list):
                # For multi-output multi-class estimator
                y_pred = np.vstack([p for p in y_pred]).T
            elif y_type == "binary" and "pos_label" in self._kwargs:
                self._check_pos_label(self._kwargs["pos_label"], estimator.classes_)
                if self._kwargs["pos_label"] == estimator.classes_[0]:
                    # The implicit positive class of the binary classifier
                    # does not match `pos_label`: we need to invert the
                    # predictions
                    y_pred *= -1

        except (NotImplementedError, AttributeError):
            try:
                y_pred = method_caller(estimator, "predict_proba", X)

                if y_type == "binary":
                    y_pred = self._select_proba_binary(y_pred, estimator.classes_)
                elif isinstance(y_pred, list):
                    y_pred = np.vstack([p[:, -1] for p in y_pred]).T
            except (NotImplementedError, AttributeError):
                y_pred = method_caller(estimator, "predict", X)

        if sample_weight is not None:
            return {
                f"label_{i}": self._sign * self._score_func(
                    y_true.iloc[:, i], y_pred[:, i], 
                    sample_weight=sample_weight, 
                    **self._kwargs
                ) for i in range(y_true.shape[1])
            }
        else:
            return {
                f"label_{i}": self._sign * self._score_func(
                    y_true.iloc[:, i], y_pred[:, i], 
                    **self._kwargs
                ) for i in range(y_true.shape[1])
            }