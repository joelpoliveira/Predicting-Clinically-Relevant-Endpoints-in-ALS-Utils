# Author: Joel Oliveira
# Email: fc59442@alunos.fc.ul.pt
# Description: This file contains the implementation of a gridsearch for multi-label data. It is based on the GridSearchCV from sklearn.model_selection. 
#               The main difference is that it allows for labelwise binary metrics to be calculated in the gridsearch for single label analysis.
#               Most of the code in this page is copied from the sklearn repository. It only has located changes in order to allow to perform
#               binary scoring after each step, for each of the labels.
import time
import numbers
import warnings
import numpy as np

from joblib import logger
from itertools import product
from functools import partial
from contextlib import suppress
from traceback import format_exc
from collections import defaultdict, Counter

from numpy.ma import MaskedArray
from scipy.stats import rankdata

from sklearn.base import (
    is_regressor, is_classifier, clone
)

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._validation import (
    _warn_or_raise_about_fit_failures, _score, _insert_error_scores, 
    _normalize_score_results, _aggregate_score_dicts
)

from sklearn.metrics import check_scoring, make_scorer
from sklearn.metrics._scorer import (
    _BaseScorer, _cached_call,
    _check_multimetric_scoring, _MultimetricScorer
)

from sklearn.utils.multiclass import type_of_target
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    indexable, check_is_fitted, 
    _check_fit_params, _num_samples
)


class MLGridSearchCV(BaseSearchCV):
    """
    This class is based on GridSearchCV. It only has slight modifications to allow for labelwise binary metrics to be calculated in the gridsearch for single label analysis. 
    """
    _required_parameters = ["estimator", "param_grid"]
    def __init__(self,
        estimator,
        param_grid,
        *,
        scoring=None,
        bin_scoring=None,
        cv=None,
        n_jobs=-1,
        verbose=0,
        refit=True,
        pre_dispatch='2*n_jobs',
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        self.bin_scoring = bin_scoring
    
    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self.param_grid))

    def _check_scorers(self, in_scorers):
        if callable(in_scorers):
            scorers = in_scorers
            refit_metric = "score"
        elif in_scorers is None or isinstance(in_scorers, str):
            scorers = check_scoring(self.estimator, in_scorers)
            refit_metric = "score"
        else:
            scorers = _check_multimetric_scoring(self.estimator, in_scorers)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            
        return scorers, refit_metric
    
    def fit(self, X, y,*, groups=None, **fit_params):
        estimator = self.estimator
        
        ml_scorers, refit_metric = self._check_scorers(self.scoring)
        bin_scorers, _ = self._check_scorers(self.bin_scoring | self.scoring)
        bin_scorers = dict(filter(lambda x: x[0] not in ml_scorers, bin_scorers.items()))

        X,y,groups = indexable(X,y,groups)
        fit_params = _check_fit_params(X,fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel=Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            ml_scorer=ml_scorers,
            bin_scorer=bin_scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results={}

        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results
        
            self._run_search(evaluate_candidates)

            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        self.scorer_ = ml_scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits
        return self
    
    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        n_candidates = len(candidate_params)

        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})
        for key, val in results.items():
            results[key] = np.asarray(val)
        
        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(
                ~np.isfinite(array_means)
            ):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores "
                    f"are non-finite: {array_means}",
                    category=UserWarning,
                )

            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
                )
            )
            results["std_%s" % key_name] = array_stds

            if rank:
                # When the fit/scoring fails `array_means` contains NaNs, we
                # will exclude them from the ranking process and consider them
                # as tied with the worst performers.
                if np.isnan(array_means).all():
                    # All fit/scoring routines failed.
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method="min").astype(
                        np.int32, copy=False
                    )
                results["rank_%s" % key_name] = rank_result    

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            if type(test_scores_dict[scorer_name]).__module__ == np.__name__:
                _store(
                    "test_%s" % scorer_name,
                    test_scores_dict[scorer_name],
                    splits=True,
                    rank=True,
                    weights=None,
                )
                if self.return_train_score:
                    _store(
                        "train_%s" % scorer_name,
                        train_scores_dict[scorer_name],
                        splits=True,
                    )
            else:
                bin_score_dicts = _aggregate_score_dicts(test_scores_dict[scorer_name])
                if self.return_train_score:
                    train_scores_dict = _aggregate_score_dicts(out["train_scores"][scorer_name])
                for scorer_label in bin_score_dicts:
                    _store(
                        f"test_{scorer_name}_{scorer_label}",
                        bin_score_dicts[scorer_label],
                        splits=True,
                        rank=True,
                        weights=None
                    )
                    if self.return_train_score:
                        _store(
                            f"train_{scorer_name}_{scorer_label}",
                            train_scores_dict[scorer_label],
                            splits=True
                        )

        return results

def _fit_and_score(
    estimator,
    X,
    y,
    ml_scorer,
    bin_scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):

    """
    COPY OF sklearn.model_selection._validation._fit_and_score to include both multi-label
      scores for tunning and binary scores for labelwise evaluation;
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    
    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(ml_scorer, dict):
                ml_test_scores = {name: error_score for name in ml_scorer}
                if return_train_score:
                    ml_train_scores = ml_test_scores.copy()
            else:
                ml_test_scores = error_score
                if return_train_score:
                    ml_train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        ml_test_scores = _score(estimator, X_test, y_test, ml_scorer, error_score)
        bin_test_scores = _bin_score(estimator, X_test, y_test, bin_scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            ml_train_scores = _score(estimator, X_train, y_train, ml_scorer, error_score)
            bin_train_scores = _bin_score(estimator, X_train, y_train, bin_scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(ml_test_scores, dict):
                for scorer_name in sorted(ml_test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = ml_train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={ml_test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={ml_train_scores:.3f}, test={ml_test_scores:.3f})"
                else:
                    result_msg += f"{ml_test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    test_scores = ml_test_scores | bin_test_scores
    result["test_scores"] = test_scores
    if return_train_score:
        train_scores = ml_train_scores | bin_train_scores
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result

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
        
def make_labelwise_multilabel_scorer(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    **kwargs,
):
    """
    COPY OF sklearn.metrics.make_scorer to create a scorer for binary evaluation on multi-label data
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_threshold:
        return _Labelwise_Multilabel_ThreholdScorer(score_func, sign, kwargs)
    return _Labelwise_Multilabel_PredictScorer(score_func, sign, kwargs)