from ._scoring import _Labelwise_Multilabel_ThreholdScorer, _Labelwise_Multilabel_PredictScorer

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