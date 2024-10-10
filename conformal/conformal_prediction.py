"""
Code originally from: https://github.com/google-deepmind/conformal_training/blob/main/conformal_prediction.py

Implementation of various conformal prediction methods.

Implements conformal prediction from [1,2,3]:

[1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
Classification withvalid and adaptive coverage.
NeurIPS, 2020. (APS)
[2] Anastasios N. Angelopoulos, Stephen Bates, Michael Jordan, Jitendra Malik.
Uncertainty sets for image classifiers using conformal prediction.
ICLR, 2021 (RAPS)
[3] Mauricio Sadinle, Jing Lei, and Larry A. Wasserman.
Least ambiguous set-valued classifiers with bounded error levels.
ArXiv, 2016. (THR)
"""

from typing import Optional, Callable, Any, Tuple

import numpy as np

_QuantileFn = Callable[[Any, float], float]


def conformal_quantile(array: np.ndarray, q: float) -> float:
    """
    Corrected quantile for conformal prediction.

    Wrapper for np.quantile, but instead of obtaining the q-quantile,
    it computes the (1 + 1/array.shape[0]) * q quantile. For conformal
    prediction, this is needed to obtain the guarantees for future test
    examples, see [1] Appendix Lemma for details.

    [1] Yaniv Romano, Evan Petterson, Emannuel J. Candes.
    Conformalized quantile regression. NeurIPS, 2019.

    :param array: input array to compute quantile of
    :param q: quantile to compute
    :return: (1 + 1/array.shape[0]) * q quantile of array.
    """

    return np.quantile(
        array, (1 + 1. / array.shape[0]) * q, method='midpoint')


def calibrate_threshold(
        probabilities: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.1,
        quantile_fn: _QuantileFn = conformal_quantile) -> float:
    """Probability/logit thresholding baseline calibration procedure.

    Finds a threshold based on input probabilities or logits. Confidence sets
    are defined as all classes above the threshold.

    NOTE: this is essentially the same as 'average error control' method from set-valued survey paper

    :param probabilities: predicted probabilities on validation set
    :param labels: ground truth labels on validation set
    :param alpha: confidence level
    :param quantile_fn: function to compute conformal quantile

    return: Threshold used to construct confidence sets
    """

    conformity_scores = probabilities[
        np.arange(probabilities.shape[0]), labels.astype(int)]
    return quantile_fn(conformity_scores, alpha)


def predict_threshold(probabilities: np.ndarray, tau: float) -> np.ndarray:
    """
    Probability/logit threshold baseline.

    Predicts all classes with probabilities/logits above given threshold
    as confidence sets.

    :param probabilities: predicted probabilities on test set
    :param tau: threshold for probabilities or logits
    :return: Confidence sets as 0-1array of same size as probabilities.
    """

    confidence_sets = (probabilities >= tau)
    return confidence_sets.astype(int)


def calibrate_raps(
        probabilities: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.1,
        k_reg: Optional[int] = None,
        lambda_reg: Optional[float] = None,
        rng: Optional[bool] = None,
        quantile_fn: _QuantileFn = conformal_quantile
) -> float:
    """
    Implementation of calibration for adaptive prediction sets.

    Following [1] and [2], this function implements adaptive prediction sets (APS)
    -- i.e., conformal classification. This methods estimates tau as outlined in
    [2] but without the confidence set size regularization. (i.e., RAPS not implemented atm)

    [1] Yaniv Romano, Matteo Sesia, Emmanuel J. Candes.
    Classification withvalid and adaptive coverage.
    NeurIPS, 2020.
    [2] Anastasios N. Angelopoulos, Stephen Bates, Michael Jordan, Jitendra Malik.
    Uncertainty sets for image classifiers using conformal prediction.
    ICLR, 2021

    :param probabilities: predicted probabilities on validation set
    :param labels: ground truth labels on validation set
    :param alpha: confidence level
    :param k_reg: target confidence set size for regularization
    :param lambda_reg: regularization weight
    :param rng: random key for uniform variables
    :param quantile_fn: function to compute conformal quantile
    :return: Threshold tau such that with probability 1 - alpha, the confidence set
    constructed from tau includes the true label
    """
    # check if regularization is being used
    reg = k_reg is not None and lambda_reg is not None

    sorting = np.argsort(-probabilities, axis=1)  # used sort from highest to lowest prob
    reverse_sorting = np.argsort(sorting)  # used to get cum_sum up to true class ID/arg
    indices = np.indices(probabilities.shape)
    sorted_probabilities = probabilities[indices[0], sorting]
    cum_probabilities = np.cumsum(sorted_probabilities, axis=1)

    rand = np.zeros(sorted_probabilities.shape[0])
    if rng:
        rand = np.random.uniform(size=(sorted_probabilities.shape[0],))
    # adding randomness to calibration improves efficency
    cum_probabilities -= np.expand_dims(rand,
                                        axis=1) * sorted_probabilities
    # get cumulative probabilities up to the correct class
    # reverse_sorting[label] find the index corresponding to the true class in the sorted/cum probs
    conformity_scores = cum_probabilities[
        np.arange(cum_probabilities.shape[0]),
        reverse_sorting[np.arange(reverse_sorting.shape[0]), labels]
    ]

    if reg:
        # in [2], it seems that L_i can be zero (i.e., true class has highest
        # probability), but we add + 1 in the second line for validation
        # as the true class is included by design and only
        # additional classes should be regularized
        conformity_reg = reverse_sorting[np.arange(reverse_sorting.shape[0]),
        labels]
        conformity_reg = conformity_reg - k_reg + 1
        conformity_reg = lambda_reg * np.maximum(conformity_reg, 0)
        conformity_scores += conformity_reg

    tau = quantile_fn(conformity_scores, 1 - alpha)
    return tau


def predict_raps(
        probabilities: np.ndarray,
        tau: float,
        k_reg: Optional[int] = None,
        lambda_reg: Optional[float] = None,
        rng: Optional[bool] = None) -> np.ndarray:
    """
    Get confidence sets using tau computed via aps_calibrate.
    :param probabilities: predicted probabilities on test set
    :param tau: threshold for probabilities or logits
    :param k_reg: target confidence set size for regularization
    :param lambda_reg: regularization weight
    :param rng: random key for uniform variables
    :return: Confidence sets as 0-1array of same size as probabilities
    """
    reg = k_reg is not None and lambda_reg is not None

    sorting = np.argsort(-probabilities, axis=1)  # used sort from highest to lowest prob
    indices = np.indices(probabilities.shape)
    sorted_probabilities = probabilities[indices[0], sorting]
    cum_probabilities = np.cumsum(sorted_probabilities, axis=1)

    if reg:
        # in [2], L is the number of classes for which cumulative probability
        # mass and regularizer are below tau + 1, we account for that in
        # the first line by starting to count at 1
        # cum_probabilities is shape (n_examples, n_classes)
        # first add +1 so indices start at 1
        reg_probabilities = np.repeat(  # np.arange creates 1D array from 0 to n_classes-1
            np.expand_dims(1 + np.arange(cum_probabilities.shape[1]), axis=0),  # expand dim to (1,n-1)
            cum_probabilities.shape[0], axis=0)  # repeat this n_examples times
        reg_probabilities = reg_probabilities - k_reg
        reg_probabilities = np.maximum(reg_probabilities, 0)  # only regularize classes after class k_reg
        cum_probabilities += lambda_reg * reg_probabilities  # add regularization penalty to these classes

    rand = np.ones((sorted_probabilities.shape[0]))
    if rng:
        rand = np.random.uniform(low=1, high=0, size=(sorted_probabilities.shape[0],))
    cum_probabilities -= np.expand_dims(rand, axis=1) * sorted_probabilities

    # get all the classes with cumulative probabilities less than threshold
    sorted_confidence_sets = (cum_probabilities <= tau)

    # reverse sorting by argsort the sorting indices
    reverse_sorting = np.argsort(sorting, axis=1)
    confidence_sets = sorted_confidence_sets[indices[0], reverse_sorting]  # get back original classes
    return confidence_sets.astype(int)