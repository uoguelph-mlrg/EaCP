"""
Code originally from: https://github.com/google-deepmind/conformal_training/blob/main/evaluation.py
"""
from typing import Tuple

import numpy as np


def compute_conditional_accuracy(
        probabilities: np.ndarray, labels: np.ndarray,
        conditional_labels: np.ndarray, conditional_label: int) -> float:
    """Computes conditional accuracy given softmax probabilities and labels.

    Conditional accuracy is defined as the accuracy on a subset of the examples
    as selected using the conditional label(s). For example, this allows
    to compute accuracy conditioned on class labels.

    Args:
        :param probabilities: predicted probabilities on test set
        :param labels: ground truth labels on test set
        :param conditional_labels: conditional labels to compute accuracy on
        :param conditional_label: selected conditional label to compute accuracy on
        :return Accuracy
    """
    selected = (conditional_labels == conditional_label)
    num_examples = np.sum(selected)
    predictions = np.argmax(probabilities, axis=1)
    error = selected * (predictions != labels)
    error = np.where(num_examples == 0, 1, np.sum(error) / num_examples)
    return 1 - error


def compute_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute unconditional accuracy using compute_conditional_accuracy."""
    return compute_conditional_accuracy(
        probabilities, labels, np.zeros(labels.shape, int), 0)


def compute_conditional_multi_coverage(
        confidence_sets: np.ndarray, one_hot_labels: np.ndarray,
        conditional_labels: np.ndarray, conditional_label: int) -> float:
    """
    Compute coverage of confidence sets, potentially for multiple labels.

    The given labels are assumed to be one-hot labels and the implementation
    supports checking coverage of multiple classes, i.e., whether one of
    the given ground truth labels is in the confidence set.

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param one_hot_labels: ground truth labels on test set in one-hot format
    :param conditional_labels: conditional labels to compute coverage on a subset
    :param conditional_label: selected conditional to compute coverage for
    :return:
    """
    selected = (conditional_labels == conditional_label)  # select subset of labels
    num_examples = np.sum(selected)
    coverage = selected * np.clip(
        np.sum(confidence_sets * one_hot_labels, axis=1), 0, 1)
    coverage = np.where(num_examples == 0, 1, np.sum(coverage / num_examples))
    return coverage


def compute_coverage(
        confidence_sets: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute unconditional coverage using compute_conditional_multi_coverage

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param labels: ground truth labels on test set (not in one-hot format)
    :return: Coverage
    """
    one_hot_labels = np.eye(confidence_sets.shape[1])[labels]
    return compute_conditional_multi_coverage(
        confidence_sets, one_hot_labels, np.zeros(labels.shape, int), 0)


def compute_conditional_coverage(
        confidence_sets: np.ndarray, labels: np.ndarray,
        conditional_labels: np.ndarray, conditional_label: int) -> float:
    """
    Compute conditional coverage using compute_conditional_multi_coverage.

    :param confidence_sets: confidence sets on test set as 0-1 array
    :param labels: truth labels on test set (not in one-hot format)
    :param conditional_labels: conditional labels to compute coverage on a subset
    :param conditional_label: selected conditional to compute coverage for
    :return: Conditional coverage.
    """

    one_hot_labels = np.eye(confidence_sets.shape[1])[labels]
    return compute_conditional_multi_coverage(
        confidence_sets, one_hot_labels, conditional_labels, conditional_label)


def compute_conditional_size(
        confidence_sets: np.ndarray,
        conditional_labels: np.ndarray,
        conditional_label: int) -> Tuple[float, int]:
    """
    Compute confidence set size.

    :param confidence_sets: confidence sets on test set
    :param conditional_labels: conditional labels to compute size on
    :param conditional_label: selected conditional to compute size for
    :return: Average size.
    """

    selected = (conditional_labels == conditional_label)
    num_examples = np.sum(selected)
    size = selected * np.sum(confidence_sets, axis=1)
    size = np.where(num_examples == 0, 0, np.sum(size) / num_examples)
    return size, num_examples


def compute_size(confidence_sets: np.ndarray) -> Tuple[float, int]:
    """
    Compute unconditional coverage using compute_conditional_coverage
    :param confidence_sets: confidence sets on test set
    :return: Average size.
    """
    return compute_conditional_size(
        confidence_sets, np.zeros(confidence_sets.shape[0], int), 0)