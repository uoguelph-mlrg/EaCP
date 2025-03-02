"""
Here we implement some CP / training helper functions.
"""
import numpy as np
import torch.nn
import torch

import conformal_prediction as cp
from uncertainty_functions import smx_entropy
import evaluation


def pinball_loss_grad(y: float, yhat: np.ndarray, q: float) -> np.ndarray:
    """
    Compute the gradient of the pinball loss function.

    :param y: True values.
    :param yhat: Predicted values.
    :param q: Quantile level for the loss calculation.
    :return: Gradient of the pinball loss.
    """
    return -q * (y > yhat) + (1 - q) * (y < yhat)


def split_conformal(results: list[dict],
                    cal_path: str,
                    alpha: float,
                    cp_method: str) -> tuple[list[dict], float]:
    """
    Perform split conformal prediction and obtain the conformal threshold.

    :param results: List of dicts that will store metrics of interest.
    :param cal_path: Path to saved softmax outputs / labels from the calibration dataset.
    :param alpha: Target error level for conformal prediction; coverage is 1 - alpha.
    :param cp_method: Which cp method to use
    :param cp_method: The conformal prediction method to use ('thr' or 'raps').
    :return: Updated results list, conformal threshold (tau_thr), upper and lower entropy quantiles (upper_q, lower_q),
             and the calibration softmax scores and labels (cal_smx, cal_labels).
    """
    # # # # # # # # CALIBRATION # # # # # # #
    print('Calibrating conformal')

    # start by loading and calibrating on imagenet1k validation set
    data = np.load(cal_path)
    smx = data['smx']  # get softmax scores
    labels = data['labels'].astype(int)

    # Split the softmax scores into calibration and validation sets
    n = int(len(labels) * 0.5)
    idx = np.array([1] * n + [0] * (smx.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = smx[idx, :], smx[~idx, :]
    cal_labels, val_labels = labels[idx], labels[~idx]

    # evaluate accuracy
    acc_cal = evaluation.compute_accuracy(val_smx, val_labels)
    # calibrate on imagenet calibration set
    if cp_method == 'thr':
        tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)  # get conformal quantile
    elif cp_method == 'raps':
        tau_thr = cp.calibrate_raps(cal_smx, cal_labels, alpha, k_reg=5, lambda_reg=0.01, rng=True)
    else:
        raise ValueError('CP method not supported choose from [thr, raps]')

    # get confidence sets
    if cp_method == 'thr':
        conf_set_thr = cp.predict_threshold(val_smx, tau_thr)
    elif cp_method == 'raps':
        conf_set_thr = cp.predict_raps(val_smx, tau_thr, k_reg=5, lambda_reg=0.01, rng=True)
    else:
        raise ValueError('CP method not supported choose from [thr, raps]')

    # evaluate coverage
    cov_thr_in1k = float(evaluation.compute_coverage(conf_set_thr, val_labels))
    # evaluate set size
    size_thr_in1k, _ = evaluation.compute_size(conf_set_thr)
    print(f'Accuracy on Calibration data: {acc_cal}')
    print(f'Coverage on Calibration data: {cov_thr_in1k}')
    print(f'Inefficiency on Calibration data: {size_thr_in1k}')

    results_dict = {
        'update': 'calibration',
        'cal_acc': acc_cal,
        'cal_cov': cov_thr_in1k,
        'cal_size': size_thr_in1k
    }
    results.append(results_dict)

    return results, tau_thr


def update_beta_online(output_ent: torch.Tensor, beta: float, alpha: float) -> float:
    """
    Update the estimated \beta entropy quantile online for use in adapting the conformal prediction threshold, see
    Eq. 3 of the paper.

    :param output_ent: Entropy of the output predictions.
    :param beta: Entropy quantile estimate.
    :param alpha: Target error level (1 - alpha = coverage).
    :return: Updated entropy quantile.
    """
    # update the beta entropy quantile using pinball loss
    loss = pinball_loss_grad(beta, output_ent.cpu().detach().numpy(), alpha).mean()
    beta += loss

    return beta


def update_beta_batch(output_ent: torch.Tensor, alpha: float) -> float:
    """
    Instead of updating the \beta quantile online, we can simply use the entropy quantile on a particular batch of data
    (or the entire dataset if available). On a large enough batch size, the difference with online estimate is
    negligible.

    :param output_ent: Entropy of the output predictions.
    :param alpha: Target error level (1 - alpha = coverage).
    :return: Entropy quantile of the batch / dataset.
    """

    # Find the entropy quantile on the batch of data
    upper_q = np.quantile(output_ent.cpu().detach().numpy(), 1 - alpha)

    return upper_q


def t2sev(t, run_length=7, schedule=None):
    """
    Time step to severity level, for continious shifts.
    """
    t_base = t
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    else:
        return 5 * ((t_base // run_length) % 2)  # default: sudden schedule
