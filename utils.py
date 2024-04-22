import numpy as np
import torch.nn
import torchvision

import conformal_prediction as cp
from uncertainty_functions import smx_entropy
import evaluation


def pinball_loss_grad(y, yhat, q: float):
    return -q * (y > yhat) + (1 - q) * (y < yhat)


def split_conformal(results: list, cal_path: str, alpha: float):
    """Perform split conformal prediction and get conformal threshold"""
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

    # find quantiles for the entropy of the prediction distribution
    cal_ent, val_ent = smx_entropy(torch.Tensor(cal_smx)).numpy(), smx_entropy(torch.Tensor(val_smx)).numpy()
    lower_q = np.quantile(cal_ent, alpha / 2)
    upper_q = np.quantile(cal_ent, 1 - alpha / 2)
    # use this to form a prediction interval & check coverage
    pred_int = ((lower_q <= val_ent) & (val_ent <= upper_q)).sum()  # entropy should be within these quantiles
    print(f'Entropy Coverage on validation set: {pred_int / len(val_ent)}')

    # evaluate accuracy
    acc_cal = evaluation.compute_accuracy(val_smx, val_labels)
    # calibrate on imagenet calibration set
    tau_thr = cp.calibrate_threshold(cal_smx, cal_labels, alpha)  # get conformal quantile
    # get confidence sets
    conf_set_thr = cp.predict_threshold(val_smx, tau_thr)
    # evaluate coverage
    cov_thr_in1k = float(evaluation.compute_coverage(conf_set_thr, val_labels))
    # evaluate set size
    size_thr_in1k, _ = evaluation.compute_size(conf_set_thr)
    print(f'Accuracy on Calibration data: {acc_cal}')
    print(f'Coverage on Calibration data: {cov_thr_in1k}')
    print(f'Inefficiency on Calibration data: {size_thr_in1k}')

    results_dict = {}
    results_dict['update'] = 'calibration'
    results_dict['cal_acc'] = acc_cal
    results_dict['cal_cov'] = cov_thr_in1k
    results_dict['cal_size'] = size_thr_in1k
    results.append(results_dict)

    return results, tau_thr, upper_q, lower_q, cal_smx, cal_labels


def update_cp(output_ent, upper_q, cal_smx, cal_labels, alpha):
    # adjust the entropy quantile to ensure entropy coverage
    loss = pinball_loss_grad(upper_q, output_ent.cpu().detach().numpy(), alpha).mean()
    upper_q += loss

    # Re-calibrate, using the calibration dataset, and the new entropy quantile
    tau_thr = cp.calibrate_threshold(cal_smx / upper_q, cal_labels,
                                     alpha)  # deflate scores by the entropy quantile

    return upper_q, tau_thr


def update_cp_batch(output_ent, upper_q, cal_smx, cal_labels, alpha):
    # adjust the entropy quantile to ensure entropy coverage
    # loss = pinball_loss_grad(upper_q, output_ent.cpu().detach().numpy(), 0.1).mean()
    # upper_q += loss
    upper_q = np.quantile(output_ent.cpu().detach().numpy(),
                          1 - alpha)  # using the 90th quantile of just batch not whole dist
    # Re-calibrate, using the calibration dataset, and the new entropy quantile
    tau_thr = cp.calibrate_threshold(cal_smx / upper_q, cal_labels,
                                     alpha)  # deflate scores by the entropy quantile

    return upper_q, tau_thr


def t_to_sev(t, window, run_length=500, schedule=None):
    if t < window or schedule in [None, "None", "none"]:
        return 0
    t_base = t - window // 2
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    if schedule == "random_sudden":
        return np.clip(np.random.randint(0, 10) * ((t_base // run_length) % 2), 0, 5)
    if schedule == "random_gradual":
        k = (((t_base * abs(np.random.uniform(1, 1.5))) // run_length) % 10)
        return (k if k <= 5 else 10 - k) * np.random.randint(1, 2)
    return 5 * ((t_base // run_length) % 2)  # default: sudden schedule


def t2sev(t, run_length=7, schedule=None):
    t_base = t
    if schedule == "gradual":
        k = (t_base // run_length) % 10
        return k if k <= 5 else 10 - k
    else:
        return 5 * ((t_base // run_length) % 2)  # default: sudden schedule
