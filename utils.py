import numpy as np
import torch.nn
import torchvision

import conformal_prediction as cp
import evaluation


def logit_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def smx_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from softmax scores."""
    return -(x * x.log()).sum(1)


def pinball_loss_grad(y, yhat, q: float):
    return -q * (y > yhat) + (1 - q) * (y < yhat)


def split_conformal(results_dict: dict, in1k_path: str, alpha: float):
    # # # # # # # # CALIBRATION # # # # # # #
    print('Calibrating conformal')
    print('Working on ImageNet')

    # start by loading and calibrating on imagenet1k validation set
    in1k_data = np.load(in1k_path)
    in1k_smx = in1k_data['smx']  # get softmax scores
    in1k_labels = in1k_data['labels'].astype(int)

    # Split the softmax scores into calibration and validation sets
    n = int(len(in1k_labels) * 0.5)
    idx = np.array([1] * n + [0] * (in1k_smx.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_smx, val_smx = in1k_smx[idx, :], in1k_smx[~idx, :]
    cal_labels, val_labels = in1k_labels[idx], in1k_labels[~idx]

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

    results_dict['cal_acc'] = acc_cal
    results_dict['cal_cov'] = cov_thr_in1k
    results_dict['cal_size'] = size_thr_in1k

    return results_dict, tau_thr, upper_q, lower_q, cal_smx, cal_labels


def update_cp(output_ent, upper_q, cal_smx, cal_labels, alpha):
    # adjust the entropy quantile to ensure entropy coverage
    loss = pinball_loss_grad(upper_q, output_ent.cpu().detach().numpy(), 0.1).mean()
    upper_q += loss

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

