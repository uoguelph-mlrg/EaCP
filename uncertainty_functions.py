"""This file defines various uncertainty functions that can be used to adjust the conformal thresholds"""
import torch


def logit_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def smx_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of prediction distribution from softmax scores."""
    return -(x * x.log()).sum(1)


#the last layer mapping rep_x to y is assumed to be linear here
def calculate_dist_torch(rep_x_source, rep_x_target, z, device):
<<<<<<< HEAD
=======

>>>>>>> 51c447057d10dde3b0f6513712812cde992a1b94
    rep_x_source = torch.hstack((rep_x_source, torch.ones(rep_x_source.shape[0], 1).to(torch.device(device))))
    rep_x_target = torch.hstack((rep_x_target, torch.ones(rep_x_target.shape[0], 1).to(torch.device(device))))
    d = rep_x_source.shape[1]
    num_samples_source = rep_x_source.shape[0]
    num_samples_target = rep_x_target.shape[0]
    M0 = torch.zeros((d, d)).to(torch.device(device))
    for i in range(num_samples_target):
        M0 = M0 + 1.0/num_samples_target * torch.outer(rep_x_target[i], rep_x_target[i])
<<<<<<< HEAD
    M_z = M0
    for i in range(num_samples_source):
        M_z = M_z - z[i] * torch.outer(rep_x_source[i], rep_x_source[i])
=======

    M_z = M0
    for i in range(num_samples_source):
        M_z = M_z - z[i] * torch.outer(rep_x_source[i], rep_x_source[i])

>>>>>>> 51c447057d10dde3b0f6513712812cde992a1b94
    distance = torch.max(torch.max((torch.linalg.eigvals(M_z)).real), torch.max((torch.linalg.eigvals(-M_z)).real))
    return distance