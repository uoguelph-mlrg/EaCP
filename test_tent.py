import argparse
import csv
from pathlib import Path

import torch
import torchvision
import numpy as np
from tqdm import tqdm
import pdb

import data

import utils
import tent
from data import imagenet_r_mask, indices_in_1k
import conformal_prediction as cp
import evaluation


def get_args_parser():
    parser = argparse.ArgumentParser('TTA & Conformal Prediction Distribution Shift', add_help=False)

    # data args
    parser.add_argument('--in1k_path', type=str,
                        default=r'/scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-resnet50.npz',
                        help='Location of imagenet1k validation set.')
    parser.add_argument('--dataset', type=str, help='Choose from [imagenet-r, imagenet-a, imagenet-v2, imagenet-c]')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type for imagenet-c')
    parser.add_argument('--severity', type=int, default=1, help='Severity level for imagenet-c')

    # training args
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for TTA')
    parser.add_argument('--alpha', type=float, default=0.1, help='Desired error rate (cov=1-alpha)')
    parser.add_argument('--ucp', action='store_true', help='Updated the conformal threshold')
    parser.add_argument('--tta', action='store_true', help='Perform TTA')
    parser.add_argument('--model', type=str, help='Base neural network model')

    parser.add_argument('--save-name', type=str, help='Name for results file')

    return parser


def evaluate(args):
    results = {}
    save_name = Path(args.save_name)
    save_loc = 'results' / save_name

    results_dict, tau_thr, upper_q, lower_q, cal_smx, cal_labels = utils.split_conformal(results, args.in1k_path,
                                                                                         args.alpha)

    if args.dataset == 'imagenet-r':
        dataset = data.INr()
        mask = imagenet_r_mask
    elif args.dataset == 'imagenet-a':
        dataset = data.INa()
        mask = indices_in_1k
    elif args.dataset == 'imagenet-v2':
        dataset = data.INv2()
        mask = None
    elif args.dataset == 'imagenet-c':
        dataset = data.INc(args.corruption, args.severity)
        mask = None
    else:
        raise ValueError('Dataset not supported')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = torchvision.models.resnet50(pretrained=True, progress=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.tta:
        model = setup_tent(model, mask)
    else:
        model.eval()

    correct = 0
    seen = 0
    cov = []
    sizes = []
    ent_covs = []
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        if (not args.tta) and (mask is not None):
            outputs = outputs[:, mask]

        correct += (outputs.argmax(1) == labels).sum()
        seen += outputs.shape[0]

        output_ent = utils.logit_entropy(outputs)  # get enrtopy from logits

        if args.ucp:  # update entropy quantile
            upper_q, tau_thr = utils.update_cp(output_ent, upper_q, cal_smx, cal_labels, args.alpha)

        ent_covs.append(((lower_q <= output_ent) & (output_ent <= upper_q)).sum() / len(output_ent))

        if args.ucp:
            # multiply by upper quantile to grow the softmax scores
            cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy() * upper_q, tau_thr)
        else:
            cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy(), tau_thr)

        # cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy(), tau_ent * upper_q.cpu().numpy())
        # tau_ent = cp.calibrate_threshold(cal_smx / cal_ent.reshape(-1,1), cal_labels, alpha)
        # cov_set = cp.predict_threshold(
        #     outputs.softmax(1).cpu().detach().numpy(), tau_ent * output_ent.detach().cpu().numpy().reshape(-1, 1))

        cov.append(float(evaluation.compute_coverage(cov_set, labels.cpu())))
        size, _ = evaluation.compute_size(cov_set)
        sizes.append(size)

    print(f'Accuracy: {(correct / seen) * 100} %')
    print(f'Coverage on OOD: {np.mean(cov)}')
    print(f'Inefficiency on OOD: {np.mean(sizes)}')

    results['ood_acc'] = ((correct / seen) * 100).item()
    results['ood_cov'] = np.mean(cov)
    results['ood_size'] = np.mean(sizes)

    with open(save_loc, 'w') as f:
        w = csv.writer(f)
        w.writerows(results.items())


def setup_tent(model, mask=None):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.SGD(params,
                                lr=0.00025,
                                momentum=0.9,
                                weight_decay=0.0)

    tent_model = tent.Tent(model, optimizer,
                           steps=1,
                           episodic=False,
                           mask=mask)
    print(f"model for adaptation: %s", model)
    print(f"params for adaptation: %s", param_names)
    print(f"optimizer for adaptation: %s", optimizer)
    return tent_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TTA & Conformal Prediction Distribution Shift',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate(args)
