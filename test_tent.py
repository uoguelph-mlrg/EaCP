import argparse
import csv
from pathlib import Path

import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb

import data
import models
import utils
import tent
from uncertainty_functions import logit_entropy
import conformal_prediction as cp
import evaluation


def get_args_parser():
    parser = argparse.ArgumentParser('TTA & Conformal Prediction Distribution Shift', add_help=False)

    # data args
    parser.add_argument('--cal-path', type=str,
                        default=r'/scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-resnet50.npz',
                        help='Location of calibration data (e.g. imagenet1k validation set.)')
    parser.add_argument('--dataset', type=str,
                        help='Choose from [imagenet-r, imagenet-a, imagenet-v2, imagenet-c, rxrx1, fmow, iwildcam]')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type for imagenet-c')
    parser.add_argument('--severity', type=int, default=1, help='Severity level for imagenet-c')

    # training args
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for TTA')
    parser.add_argument('--lr', type=float, default=0.00025, help='TTA Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='TTA momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='TTA weight decay')

    parser.add_argument('--alpha', type=float, default=0.1, help='Desired error rate (cov=1-alpha)')
    # parser.add_argument('--ucp', action='store_true', help='Updated the conformal threshold')
    # parser.add_argument('--tta', action='store_true', help='Perform TTA')
    parser.add_argument('--model', type=str, default='resnet50', help='Base neural network model')

    parser.add_argument('--save-name', type=str, help='Name for results file')

    return parser


def evaluate(args):
    print(f'Working on {args.dataset}')

    results = []
    save_name = Path(args.save_name + '.csv')
    save_folder = Path(f'results/{args.dataset}')
    if args.dataset=='imagenet-c':
        save_folder / args.corruption
    save_folder.mkdir(parents=True, exist_ok=True)
    save_loc = save_folder / save_name

    results, tau_thr, upper_q, lower_q, cal_smx, cal_labels = utils.split_conformal(results, args.cal_path,
                                                                                    args.alpha)

    dataloader, mask = data.get_data(data_name=args.dataset, args=args)

    updates = ['none', 'tta', 'ucp', 'both']  # what (if any) updates to perform at test time

    for update in updates:
        print(f'Working on update type: {update}')
        model = models.get_model(args.dataset, args.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if update == 'none':
            args.tta = False
            args.ucp = False
        elif update == 'tta':
            args.tta = True
            args.ucp = False
        elif update == 'ucp':
            args.tta = False
            args.ucp = True
        elif update == 'both':
            args.tta = True
            args.ucp = True
        print(f'TTA: {args.tta}\nUCP: {args.ucp}')
        if args.tta:
            model = setup_tent(model, mask)
        else:
            model.eval()

        correct = 0
        seen = 0
        cov = []
        sizes = []
        ent_covs = []
        for batch in tqdm(dataloader):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if (not args.tta) and (mask is not None):
                outputs = outputs[:, mask]

            correct += (outputs.argmax(1) == labels).sum()
            seen += outputs.shape[0]

            output_ent = logit_entropy(outputs)  # get entropy from logits

            if args.ucp:  # update entropy quantile
                upper_q, tau_thr = utils.update_cp_batch(output_ent, upper_q, cal_smx, cal_labels, args.alpha)

            ent_covs.append(((lower_q <= output_ent) & (output_ent <= upper_q)).sum() / len(output_ent))

            if args.ucp:
                # multiply by upper quantile to grow the softmax scores
                cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy() * upper_q, tau_thr)
            else:
                cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy(), tau_thr)

            cov.append(float(evaluation.compute_coverage(cov_set, labels.cpu())))
            size, _ = evaluation.compute_size(cov_set)
            sizes.append(size)

        print(f'Accuracy: {(correct / seen) * 100} %')
        print(f'Coverage on OOD: {np.mean(cov)}')
        print(f'Inefficiency on OOD: {np.mean(sizes)}')

        results_dict = {}  # temporary dict to store results
        results_dict['update'] = update
        results_dict['ood_acc'] = ((correct / seen) * 100).item()
        results_dict['ood_cov'] = np.mean(cov)
        results_dict['ood_size'] = np.mean(sizes)
        results.append(results_dict)

    # Convert the data list to a pandas DataFrame
    results_df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    results_df.to_csv(save_loc, index=False)

    # with open(save_loc, 'w') as f:
    #     w = csv.writer(f)
    #     w.writerows(results.items())


def setup_tent(model, mask=None):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd)

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
