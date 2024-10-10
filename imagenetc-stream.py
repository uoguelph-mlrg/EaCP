import argparse
import csv
import pickle
from pathlib import Path
from collections import defaultdict

import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
import pdb

import datasets
import models
import utils
import eata
import tent
import conformal_prediction as cp
import evaluation
from uncertainty_functions import logit_entropy
# Our new algorithm
from online_conformal.magnitude_learner import MagnitudeLearner, MagnitudeLearnerV2
from online_conformal.mag_learner_undiscounted import MagLearnUndiscounted
from online_conformal.saocp import SAOCP
from online_conformal.faci import FACI


def get_args_parser():
    parser = argparse.ArgumentParser('TTA & Conformal Prediction Distribution Shift', add_help=False)

    # data args
    parser.add_argument('--cal-path', type=str,
                        default=r'/scratch/ssd004/scratch/kkasa/inference_results/IN1k/imagenet-resnet50.npz',
                        help='Location of calibration data (e.g. imagenet1k validation set.)')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type for imagenet-c')
    parser.add_argument('--schedule', type=str, default='sudden', help='[sudden, gradual]')

    # training args
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for TTA')
    parser.add_argument('--lr', type=float, default=0.00025, help='TTA Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='TTA momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='TTA weight decay')

    parser.add_argument('--alpha', type=float, default=0.1, help='Desired error rate (cov=1-alpha)')
    parser.add_argument('--ucp', action='store_true', help='Updated the conformal threshold')
    parser.add_argument('--tta', action='store_true', help='Perform TTA')
    parser.add_argument('--model', type=str, default='resnet50', help='Base neural network model')
    parser.add_argument('--cp', type=str, default='thr', help='CP Method')

    parser.add_argument('--e-margin', type=float, default=1000,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--e-margin-scale', type=float, default=0.4,
                        help='hyperparameter for scaling margin')

    parser.add_argument('--d-margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    parser.add_argument('--save-name', type=str, help='Name for results file')

    return parser


def evaluate(args):
    print(f'Working on {args.corruption}')

    args.e_margin = math.log(args.e_margin) * args.e_margin_scale

    results = []
    save_name = Path(args.save_name + '.csv')
    save_folder = Path(f'results/{args.corruption}/{args.schedule}')
    save_folder.mkdir(parents=True, exist_ok=True)
    save_loc = save_folder / save_name

    results, tau_thr, upper_q, lower_q, cal_smx, cal_labels = utils.split_conformal(results, args.cal_path,
                                                                                    args.alpha, cp_method=args.cp)

    if args.corruption == 'mixed':
        sev_datasets = datasets.INc_stream_mixed(args.model)
    else:
        sev_datasets = datasets.INc_stream_single(args.corruption, args.model)
    sev_datasets[0] = datasets.IN1k(args.model)

    mask = None

    # what (if any) updates to perform at test time
    # updates = ['none', 'tta', 'ucp', 'both', MagnitudeLearner, MagLearnUndiscounted, MagnitudeLearnerV2, SAOCP, FACI]
    updates = [MagLearnUndiscounted]

    # the following assumes batch_size=64 TODO: make modular
    if args.schedule == 'gradual':
        run_length = 78
    else:
        run_length = 156
    total_run = 780  # total number of samples; 780*64 ~=50,000

    all_sizes = defaultdict(list)
    all_covs = defaultdict(list)
    all_lce = {}
    all_lss = {}
    args.ocp = False

    for update in updates:
        print(f'Working on update type: {update}')
        model = models.get_model('imagenet-c', args.model)
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
        elif update in [MagnitudeLearner, MagLearnUndiscounted, MagnitudeLearnerV2, SAOCP, FACI]:
            args.tta = False
            args.ucp = False
            args.ocp = True
            predictor = update(None, None, lifetime=32, coverage=args.alpha if args.cp == 'thr' else 1 - args.alpha, )
            update = update.__name__

        print(f'TTA: {args.tta}\nUCP: {args.ucp}')
        if args.tta:
            model = eata.configure_model(model)
            params, param_names = eata.collect_params(model)
            optimizer = torch.optim.SGD(params,
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.wd)
            model = eata.EATA(model, optimizer, e_margin=args.e_margin, d_margin=args.d_margin, mask=mask)

        else:
            model.eval()

        correct = 0
        seen = 0
        cov = []
        sizes = []
        ent_covs = []
        sevs = []
        lce = float('-inf')  # (worst) local coverage error, initialize to the smallest possible float
        lss = float('-inf')  # (worst) local set size
        state = np.random.RandomState(0)

        for t in tqdm(range(total_run)):
            lmbda = 0.5
            k_reg = 7
            n_class = 1000
            sev = utils.t2sev(t, run_length=run_length, schedule=args.schedule)
            # print(sev)
            if sev == 0:
                # continue
                sev = 1
            loader = torch.utils.data.DataLoader(sev_datasets[sev], batch_size=args.batch_size, shuffle=True)
            batch = next(iter(loader))
            sevs.append(sev)
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if (not args.tta) and (mask is not None):
                outputs = outputs[:, mask]

            correct += (outputs.argmax(1) == labels).sum()
            seen += outputs.shape[0]

            output_ent = logit_entropy(outputs)  # get entropy from logits

            if args.ucp:  # update entropy quantile
                upper_q, tau_thr = utils.update_cp(output_ent, upper_q, cal_smx, cal_labels, args.alpha, args.cp)

            ent_covs.append(((lower_q <= output_ent) & (output_ent <= upper_q)).sum() / len(output_ent))
            if args.ucp:
                if args.cp == 'thr':
                    # multiply by upper quantile to grow the softmax scores
                    cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy() * upper_q, tau_thr)
                elif args.cp == 'raps':
                    cov_set = cp.predict_raps(outputs.softmax(1).cpu().detach().numpy() / upper_q, tau_thr, k_reg=7,
                                              lambda_reg=0.5, rng=True)
                else:
                    raise ValueError('CP Method not supported.')
                # record stats
                cov.append(float(evaluation.compute_coverage(cov_set, labels.cpu())))
                size, _ = evaluation.compute_size(cov_set)
                sizes.append(size)

            elif args.ocp:
                batch_cov = []  # coverage for this batch
                batch_size = []  # size for this batch
                for output, label in zip(outputs, labels):
                    label = label.cpu().numpy().astype(int)
                    output = output.softmax(0).cpu().detach().numpy()
                    if args.cp == 'thr':
                        _, s_hat = predictor.predict(horizon=1)  # get threshold prediction from OCP
                        # form confidence set
                        cov_set = cp.predict_threshold(output, s_hat)
                        s_opt = output[label]  # true value of output
                        # update ocp
                        predictor.update(ground_truth=pd.Series([s_opt]), forecast=pd.Series([0]), horizon=1)

                        # record stats
                        batch_cov.append(float(evaluation.compute_coverage(cov_set.reshape(1, -1), label)))
                        size, _ = evaluation.compute_size(cov_set.reshape(1, -1))
                        batch_size.append(size)

                    elif args.cp == 'raps':
                        # Convert probability to APS score
                        i_sort = np.flip(np.argsort(output))
                        p_sort_cumsum = np.cumsum(output[i_sort]) - state.rand() * output[i_sort]
                        s_sort_cumsum = p_sort_cumsum + lmbda * np.sqrt(np.cumsum([i > k_reg for i in range(n_class)]))
                        w_opt = np.argsort(i_sort)[label] + 1
                        s_opt = s_sort_cumsum[w_opt - 1]

                        _, s_hat = predictor.predict(horizon=1)
                        # update ocp
                        predictor.update(ground_truth=pd.Series([s_opt]), forecast=pd.Series([0]), horizon=1)
                        w = np.sum(s_sort_cumsum <= s_hat)
                        batch_size.append(w)
                        batch_cov.append(w >= w_opt)
                    else:
                        raise ValueError('CP Method not supported.')
                # average on batch
                cov.append(np.mean(batch_cov))
                sizes.append(np.mean(batch_size))

            else:
                if args.cp == 'thr':
                    cov_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy(), tau_thr)
                elif args.cp == 'raps':
                    cov_set = cp.predict_raps(outputs.softmax(1).cpu().detach().numpy(), tau_thr, k_reg=7,
                                              lambda_reg=0.5,
                                              rng=True)

                cov.append(float(evaluation.compute_coverage(cov_set, labels.cpu())))
                size, _ = evaluation.compute_size(cov_set)
                sizes.append(size)

            # (target cov) - (empirical cov) - lower is better
            if len(cov) >= 2:  # error across two batches (corresponds to 128 samples)
                err = (((1 - args.alpha) - cov[-2]) +
                       ((1 - args.alpha) - cov[-1])) / 2
                lce = max(lce, err)

                worst_size = (sizes[-1] + sizes[-2]) / 2
                lss = max(lss, worst_size)

        # store the size and coverage results for this method
        all_sizes[update] = sizes
        all_covs[update] = cov
        all_lce[update] = lce
        all_lss[update] = lss

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

    save_dict = {}
    save_dict['covs'] = all_covs
    save_dict['sevs'] = sevs
    save_dict['sizes'] = all_sizes
    save_dict['lce'] = all_lce
    save_dict['lss'] = all_lss

    with open(save_folder / f'{args.save_name}-results.pkl', "wb") as fp:  # save worst local set size
        pickle.dump(save_dict, fp)

    # with open(save_folder / f'{args.save_name}-covs.pkl', "wb") as fp:  # save coverage stats for further processing
    #     pickle.dump(all_covs, fp)

    # with open(save_folder / f'{args.save_name}-sevs.pkl', "wb") as fp:  # save sev levels
    #     pickle.dump(sevs, fp)
    #
    # with open(save_folder / f'{args.save_name}-sizes.pkl', "wb") as fp:  # save size stats
    #     pickle.dump(all_sizes, fp)
    #
    # with open(save_folder / f'{args.save_name}-LCE.pkl', "wb") as fp:  # save worst local coverage error
    #     pickle.dump(all_lce, fp)
    #
    # with open(save_folder / f'{args.save_name}-LSS.pkl', "wb") as fp:  # save worst local set size
    #     pickle.dump(all_lce, fp)


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
