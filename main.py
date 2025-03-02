import argparse
import math
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import loader
from TTA import eata
from models import models
import utils
from uncertainty_functions import logit_entropy
from conformal import conformal_prediction as cp
from conformal import evaluation


def get_args_parser():
    parser = argparse.ArgumentParser('EaCP and ECP Experiments', add_help=False)
    # path args
    parser.add_argument('--cal-path', type=str,
                        default=r'inference_results/IN1k/imagenet-resnet50.npz',
                        help='Location of calibration data (e.g. imagenet1k validation set.)')
    parser.add_argument('--save-name', type=str, help='Name for results file')

    # data args
    parser.add_argument('--dataset', type=str,
                        help='Choose from [imagenet-r, imagenet-a, imagenet-v2, imagenet-c, rxrx1, fmow, iwildcam]')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type for imagenet-c')
    parser.add_argument('--severity', type=int, default=1, help='Severity level for imagenet-c')

    # CP args
    parser.add_argument('--scaling-factor', type=int, default=2,
                        help='Scaling factor for entropy quantile; refer to Sec 4.3')
    parser.add_argument('--alpha', type=float, default=0.1, help='Desired error rate (cov=1-alpha)')
    parser.add_argument('--cp', type=str, default='thr', help='CP Method')
    parser.add_argument('--updates', '--list', nargs='+', default=['none', 'tta', 'ecp', 'eacp', 'naive'],
                        help='What (if any) updates to perform at test time; none is eq. to splitCP',)

    # training args
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for TTA')
    parser.add_argument('--lr', type=float, default=0.00025, help='TTA Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='TTA momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='TTA weight decay')
    parser.add_argument('--model', type=str, default='resnet50', help='Base neural network model')

    # EATA TTA Hparams
    parser.add_argument('--e-margin', type=float, default=1000,
                        help='entropy margin E_0 in Eqn. (3) of EATA paper, for filtering reliable samples')
    parser.add_argument('--e-margin-scale', type=float, default=0.4,
                        help='hyperparameter for scaling margin, for EATA')
    parser.add_argument('--d-margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) of EATA paper, for filtering redundant samples')

    return parser


def evaluate(args):
    print(f'Working on {args.dataset}')

    args.e_margin = math.log(args.e_margin) * args.e_margin_scale  # for EATA TTA

    results = []
    save_name = Path(args.save_name + '.csv')
    save_folder = Path(f'results/{args.dataset}')
    if args.dataset == 'imagenet-c':
        save_folder = save_folder / args.corruption

    save_folder.mkdir(parents=True, exist_ok=True)
    save_loc = save_folder / save_name

    results, tau_thr = utils.split_conformal(results, args.cal_path, args.alpha, args.cp)

    dataloader, mask = loader.get_data(data_name=args.dataset, args=args)

    for update in args.updates:
        print(f'Working on update type: {update}')
        model = models.get_model(args.dataset, args.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if update == 'none':
            args.tta = False
            args.ecp = False
            args.naive = False
        elif update == 'tta':
            args.tta = True
            args.ecp = False
            args.naive = False
        elif update == 'ecp':
            args.tta = False
            args.ecp = True
            args.naive = False
        elif update == 'eacp':
            args.tta = True
            args.ecp = True
            args.naive = False
        elif update == 'naive':
            args.tta = False
            args.ecp = False
            args.naive = True
        print(f'TTA: {args.tta}\nECP: {args.ecp}')

        # initialize model for test time adaptation.
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
        for batch in tqdm(dataloader):
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            if (not args.tta) and (mask is not None):  # mask for IN1k variants that use a subset of classes
                outputs = outputs[:, mask]

            correct += (outputs.argmax(1) == labels).sum()
            seen += outputs.shape[0]

            output_ent = logit_entropy(outputs)  # get entropy from logits

            if args.ecp:  # update entropy quantile
                beta = utils.update_beta_batch(output_ent, args.alpha)
                # beta = utils.update_beta_online(output_ent, beta, args.alpha)

            if args.ecp:
                if args.cp == 'thr':
                    # form prediction set by adjusting scores
                    pred_set = cp.predict_threshold(
                        outputs.softmax(1).cpu().detach().numpy() * (beta ** args.scaling_factor), tau_thr)
                else:
                    raise ValueError(f'{args.cp} CP Method not supported')
            elif args.naive:
                pred_set = cp.predict_raps(outputs.softmax(1).cpu().detach().numpy(), 1 - args.alpha)
            elif args.cp == 'thr':  # args.cp is necessary arg so entering here is equiv. to regular SplitCP
                pred_set = cp.predict_threshold(outputs.softmax(1).cpu().detach().numpy(), tau_thr)

            cov.append(float(evaluation.compute_coverage(pred_set, labels.cpu())))
            size, _ = evaluation.compute_size(pred_set)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Entropy Adapted CP',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate(args)
