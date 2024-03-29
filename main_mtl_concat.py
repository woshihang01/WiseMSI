import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils_mtl_concat import train
from datasets.dataset_mtl_concat import Generic_WSI_MTL_Dataset, Generic_MIL_MTL_Dataset
from datasets.dataset_rnn import Generic_WSI_RNN_Dataset
# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import logging

# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, help='h5file directory')
parser.add_argument('--max_epochs', type=int, default=300,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate (default: 0.00005)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use, '
                         + 'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--task', type=str, choices=['msi_classifier'])
parser.add_argument('--dataset_csv', type=str, help='dataset_csv')
parser.add_argument('--model_type', default=False, choices=['toad', 'toad_cosine', 'rnn', 'mil', 'attmil', 'varmil'])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_cls_test_auc = []
    all_cls_val_auc = []
    all_cls_test_acc = []
    all_cls_val_acc = []

    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))

        logging.info(
            'training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, cls_test_auc, cls_val_auc, cls_test_acc, cls_val_acc = train(datasets, i, args)
        all_cls_test_auc.append(cls_test_auc)
        all_cls_val_auc.append(cls_val_auc)
        all_cls_test_acc.append(cls_test_acc)
        all_cls_val_acc.append(cls_val_acc)

        # write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'cls_test_auc': all_cls_test_auc,
                             'cls_val_auc': all_cls_val_auc, 'cls_test_acc': all_cls_test_acc,
                             'cls_val_acc': all_cls_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


if __name__ == "__main__":
    set_log('./logs/main_mtl_concat.log')


    def seed_torch(seed=7):
        import random
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    seed_torch(args.seed)

    settings = {'num_splits': args.k,
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs,
                'results_dir': args.results_dir,
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'seed': args.seed,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt}

    logging.info('\nLoad Dataset')

    if args.task == 'msi_classifier':
        if args.model_type == 'toad' or args.model_type == 'toad_cosine' or args.model_type == 'mil' or args.model_type == 'attmil' or args.model_type == 'varmil':
            args.n_classes = 2
            dataset = Generic_MIL_MTL_Dataset(csv_path=args.dataset_csv,
                                              data_dir=args.data_root_dir,
                                              shuffle=False,
                                              seed=args.seed,
                                              print_info=True,
                                              label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                              label_cols=['label'],
                                              patient_strat=False)
        elif args.model_type == 'rnn':
            args.n_classes = 2
            dataset = Generic_WSI_RNN_Dataset(csv_path=args.dataset_csv,
                                              data_dir=args.data_root_dir, n=100,
                                              print_info=True,
                                              label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                              label_cols=['label'])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(100)))
    else:
        args.split_dir = os.path.join('splits', args.split_dir)
    assert os.path.isdir(args.split_dir)

    settings.update({'split_dir': args.split_dir})

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    logging.info("################# Settings ###################")
    for key, val in settings.items():
        logging.info("{}:  {}".format(key, val))
    results = main(args)
    logging.info("finished!")
    logging.info("end script")
