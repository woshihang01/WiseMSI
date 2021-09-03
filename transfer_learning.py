'''
1.设置训练集、验证集、测试集
2.加载现有模型
3.开始训练
'''

import torch
from torch import nn
import numpy as np
import logging, os, argparse
from datasets.dataset_mtl_concat import save_splits
from utils.utils import get_split_loader
from models.model_toad import TOAD_fc_mtl_concat
from utils.utils import get_optim
from utils.core_utils_mtl_concat import EarlyStopping, train_loop, validate, summary
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_log(logfileName='./transfer_learning.log', level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        filename=logfileName,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


set_log()


def train(datasets, cur, args):
    """
        train for a single fold
    """
    logging.info('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(0 if test_split is None else len(test_split)))
    weight_CE = torch.FloatTensor([1, 5]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_CE).to(device)

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'model_type': args.model_type}

    model = TOAD_fc_mtl_concat(**model_dict).to(device)

    model.load_state_dict(
        torch.load('C:/Code/TOAD/results/msi_classifier_stain_tools_lossweight_1vs5_s1/s_9_checkpoint.pt'))
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split)
    test_loader = get_split_loader(test_split)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=40, stop_epoch=100, verbose=True)  # 连续patience轮，并且总论此超过stop_epoch轮就会终止

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
        stop = validate(cur, epoch, model, val_loader, args.n_classes,
                        early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, cls_val_error, cls_val_auc, _ = summary(model, val_loader, args.n_classes)
    logging.info('Cls Val error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_val_error, cls_val_auc))

    results_dict, cls_test_error, cls_test_auc, acc_loggers = summary(model, test_loader, args.n_classes)
    logging.info('Cls Test error: {:.4f}, Cls ROC AUC: {:.4f}'.format(cls_test_error, cls_test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_loggers.get_summary(i)
        logging.info('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_tpr'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/cls_val_error', cls_val_error, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/cls_test_error', cls_test_error, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)

    writer.close()
    return results_dict, cls_test_auc, cls_val_auc, 1 - cls_test_error, 1 - cls_val_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
    parser.add_argument('--pt_files_dir', default='C:/RESULTS_TUMOR_NORM/patches', type=str,
                        help='h5file directory')
    parser.add_argument('--task', type=str, choices=['msi_classifier_transfer_learning_370'])
    parser.add_argument('--exp_code', type=str)
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='manually specify the set of splits to use, instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=1e-6, help='weight decay (default: 1e-5)')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
    parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='maximum number of epochs to train (default: 200)')
    args = parser.parse_args()

    dataset = Generic_MIL_MTL_Dataset(csv_path='dataset_csv/dataset_tcga_2021.07.22.csv',
                                      data_dir=args.pt_files_dir,
                                      shuffle=False,
                                      seed=args.seed,
                                      print_info=True,
                                      label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                      label_cols=['label'],
                                      patient_strat=False)

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
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path='{}/splits_{}.csv'.format(
                                                                             args.split_dir, i))

        print(
            'training: {}, validation: {}, testing: {}'.format(len(train_dataset), len(val_dataset), len(test_dataset) if test_dataset is not None else 0))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, cls_test_auc, cls_val_auc, cls_test_acc, cls_val_acc = train(datasets, i, args)
        all_cls_test_auc.append(cls_test_auc)
        all_cls_val_auc.append(cls_val_auc)
        all_cls_test_acc.append(cls_test_acc)
        all_cls_val_acc.append(cls_val_acc)
