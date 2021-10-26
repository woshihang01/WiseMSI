import argparse, torch, os, h5py, openslide
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils.utils import print_network, get_split_loader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.model_toad import TOAD_fc_mtl_concat
from models.model_rnn import rnn_classify
from datasets.dataset_mtl_concat import Generic_MIL_MTL_Dataset
from datasets.dataset_rnn import Generic_WSI_RNN_Dataset
import logging
from utils.core_utils_mtl_concat import Accuracy_Logger


def set_log(logfileName='./logs/wsi_test.log', level=logging.INFO):
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


def validate(model, loader, n_classes, loss_fn=None):
    model.eval()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_val_error = 0.
    cls_val_loss = 0.

    cls_probs = np.zeros((len(loader), n_classes))
    cls_labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)

            results_dict = model(data)
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']
            # Y_hat = torch.tensor(1).to(device) if Y_prob.squeeze()[1] > 0.05 else torch.tensor(0).to(device)
            Y_hat = torch.topk(logits, 1, dim=1)[1].squeeze(1)
            cls_logger.log(Y_hat, label)

            cls_loss = loss_fn(logits, label)
            cls_loss_value = cls_loss.item()

            cls_probs[batch_idx] = Y_prob.cpu().numpy()
            cls_labels[batch_idx] = label.item()

            cls_val_loss += cls_loss_value
            cls_error = calculate_error(Y_hat, label)
            cls_val_error += cls_error

    cls_val_error /= len(loader)
    cls_val_loss /= len(loader)

    loader.dataset.slide_data['prob'] = cls_probs[:, 1]
    loader.dataset.slide_data.to_csv(
        os.path.join(args.results_dir, args.exp_code + "_s1", 'prob_{}.csv'.format(args.k)))
    cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
    # fpr, tpr, threshold = roc_curve(cls_labels, cls_probs[:, 1])
    # lw = 2
    # plt.figure(1, figsize=(5, 5))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw,
    #          label='AUROC:{:0.4f} (95% CI {:0.4f}-{:0.4f})'.format(0.9503, 0.9447, 0.9560))  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([-0.01, 1.01])
    # plt.ylim([-0.01, 1.01])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(args.results_dir, args.exp_code + "_s1", 'test' + str(args.k) + '.png'))

    logging.info(
        'Val Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_val_error,
                                                                                       cls_auc))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        logging.info('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))


def validate_rnn(model, loader, n_classes, loss_fn=None):
    model.eval()
    logger = Accuracy_Logger(n_classes=n_classes)
    val_error = 0.
    val_loss = 0.

    probs = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))

    with torch.no_grad():
        for batch_idx, (feature, label_batch) in enumerate(loader):
            feature = feature.to(device)
            results_dict = model(feature)
            logits_batch, Y_prob_batch, Y_hat_batch = results_dict['logits'], results_dict['Y_prob'], results_dict[
                'Y_hat']

            Y_hat_batch = Y_hat_batch.squeeze(-1).cpu()
            logger.log_batch_rnn(Y_hat_batch, label_batch)
            acc_num = torch.eq(Y_hat_batch, label_batch).sum().float().item()
            loss = loss_fn(logits_batch, label_batch.to(device).to(torch.long))
            loss_value = loss.item()
            val_loss += loss_value
            error = calculate_error(Y_hat_batch, label_batch)
            val_error += error
            probs[
            batch_idx * loader.batch_size:batch_idx * loader.batch_size + feature.shape[0]] = Y_prob_batch.cpu().numpy()
            labels[
            batch_idx * loader.batch_size:batch_idx * loader.batch_size + feature.shape[0]] = label_batch.numpy()

    val_error /= len(loader)
    val_loss /= len(loader)

    loader.dataset.slide_data['prob'] = probs[:, 1]
    loader.dataset.slide_data.to_csv(
        os.path.join(args.results_dir, args.exp_code + "_s1", 'prob_{}.csv'.format(args.k)))
    auc = roc_auc_score(labels, probs[:, 1])

    logging.info(
        '\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(val_loss, val_error,
                                                                                         auc))
    for i in range(n_classes):
        acc, correct, count = logger.get_summary(i)
        logging.info('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--n_classes', default=2)
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
    parser.add_argument('--model_type', default='rnn', choices=['toad', 'toad_cosine', 'rnn'])
    parser.add_argument('--pt_files_dir', default='D:/RESULTS_TUMOR_STAIN_NORM_95/patches', type=str,
                        help='h5file directory')
    parser.add_argument('--patches_dir', default='D:/RESULTS_TUMOR_STAIN_NORM_95/patches', type=str,
                        help='patches directory')
    parser.add_argument('--task', type=str, choices=['msi_classifier'], default='msi_classifier')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results',
                        default='msi_classifier_2021_09_26_rnn_95tumor')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = 'dataset_csv/dataset_train_test2_test3_msi_95tumor.csv'
    logging.info("csv_path is {}".format(csv_path))
    model_path = os.path.join(args.results_dir, args.exp_code + "_s1", "s_{}_checkpoint.pt".format(args.k))
    loss_fn = nn.CrossEntropyLoss()
    print('\nInit Model...', end=' ')
    if args.model_type == 'toad':
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'model_type': args.model_type}

        model = TOAD_fc_mtl_concat(**model_dict)
        model.relocate()
        print('Done!')
        dataset = Generic_MIL_MTL_Dataset(csv_path=csv_path,
                                          data_dir=args.pt_files_dir,
                                          shuffle=False,
                                          seed=args.seed,
                                          print_info=True,
                                          label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                          label_cols=['label'],
                                          patient_strat=False)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path=os.path.join(args.results_dir,
                                                                                               args.exp_code + "_s1",
                                                                                               'splits_{}.csv'.format(
                                                                                                   args.k)))
        test_loader = get_split_loader(dataset)

        logging.info("model load_state {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        validate(model, test_loader, 2, loss_fn)
        logging.info("\n")
    elif args.model_type == 'rnn':
        model = rnn_classify().to(device)
        print('Done!')
        dataset = Generic_WSI_RNN_Dataset(csv_path=csv_path,
                                          data_dir=args.pt_files_dir, n=100,
                                          print_info=True,
                                          label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                          label_cols=['label'])
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
                                                                         csv_path=os.path.join(args.results_dir,
                                                                                               args.exp_code + "_s1",
                                                                                               'splits_{}.csv'.format(
                                                                                                   args.k)))
        test_loader = DataLoader(test_dataset, 64, shuffle=False)
        logging.info("model load_state {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        validate_rnn(model, test_loader, 2, loss_fn)
        logging.info("\n")
    else:
        raise Exception('有问题')


