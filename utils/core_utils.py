import numpy as np
import pandas as pd
import torch
from utils.utils import *
import os, time, logging
from datasets.dataset_generic import save_splits
from models.model_cnn import build_model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


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
    # save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)), end=' ')
    print("Validating on {} samples".format(len(val_split)), end=' ')
    print("Testing on {} samples".format(len(test_split)), end=' ')

    print('\nInit Model...', end=' ')
    model = build_model(args.n_classes)
    print('Done!')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    print('\nInit loss function ...', end=' ')
    loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    print('\nInit Loaders...', end=' ')
    train_loader = DataLoader(train_split, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_split, batch_size=args.batch_size)
    test_loader = DataLoader(test_split, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        # 连续patience轮次损失值不再下降且总轮次超过stop_epoch时就会终止
        early_stopping = EarlyStopping(patience=2, stop_epoch=5, verbose=True)
        # early_stopping.early_stop = True
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

    val_patch_result, val_wsi_result, val_error, val_auc, _ = summary(model, val_loader, args.n_classes)
    logging.info('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    val_patch_result.to_csv(os.path.join(args.results_dir, "s_{}_val_patch_result.csv".format(cur)))
    val_wsi_result.to_csv(os.path.join(args.results_dir, "s_{}_val_wsi_result.csv".format(cur)))

    test_patch_result, test_wsi_result, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    logging.info('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    test_patch_result.to_csv(os.path.join(args.results_dir, "s_{}_test_patch_result.csv".format(cur)))
    test_wsi_result.to_csv(os.path.join(args.results_dir, "s_{}_test_wsi_result.csv".format(cur)))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logging.info('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)

    writer.close()
    return test_auc, val_auc, 1 - test_error, 1 - val_error


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    logging.info('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits = model(data)
        Y_hat = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        logger.log_batch(Y_hat.cpu(), label.cpu())
        acc_num = torch.eq(Y_hat, label).sum().float().item()
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info('time {}, batch {}, loss: {:.4f}, acc: {:.4f}'.format(localtime, batch_idx, loss_value,
                                                                               acc_num / data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    logging.info('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = logger.get_summary(i)
        logging.info('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits = model(data)
            Y_hat = torch.topk(logits, 1, dim=1)[1].squeeze(1)
            acc_logger.log_batch(Y_hat.cpu(), label.cpu())

            loss = loss_fn(logits, label)
            Y_prob = F.softmax(logits, dim=1)
            prob[batch_idx * loader.batch_size:batch_idx * loader.batch_size + data.size()[0]] = Y_prob.cpu().numpy()
            labels[batch_idx * loader.batch_size:batch_idx * loader.batch_size + data.size()[0]] = label.cpu().numpy()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    logging.info('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        logging.info('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            logging.info("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    wsi_list = loader.dataset.wsi_list
    all_probs = np.zeros((len(loader.dataset), n_classes))
    all_preds = np.zeros(len(loader.dataset))
    all_labels = np.zeros(len(loader.dataset))

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        with torch.no_grad():
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            logits = model(data)
            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.topk(logits, 1, dim=1)[1].squeeze(1)

        acc_logger.log_batch(Y_hat.cpu(), label.cpu())
        all_probs[batch_idx * loader.batch_size:batch_idx * loader.batch_size + data.size()[0]] = Y_prob.cpu().numpy()
        all_preds[batch_idx * loader.batch_size:batch_idx * loader.batch_size + data.size()[0]] = Y_prob[:, 1].gt(
            0.5).cpu().numpy()
        all_labels[batch_idx * loader.batch_size:batch_idx * loader.batch_size + data.size()[0]] = label.cpu().numpy()

        error = calculate_error(Y_hat, label)
        test_error += error
        if (batch_idx + 1) % 50 == 0:
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info('time {}, batch {}, error: {:.4f}'.format(localtime, batch_idx, error))
    test_error /= len(loader)
    patch_result = pd.DataFrame(
        {'wsi': wsi_list, 'logit': all_probs[:, 1].tolist(), 'pred': all_preds.tolist(), 'label': all_labels.tolist()})
    wsi_result = pd.concat([patch_result.groupby('wsi')['logit'].mean(), patch_result.groupby('wsi')['label'].mean()], axis=1)
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patch_result, wsi_result, test_error, auc, acc_logger
