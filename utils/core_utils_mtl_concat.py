import numpy as np
import torch
import pickle
import logging, time
from utils.utils import *
import os
from datasets.dataset_mtl_concat import save_splits
from sklearn.metrics import roc_auc_score
from models.model_toad import TOAD_fc_mtl_concat
from models.model_rnn import rnn_classify
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct

    def log_batch_rnn(self,Y_hat, Y):
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

    def __init__(self, patience=40, stop_epoch=100, verbose=False):  # 连续patience轮，并且总论此超过stop_epoch轮就会终止
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
        self.val_loss_max = np.Inf

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

    def save_checkpoint(self, early_stopping, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(
                f'Validation loss decreased ({self.val_loss_max:.6f} --> {early_stopping:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_max = early_stopping


def train(datasets, cur, args):
    """   
        train for a single fold
    """
    logging.info('Training Fold {}!'.format(cur))
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
    print("Testing on {} samples".format(len(test_split)))
    loss_fn = nn.CrossEntropyLoss().to(device)

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'model_type': args.model_type}
    if model_dict['model_type'] == 'toad' or model_dict['model_type'] == 'toad_cosine':
        model = TOAD_fc_mtl_concat(**model_dict)
        model.relocate()
        print('Done!')
    elif model_dict['model_type'] == 'rnn':
        model = rnn_classify().to(device)
        print('Done!')
    else:
        raise Exception('model is error')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    if model_dict['model_type'] == 'toad' or model_dict['model_type'] == 'toad_cosine':
        train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
        val_loader = get_split_loader(val_split)
        test_loader = get_split_loader(test_split)
    elif model_dict['model_type'] == 'rnn':
        train_loader = DataLoader(train_split, 64, shuffle=True)
        val_loader = DataLoader(val_split, 64, shuffle=False)
        test_loader = DataLoader(test_split, 64, shuffle=False)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=40, stop_epoch=100, verbose=True)  # 连续patience轮，并且总论此超过stop_epoch轮就会终止

    else:
        early_stopping = None
    print('Done!')
    if model_dict['model_type'] == 'toad' or model_dict['model_type'] == 'toad_cosine':
        for epoch in range(args.max_epochs):
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir)

            if stop:
                break
    elif model_dict['model_type'] == 'rnn':
        for epoch in range(args.max_epochs):
            train_loop_rnn(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate_rnn(cur, epoch, model, val_loader, args.n_classes,
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
        logging.info('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_tpr'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/cls_val_error', cls_val_error, 0)
        writer.add_scalar('final/cls_val_auc', cls_val_auc, 0)
        writer.add_scalar('final/cls_test_error', cls_test_error, 0)
        writer.add_scalar('final/cls_test_auc', cls_test_auc, 0)

    writer.close()
    return results_dict, cls_test_auc, cls_val_auc, 1 - cls_test_error, 1 - cls_val_error


def train_loop(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    cls_train_error = 0.
    cls_train_loss = 0.
    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)

        results_dict = model(data)
        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        cls_logger.log(Y_hat, label)

        cls_loss = loss_fn(logits, label)
        cls_loss_value = cls_loss.item()

        cls_train_loss += cls_loss_value
        if (batch_idx + 1) % 50 == 0:
            logging.info(
                'batch {}, cls loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, cls_loss_value, label.item(),
                                                                             data.size(0)))

        cls_error = calculate_error(Y_hat, label)
        cls_train_error += cls_error

        # backward pass
        cls_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    cls_train_loss /= len(loader)
    cls_train_error /= len(loader)

    logging.info(
        'Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}'.format(epoch, cls_train_loss, cls_train_error))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        logging.info('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_tpr'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/cls_loss', cls_train_loss, epoch)
        writer.add_scalar('train/cls_error', cls_train_error, epoch)


def train_loop_rnn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    model.train()
    logger = Accuracy_Logger(n_classes=n_classes)
    train_error = 0.
    train_loss = 0.
    for batch_idx, (feature, label_batch) in enumerate(loader):
        # order_idx = np.argsort(feature_len.numpy())[::-1]
        # label_batch = label_batch[order_idx.tolist()].long()

        results_dict = model(feature)
        logits_batch, Y_prob_batch, Y_hat_batch = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        Y_hat_batch = Y_hat_batch.squeeze(-1)
        logger.log_batch_rnn(Y_hat_batch.cpu(), label_batch.cpu())
        acc_num = torch.eq(Y_hat_batch, label_batch).sum().float().item()
        loss = loss_fn(logits_batch, label_batch.to(torch.long))
        loss_value = loss.item()

        train_loss += loss_value
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info('time {}, batch {}, loss: {:.4f}, acc: {:.4f}'.format(localtime, batch_idx, loss_value,
                                                                           acc_num / feature.shape[0]))
        error = calculate_error(Y_hat_batch, label_batch)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    logging.info(
        'Epoch: {}, cls train_loss: {:.4f}, cls train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = logger.get_summary(i)
        logging.info('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_tpr'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/cls_loss', train_loss, epoch)
        writer.add_scalar('train/cls_error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            del results_dict

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

    if n_classes == 2:
        cls_auc = roc_auc_score(cls_labels, cls_probs[:, 1])
        cls_aucs = []
    else:
        cls_aucs = []
        binary_labels = label_binarize(cls_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in cls_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], cls_probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        cls_auc = np.nanmean(np.array(cls_aucs))

    if writer:
        writer.add_scalar('val/cls_loss', cls_val_loss, epoch)
        writer.add_scalar('val/cls_auc', cls_auc, epoch)
        writer.add_scalar('val/cls_error', cls_val_error, epoch)

    logging.info(
        '\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(cls_val_loss, cls_val_error,
                                                                                         cls_auc))
    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        logging.info('class {}: tpr {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_tpr'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, cls_val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            logging.info("Early stopping")
            return True

    return False


def validate_rnn(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    model.eval()
    logger = Accuracy_Logger(n_classes=n_classes)
    val_error = 0.
    val_loss = 0.

    probs = np.zeros((len(loader.dataset), n_classes))
    labels = np.zeros(len(loader.dataset))

    with torch.no_grad():
        for batch_idx, (feature, label_batch) in enumerate(loader):
            results_dict = model(feature)
            logits_batch, Y_prob_batch, Y_hat_batch = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

            Y_hat_batch = Y_hat_batch.squeeze(-1)
            logger.log_batch_rnn(Y_hat_batch.cpu(), label_batch.cpu())
            acc_num = torch.eq(Y_hat_batch, label_batch).sum().float().item()
            loss = loss_fn(logits_batch, label_batch.to(torch.long))
            loss_value = loss.item()

            val_loss += loss_value
            # localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # logging.info('time {}, batch {}, loss: {:.4f}, acc: {:.4f}'.format(localtime, batch_idx, loss_value,
            #                                                                    acc_num / feature.shape[0]))
            error = calculate_error(Y_hat_batch, label_batch)
            val_error += error
            probs[batch_idx * loader.batch_size:batch_idx * loader.batch_size + feature.shape[0]] = Y_prob_batch.cpu().numpy()
            labels[
            batch_idx * loader.batch_size:batch_idx * loader.batch_size + feature.shape[0]] = label_batch.cpu().numpy()

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        cls_aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs[:, class_idx])
                cls_aucs.append(calc_auc(fpr, tpr))
            else:
                cls_aucs.append(float('nan'))

        auc = np.nanmean(np.array(cls_aucs))

    if writer:
        writer.add_scalar('val/cls_loss', val_loss, epoch)
        writer.add_scalar('val/cls_auc', auc, epoch)
        writer.add_scalar('val/cls_error', val_error, epoch)

    logging.info(
        '\nVal Set, cls val_loss: {:.4f}, cls val_error: {:.4f}, cls auc: {:.4f}'.format(val_loss, val_error,
                                                                                         auc))
    for i in range(n_classes):
        acc, correct, count = logger.get_summary(i)
        logging.info('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_tpr'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model,
                       ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            logging.info("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.

    all_cls_probs = np.zeros((len(loader), n_classes))
    all_cls_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data)

        logits, Y_prob, Y_hat, A = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat'], results_dict[
            'A']
        del results_dict

        cls_logger.log(Y_hat, label)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        patient_results.update(
            {slide_id: {'slide_id': np.array(slide_id), 'cls_prob': cls_probs, 'cls_label': label.item(), 'A': A}})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error

    cls_test_error /= len(loader)

    if n_classes == 2:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])

    else:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs, multi_class='ovr')

    return patient_results, cls_test_error, cls_auc, cls_logger
