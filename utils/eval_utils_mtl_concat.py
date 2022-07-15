from models.model_toad import TOAD_fc_mtl_concat
from models.model_mil import MIL_fc
from models.model_attmil import GatedAttention
from models.model_rnn import rnn_classify

import pandas as pd
from utils.utils import *
from utils.core_utils_mtl_concat import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path=None):
    print('Init Model')    
    if args.model_type == 'toad' or args.model_type == 'toad_cosine':
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'model_type': args.model_type}
        model = TOAD_fc_mtl_concat(**model_dict)
        model.relocate()
        print('Done!')
    elif args.model_type == 'mil':
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
        model = MIL_fc(**model_dict)
        model.relocate()
        print('Done!')
    elif args.model_type == 'attmil':
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
        model = GatedAttention(**model_dict)
        model.relocate()
        print('Done!')
    elif args.model_type == 'rnn':
        model = rnn_classify().to(device)
        print('Done!')
    else:
        raise Exception('model is error')

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    results_dict = summary(model, loader, 2)

    print('cls_test_error:{:.4f}'.format(results_dict['cls_test_error']))
    print('cls_auc:{:.4f}'.format(results_dict['cls_auc']))

    return model, results_dict

# Code taken from pytorch/examples for evaluating topk classification on on ImageNet
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    cls_test_error = 0.
    cls_test_loss = 0.

    all_cls_probs = np.zeros((len(loader), n_classes))
    all_cls_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = []

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            results_dict = model(data)

        logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']

        cls_logger.log(Y_hat, label)
        cls_probs = Y_prob.cpu().numpy()
        all_cls_probs[batch_idx] = cls_probs
        all_cls_labels[batch_idx] = label.item()

        patient_results.append({'slide_id': np.array(slide_id), '0_prob': cls_probs[0][0], '1_prob': cls_probs[0][1],
                                'cls_label': label.item()})
        cls_error = calculate_error(Y_hat, label)
        cls_test_error += cls_error

    cls_test_error /= len(loader)

    if n_classes == 2:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs[:, 1])

    else:
        cls_auc = roc_auc_score(all_cls_labels, all_cls_probs, multi_class='ovr')

    for i in range(n_classes):
        acc, correct, count = cls_logger.get_summary(i)
        print('class {}: tpr {:.4f}, correct {}/{}'.format(i, acc, correct, count))
    # return patient_results, cls_test_error, cls_auc, cls_logger
    inference_results = {'patient_results': patient_results, 'cls_test_error': cls_test_error,
                         'cls_auc': cls_auc}

    return inference_results

