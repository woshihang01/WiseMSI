from utils.core_utils import validate
from models.model_cnn import build_model
import argparse, torch, os
from torch.utils.data import DataLoader
from torch import nn
from datasets.dataset_cnn import WSI_Patches_Image
import logging
from utils.utils import set_log

set_log('./tumorDetectorVal.log')
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--patch_image_path', default='C:/test2_patch_staintools', type=str, help='dataset wsi path')
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--results_dir', type=str, default='results/staintools_norm_all_s1')
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()
model = build_model(args.n_classes)
print('\nInit Loaders...', end=' ')
datalist = os.listdir(args.patch_image_path)
dataset = WSI_Patches_Image(datalist, args.patch_image_path, maximum=200)
loader = DataLoader(dataset, batch_size=args.batch_size)
print('Done!')
print('\nInit loss function ...', end=' ')
loss_fn = nn.CrossEntropyLoss()
print('Done!')
for epoch in range(17):
    print('Loading model...', end=' ')
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "epoch_{}_s_{}_checkpoint.pt".format(epoch, 0))))
    print('Done')
    validate(cur=0, epoch=epoch, model=model, loader=loader, n_classes=args.n_classes, early_stopping=None, writer=None,
             loss_fn=loss_fn, results_dir=None)
