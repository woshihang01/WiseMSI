import os
import h5py
import openslide
import torch
from tqdm import tqdm
import numpy as np
import torchvision
import logging
import torch.nn.functional as F
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from wsi_core.wsi_utils import save_hdf5
from utils.utils import set_log, apply_stain_norm, get_stain_normalizer, device
from models.model_cnn import build_model
from wsi_core.wsi_utils import StitchCoords

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
normalizer = get_stain_normalizer()


def tumor_filter_save_hdf5(tumor_detect_model, wsi, coords, patch_level, patch_size, normalize, save_path_hdf5,
                           attr_dict=None, batch_size=200, ratio=0.50, extract_feature_model=None):
    mode = 'w'
    per_batch_img, per_batch_xy, coordinates = [], [], []
    asset_list = coords.tolist()
    index = 0
    tumor_detect_model.eval()
    if extract_feature_model:
        extract_feature_model.eval()
    re = (wsi.level_dimensions[0][0] // patch_size, wsi.level_dimensions[0][1] // patch_size)
    tumor_heatmap = np.zeros(re, dtype='float32')
    with torch.no_grad():
        for idx in tqdm(range(len(asset_list))):
            img = wsi.read_region(asset_list[idx], patch_level, (patch_size, patch_size)).convert('RGB').resize(
                (256, 256), resample=0)
            if normalize:
                try:
                    img = apply_stain_norm(img, normalizer)
                except Exception:
                    continue
            per_batch_img.append(img)
            per_batch_xy.append(asset_list[idx])
            index += 1
            if index >= batch_size or idx == len(asset_list) - 1:
                per_batch = torch.stack(
                    [image_transform(m).type(torch.FloatTensor) for m in per_batch_img]).to(
                    device, non_blocking=True)
                logits = tumor_detect_model(per_batch).to(torch.device('cpu'))
                Y_prob = F.softmax(logits, dim=1)
                score_index = (np.array(per_batch_xy)//patch_size-1).transpose(1, 0).tolist()
                tumor_heatmap[score_index] = Y_prob[:, 1]
                labels = Y_prob[:, 1].ge(ratio).numpy()
                if any(labels):
                    asset_dict = {'coords': np.array(per_batch_xy)[labels]}
                    if extract_feature_model:
                        features = extract_feature_model(per_batch[labels])
                        features = features.cpu().numpy()
                        asset_dict['features'] = features
                    if mode == 'w':
                        save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                        mode = 'a'
                    elif mode == 'a':
                        save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='a')
                index = 0
                per_batch_img = []
                per_batch_xy = []
        np.save(save_path_hdf5.replace('tumor_patches', 'tumor_scores').replace('.h5', '.npy'), tumor_heatmap)
        sns.heatmap(data=tumor_heatmap.transpose(), square=True, vmin=0, vmax=1, cbar=False)
        plt.axis('off')
        plt.gcf().set_size_inches(tumor_heatmap.shape[0] / 100, tumor_heatmap.shape[1] / 100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save_path_hdf5.replace('tumor_patches', 'tumor_heatmaps').replace('.h5', '.png'))
        plt.close()


if __name__ == '__main__':
    set_log('./logs/tumor_detect_from_tissue.log')
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--wsi_dir', type=str,
                        help='path to folder containing raw wsi image files')
    parser.add_argument('--tissue_file_dir', type=str, help='path to folder tissue h5 files')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--stitch', default=False, action='store_true')
    parser.add_argument('--model_path', help='path to model file')
    args = parser.parse_args()
    h5_files = os.listdir(args.tissue_file_dir)
    tumor_patches_dir = os.path.join(args.save_dir, 'tumor_patches')
    tumor_stitches_dir = os.path.join(args.save_dir, 'tumor_stitches')
    tumor_scores_dir = os.path.join(args.save_dir, 'tumor_scores')
    tumor_heatmaps_dir = os.path.join(args.save_dir, 'tumor_heatmaps')
    if not os.path.isdir(tumor_patches_dir):
        os.mkdir(tumor_patches_dir)
    if not os.path.isdir(tumor_stitches_dir):
        os.mkdir(tumor_stitches_dir)
    if not os.path.isdir(tumor_scores_dir):
        os.mkdir(tumor_scores_dir)
    if not os.path.isdir(tumor_heatmaps_dir):
        os.mkdir(tumor_heatmaps_dir)
    print('Loading tumor detect model...', end=' ')
    tumor_detect_model = build_model(2)
    tumor_detect_model.load_state_dict(torch.load(args.model_path))
    tumor_detect_model = tumor_detect_model.to(device)
    print('Done')
    for h5file in h5_files:
        wsi_name_svs = os.path.join(args.wsi_dir, h5file.replace('.h5', '.svs'))
        wsi_name_tif = os.path.join(args.wsi_dir, h5file.replace('.h5', '.tif'))
        if os.path.isfile(wsi_name_tif):
            wsi = openslide.OpenSlide(wsi_name_tif)
        elif os.path.isfile(wsi_name_svs):
            wsi = openslide.OpenSlide(wsi_name_svs)
        else:
            logging.info('{} not exist in wsi location, skipped.'.format(h5file.replace('.h5', '')))
            continue
        save_path_hdf5 = os.path.join(tumor_patches_dir, h5file)
        stitch_path = os.path.join(tumor_stitches_dir, h5file.replace('.h5', '.png'))
        if os.path.isfile(save_path_hdf5):
            logging.info('{} already exist in destination location, skipped'.format(h5file.replace('.h5', '')))
            if not os.path.isfile(stitch_path):
                if args.stitch:
                    heatmap = StitchCoords(save_path_hdf5, wsi, downscale=64, bg_color=(0, 0, 0), alpha=-1,
                                           draw_grid=False)
                    heatmap.save(stitch_path)
            continue
        tissue = h5py.File(os.path.join(args.tissue_file_dir, h5file), 'r')
        coords = tissue['coords'][()]
        patch_level = tissue['coords'].attrs['patch_level']
        patch_size = tissue['coords'].attrs['patch_size']
        tumor_filter_save_hdf5(tumor_detect_model, wsi, coords, patch_level, patch_size, args.normalize, save_path_hdf5,
                               attr_dict={'coords': dict(tissue['coords'].attrs)}, batch_size=200, ratio=args.ratio,
                               extract_feature_model=None)
        if args.stitch:
            if not os.path.isfile(save_path_hdf5):
                logging.info('tumor area not exist in {}, skipped.'.format(h5file.replace('.h5', '')))
                continue
            heatmap = StitchCoords(save_path_hdf5, wsi, downscale=64, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
            heatmap.save(stitch_path)
