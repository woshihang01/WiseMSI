import argparse
import os
import numpy as np
import openslide
import PIL.Image as Image
from skimage.filters import threshold_otsu
import seaborn as sns
from skimage.color import rgb2hsv
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch import nn
import matplotlib.pyplot as plt
import math
import staintools
from tqdm import tqdm
from utils.utils import set_log
import logging

image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_stain_normalizer(path='stainColorNormalization/template.png', method='macenko'):
    target = staintools.read_image(path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(target)
    return normalizer


def apply_stain_norm(tile, normalizer):
    to_transform = np.array(tile).astype('uint8')
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed)
    return transformed


normalizer = get_stain_normalizer()


def tissue_masker(slide, power):
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    if len(slide.level_dimensions) < power + 1:
        re = (slide.level_dimensions[0][0] // (2 ** power), slide.level_dimensions[0][1] // (2 ** power))
        img_RGB = np.transpose(np.array(
            slide.read_region((0, 0), len(slide.level_dimensions) - 1, slide.level_dimensions[-1]).convert(
                'RGB').resize(re, Image.LANCZOS)),
            axes=[1, 0, 2])
    else:
        img_RGB = np.transpose(np.array(
            slide.read_region((0, 0), power, slide.level_dimensions[power]).convert('RGB')), axes=[1, 0, 2])

    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > 50
    min_G = img_RGB[:, :, 1] > 50
    min_B = img_RGB[:, :, 2] > 50

    tissue = tissue_RGB & min_R & min_G & min_B & tissue_S
    return tissue


def build_model(n_classes):
    model = models.resnet18(pretrained=True)
    fc_in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_in_features, n_classes),
    )
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description='Get tissue mask of WSI and save it in npy format')
    parser.add_argument('--wsi_path', type=str, help='Location of the WSI file in tif format')
    parser.add_argument('--patch_path', type=str, help='Path to the output directory of tumor patch images',
                        default='C:/tumor_detect_patch')
    parser.add_argument('--model_folder', type=str, default='results/tumor_classifier_2021_10_20_s1',
                        help='Model location')
    parser.add_argument('--model_name', type=str, default='s_6_checkpoint.pt')
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    model = build_model(2)
    if not os.path.exists(args.wsi_path):
        raise Exception('Please check your wsi_path.')

    # if not os.path.exists(args.patch_path):
    #     os.mkdir(args.patch_path)
    #     logging.info('mkdir {} success.'.format(args.patch_path))
    model_path = os.path.join(args.model_folder, args.model_name)
    if os.path.isfile(model_path):
        print('Start loading model.')
        model.load_state_dict(torch.load(model_path))
        print('Model loaded successfully.')
    else:
        raise Exception('No model to load.')
    wsi_files = os.listdir(args.wsi_path)
    for basename in wsi_files:
        if basename.replace('.svs', '.npy') in os.listdir('C:/tumor_detect_npy'):
            continue
        wsi_file = os.path.join(args.wsi_path, basename)
        try:
            slide = openslide.OpenSlide(wsi_file)
            if 'aperio.AppMag' in slide.properties:
                if slide.properties.get('aperio.AppMag') == '40':
                    power = 9
                elif slide.properties.get('aperio.AppMag') == '20':
                    power = 8
                else:
                    raise Exception('slide用的不是40或20倍镜')
            else:
                raise Exception('不知道slide用的几倍镜')
        except:
            logging.info('{} is error,continue.'.format(wsi_file))
            continue
        tissue_mask = tissue_masker(slide, power)
        save_path = os.path.join(args.patch_path, basename)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        patch_images, _, tissue_mask_heatmap, zuobiao = Predict_tumor(model, device, slide, tissue_mask, wsi_file, 0.5,
                                                                      args.batch_size, save_path, power)
        np.save('C:/tumor_detect_npy/' + basename.replace('.svs', '.npy'), tissue_mask_heatmap)
        sns.heatmap(data=tissue_mask_heatmap.transpose(), square=True, vmin=0, vmax=1)
        plt.savefig('C:/tumor_detect_heatmap/' + basename.replace('.svs', '.png'))
        plt.close()


def Predict_tumor(model, device, slide, tissue_mask, file_name, ratio, batch_size, save_path, power):
    new_high = tissue_mask.shape[0]
    new_wight = tissue_mask.shape[1]
    tissue_mask_heatmap = np.zeros((new_high, new_wight), dtype='float32')
    patch_images, per_batch_img, per_batch_img2, per_batch_xy, heatmap_coordinate = [], [], [], [], []
    normalized_patch_images = torch.arange(0).to(torch.device('cpu'))
    logging.info('Start predict tumor {}.'.format(file_name))
    global_num = 0
    model.eval()
    with torch.no_grad():
        for x in tqdm(range(new_high)):
            for y in range(new_wight):
                if tissue_mask[x][y]:
                    x_upperLeftCorner = x * (2 ** power)
                    y_upperLeftCorner = y * (2 ** power)
                    img_2 = img = slide.read_region(
                        (int(x_upperLeftCorner), int(y_upperLeftCorner)), 0,
                        ((2 ** power), (2 ** power))).convert('RGB')
                    if power == 9:
                        img_2 = img = img.resize((256, 256), Image.LANCZOS)
                    try:
                        img = apply_stain_norm(img, normalizer)
                    except Exception as e:
                        continue
                    per_batch_img.append(img)
                    per_batch_img2.append(img_2)
                    per_batch_xy.append((x, y))
                if len(per_batch_img) >= batch_size or (x == new_high - 1 and y == new_wight - 1):
                    if x == new_high - 1 and y == new_wight - 1:
                        if len(per_batch_img) == 0:
                            continue
                    per_batch = torch.stack([image_transform(m).type(torch.FloatTensor) for m in per_batch_img])
                    logits = model(per_batch.to(device)).to(torch.device('cpu'))
                    Y_prob = F.softmax(logits, dim=1)
                    labels = Y_prob[:, 1].ge(ratio).numpy()
                    # normalized_patch_images = torch.cat(
                    #     (normalized_patch_images, per_batch[labels]), 0)
                    for batch_size_num, label in zip(range(per_batch.shape[0]), labels):
                        i_, j_ = per_batch_xy[batch_size_num]
                        # if label:
                        # heatmap_coordinate.append([i_, j_])
                        # patch_image_name = ''.join(os.path.basename(file_name).split('.')[:-1]) + '-' + str(
                        #     global_num) + '-' + str(
                        #     i_ * (2 ** power)) + '-' + str(
                        #     j_ * (2 ** power)) + '.png'
                        # patch_images.append(per_batch_img[batch_size_num])
                        # per_batch_img2[batch_size_num].save(os.path.join(save_path, patch_image_name))
                        # global_num += 1
                        tissue_mask_heatmap[i_][j_] = Y_prob[batch_size_num, 1]
                    per_batch_img = []
                    per_batch_img2 = []
                    per_batch_xy = []
    return patch_images, normalized_patch_images, tissue_mask_heatmap, heatmap_coordinate


def hanshu(a, bianchang):
    # 每到一个坐标点记录到这个坐标点最好的一条路径
    a = [tuple(item) for item in a]
    # start = time.time()
    array = np.array(a)
    M = array[:, 0]
    N = array[:, 1]
    donggui_dict = {}
    max_n = max(N)
    min_n = min(N)
    max_m = max(M)
    min_m = min(M)
    for m in range(min_m, max_m + 1):
        for n in range(min_n, max_n + 1):
            yaobuyaotiao = math.floor(pow(bianchang, 2) * 0.15)
            for x in range(bianchang):
                for y in range(bianchang):
                    if (m + x, n + y) not in a:
                        yaobuyaotiao -= 1
                        if yaobuyaotiao < 0:
                            break
                if yaobuyaotiao < 0:
                    break
            if yaobuyaotiao < 0:
                continue
            if len(donggui_dict) == 0:
                donggui_dict = {(m, n): ((m, n),)}
            else:
                donggui_dict_new = donggui_dict.copy()
                max_len = 0
                for key in donggui_dict:
                    key_len = len(donggui_dict[key])
                    jianji = set([])
                    for x in range(m - bianchang + 1, m + bianchang):
                        for y in range(n - bianchang + 1, n + bianchang):
                            if (x, y) in donggui_dict[key]:
                                jianji.add((x, y))
                                key_len -= 1
                    xinkeneng = list(set(donggui_dict[key]).difference(jianji))
                    if key_len + 1 > max_len:
                        xinkeneng.append((m, n))
                        donggui_dict_new[(m, n)] = tuple(xinkeneng)
                        max_len = key_len + 1
                donggui_dict = donggui_dict_new
    if len(donggui_dict) == 0:
        return []
    dict_max_len = 0
    for item in donggui_dict:
        if dict_max_len < len(donggui_dict[item]):
            dict_max_len = len(donggui_dict[item])
            key = item
    result = list(donggui_dict[key])
    # print(dict_max_len)
    return result


if __name__ == '__main__':
    set_log('./logs/tumorHeatmap.log')
    main()
