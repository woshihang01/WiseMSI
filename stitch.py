import openslide
from wsi_core.wsi_utils import StitchCoords
from wsi_core.WholeSlideImage import WholeSlideImage
import argparse

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--hdf5_file_path', type=str, default='D:/RESULTS_TUMOR_STAIN_NORM_95/patches/0-5-1815353.h5')
parser.add_argument('--wsi_file_path', type=str, default='D:/train/0-5-1815353.tif')

args = parser.parse_args()

wsi_object = WholeSlideImage(args.wsi_file_path)
heatmap = StitchCoords(args.hdf5_file_path, wsi_object, downscale=64, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
heatmap.save('a.png')