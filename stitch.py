import openslide
from wsi_core.wsi_utils import StitchCoords
from wsi_core.WholeSlideImage import WholeSlideImage
import argparse
import os


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--patches_dir', type=str, default=r'C:\RESULTS_TUMOR_NORM\patches')
parser.add_argument('--wsi_dir', type=str, default=r'D:\train')
parser.add_argument('--file', type=str, default='0-166-bi16-29414.tif')
args = parser.parse_args()
wsi_object = WholeSlideImage(args.wsi_dir+'/'+args.file)
patch_name = os.path.splitext(args.file)[0]+'.h5'
stitch = StitchCoords(args.patches_dir+'/'+patch_name, wsi_object, downscale=64, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
stitch_name = os.path.splitext(args.file)[0]+'.png'
stitch.save('C:/RESULTS_TUMOR_NORM/stitches/{}'.format(stitch_name))