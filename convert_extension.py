import os
import cv2
import glob
import argparse

parser = argparse.ArgumentParser(description='Converte entre extensoes')
parser.add_argument('--folder', type=str)
parser.add_argument('--ext', default='.jpg', type=str)

args = parser.parse_args()
dirname = os.path.dirname(args.folder)
to_ext = args.ext

os.makedirs(dirname+'_converted', exist_ok=True)
path = os.path.join(dirname, '*')

files = glob.glob(path)

for f in files:
    img = cv2.imread(f)
    ext = os.path.splitext(f)[1]
    path2 = os.path.join(dirname+'_converted', os.path.basename(f).replace(ext, to_ext))
    cv2.imwrite(path2, img)

