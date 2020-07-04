import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base', required=True, type=str)
parser.add_argument('--unused', default='unused_imgs', type=str)

args = parser.parse_args()
base_folder = args.base
unused_folder = args.unused

path_unused = os.path.join(base_folder, unused_folder)

os.makedirs(path_unused, exist_ok=True)
imgs = (glob.glob(os.path.join(base_folder, '*jpg')) + 
        glob.glob(os.path.join(base_folder, '*JPG')) + 
        glob.glob(os.path.join(base_folder, '.jpeg')) +
        glob.glob(os.path.join(base_folder, '*png')))

c = 0
for img in imgs:
    xml_path = os.path.splitext(img)[0] + '.xml'
    if not os.path.exists(xml_path):
        #print('from: ', img)
        #print('to: ', os.path.join(path_unused, os.path.basename(img)))
        os.rename(img, os.path.join(path_unused, os.path.basename(img)))
        c += 1
print('Total de imagens movidas: ', c)