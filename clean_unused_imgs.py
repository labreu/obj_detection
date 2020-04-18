import glob
import os
base_folder = 'images'
unused_folder = 'unused_imgs'
path_unused = os.path.join(base_folder, unused_folder)

os.makedirs(path_unused, exist_ok=True)
imgs = (glob.glob(os.path.join(base_folder, '*jpg')) + 
        glob.glob(os.path.join(base_folder, '.jpeg')) +
        glob.glob(os.path.join(base_folder, '*png')))

c = 0
for img in imgs:
    xml_path = img.split('.')[0] + '.xml'
    if not os.path.exists(xml_path):
        #print('from: ', img)
        #print('to: ', os.path.join(path_unused, os.path.basename(img)))
        os.rename(img, os.path.join(path_unused, os.path.basename(img)))
        c += 1
print('Total de imagens movidas: ', c)
