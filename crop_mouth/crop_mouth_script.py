import os
from tqdm import tqdm
import pickle

dir_path = ""
txt_file_path = ""
mp4_files = []
if dir_path != "":
    for root, dirs, files in tqdm(os.walk(dir_path)):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))

if txt_file_path != "":
    with open(txt_file_path, "rb") as fp:   # Unpickling
        mp4_files = pickle.load(fp)


print(len(mp4_files))

for i in tqdm(mp4_files):
    src_path = i
    dest_path = i[:-4] + '_crop.mp4'
    print(src_path, dest_path)
    os.system(f'python crop_mouth.py data_filename={src_path} dst_filename={dest_path}')
