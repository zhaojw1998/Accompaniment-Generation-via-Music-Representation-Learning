import os
from tqdm import tqdm

files = []
root = 'E:\\zhaojw\\data_root\\MuseScoreNYUPort\\MuseScore'
for group in os.listdir(root):
    if '.git' in group:
        continue
    print(group)
    group_dir = os.path.join(root, group)
    for file in os.listdir(group_dir):
        file_dir = os.path.join(group_dir, file)
        files.append(file_dir+'\n')

with open('file_list.txt', 'w') as f:
    f.writelines(files)
