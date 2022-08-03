import torch
import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob

class branch_dataset(Dataset):
    def __init__(self, mode, angle):
        super(branch_dataset, self).__init__()

        self.map = json.load(open(os.path.join('..','data', 'label.json'), 'r', encoding='UTF-8'))

        self.samples = []
        sample_files = glob(os.path.join('..', 'data', mode, '*', '*'))
        for sample_file in tqdm(sample_files):
            if self.map.get(sample_file.split(os.sep)[-2].split('-')[0]) is not None:
                img_file = glob(os.path.join(sample_file, '*.jpg'))
                img_file.extend(glob(os.path.join(sample_file, '*.JPG')))
                if len(img_file) == 3:
                    img_file = img_file[angle]
                img = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),-1)
                img = cv2.resize(img, (224, 224))
                img = torch.Tensor(img)
                img = np.transpose(img, (2, 0, 1))
                label = self.map.get(sample_file.split(os.sep)[-2].split('-')[0])[2]
                label_numpy = np.array(label)
                label_tensor = torch.from_numpy(label_numpy)
                self.samples.append((img, label_tensor))
            else:
                print('not found {} species'.format(sample_file.split(os.sep)[-2].split('-')[0]))
                exit(0)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def getLabelDict(self):
        return self.map

class multi_dataset(Dataset):
    def __init__(self, mode):
        super(multi_dataset, self).__init__()

        self.map = json.load(open(os.path.join('..', 'data', 'label.json'), 'r', encoding='UTF-8'))

        self.samples = []
        sample_files = glob(os.path.join('..', 'data', mode, '*', '*'))
        for sample_file in tqdm(sample_files):
            if self.map.get(sample_file.split(os.sep)[-2].split('-')[0]) is not None:
                img_files = glob(os.path.join(sample_file, '*.jpg'))
                img_files.extend(glob(os.path.join(sample_file, '*.JPG')))
                combineImg = torch.zeros(0, 3, 224, 224)
                for img_file in img_files:
                    img = cv2.imdecode(np.fromfile(
                        img_file, dtype=np.uint8), -1)
                    img = cv2.resize(img, (224, 224))
                    img = torch.Tensor(img)
                    img = np.transpose(img, (2, 0, 1))
                    combineImg = torch.cat(
                        (combineImg, img.unsqueeze(0)), dim=0)
                label = self.map.get(sample_file.split(os.sep)[-2].split('-')[0])[2]
                if type(label) == list:
                    label = torch.Tensor(label)
                self.samples.append((combineImg, label))
            else:
                print('not found {} species'.format(sample_file.split(os.sep)[-2].split('-')[0]))
                exit(0)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def getLabelDict(self):
        return self.map