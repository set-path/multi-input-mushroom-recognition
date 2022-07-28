import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


class DataSet(Dataset):
    def __init__(self, path):
        super(DataSet, self).__init__()

        self.data = []
        self.label_dict = {'WDF108':0,'WDF110':1,'WDF112':2,'WDF114':3,'WDF115':4,'WDF116':5,'WDF117':6,'WDF120':7,'WDF122':8,'WDF123':9,'WDF124':10,'WDF127':11,'WDF128':12,'WDF129':13,'WJ1124':14,'WJ1125':15,'WJ1126':16,'WJ1127':17,'WJ1128':18,'WJ1129':19,'WJ1130':20,'WJ1131':21,'WJ1132':22,'WJ1133':23,'WJ1134':24,'WJ1135':25,'WJ1136':26,'WJ1137':27,'WJ1139':28,'WJ1140':29,'WJ1141':30}
        self.labels = []
        entries = os.listdir(path)
        num = len(entries)
        i = 0
        for entry in entries:
            if os.path.isdir(path+entry):
                files = os.listdir(path+entry)
                for file in files:
                    if file.endswith('0.jpg') and not file.startswith('.'):
                        print(str(i)+'/'+str(num))
                        i += 1
                        img = cv2.imread(path+entry+'/'+file)
                        img = cv2.resize(img, (224, 224))
                        img = torch.Tensor(img)
                        img = np.transpose(img, (2, 0, 1))
                        self.data.append(img)
                        for key in self.label_dict.keys():
                            if file.startswith(key):
                                self.labels.append(self.label_dict[key])
                        break
                            # if entry.name.startswith(key) or entry.name.startswith('v'+key) or entry.name.startswith('t'+key):
                            #     self.labels.append(self.label_dict[key])

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def getLabelDict(self):
        return self.label_dict