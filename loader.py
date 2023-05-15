import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# import torch.utils.data.distributed
import torchvision.transforms as T
import torchvision.datasets as datasets


class CroppedDataset(Dataset):
    def __init__(self, json_path, augmentation=False, sub_data='train'):
        self.augmentation = augmentation
        self.sub_data = sub_data
        self.data = []
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
            
        for image_path, info_dict in data_dict[sub_data].items():
            label = info_dict['classe']
            glob_idx = info_dict['index']
            crop_dims = [float(dim) for dim in info_dict['crop']]
            self.data.append((image_path, label, glob_idx, crop_dims))
    
    def __getitem__(self, index):
        img_path, label, glob_idx, crop_dims = self.data[index]
        image = Image.open(os.path.join('./CUB_200_2011/images', img_path))
        x, y, width, height = crop_dims
    
        img_cropped = image.crop((int(x), int(y), int(x) + int(width), int(y) + int(height)))
        torch.manual_seed(17)
        if self.augmentation:
             transforms = torch.nn.Sequential(T.RandomHorizontalFlip(p=0.5),
                                              T.RandomAffine(degrees=15, shear=10))
             img_cropped = transforms(img_cropped)
        return img_cropped, label
    
    def __len__(self):
        return len(self.data)

class UncroppedDataset(Dataset):
    def __init__(self, json_path, augmentation=False, sub_data='train'):
        self.augmentation = augmentation
        self.sub_data = sub_data
        self.data = []
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
            
        for image_path, info_dict in data_dict[sub_data].items():
            label = info_dict['classe']
            glob_idx = info_dict['index']
            crop_dims = [float(dim) for dim in info_dict['crop']]
            self.data.append((image_path, label, glob_idx, crop_dims))
    
    def __getitem__(self, index):
        img_path, label, glob_idx, crop_dims = self.data[index]
        image = Image.open(os.path.join('./CUB_200_2011/images', img_path))
        #x, y, width, height = crop_dims
    
        #img_cropped = image.crop((int(x), int(y), int(x) + int(width), int(y) + int(height)))
        torch.manual_seed(17)
        if self.augmentation:
             transforms = torch.nn.Sequential(T.RandomHorizontalFlip(p=0.5),
                                              T.RandomAffine(degrees=15, shear=10))
             image = transforms(image)
        return image, label
    
    def __len__(self):
        return len(self.data)
    
    
# if __name__ == '__main__':
#     path = './split_info.json'
#     training_data = UncroppedDataset(path, augmentation=True)
#     #print(len(training_data))
#     x = 0
#     for image, label in training_data:
#         if x >= 4:
#             break
#         print(np.asarray(image).shape)

#         x += 1


            
    
        