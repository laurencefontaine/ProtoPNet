import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class CroppedDataset(Dataset):
    def __init__(self, file_path, augmentation=False, sub_data='train'):
        self.augmentation = augmentation
        self.sub_data = sub_data
        self.data = []
        with open(file_path, 'r') as f:
            data_dict = json.load(f)
            
        for image_path, info_dict in data_dict['test'].items():
            label = info_dict['classe']
            glob_idx = info_dict['index']
            crop_dims = [float(dim) for dim in info_dict['crop']]
            self.data.append((image_path, label, glob_idx, crop_dims))
    
    def __get_item__(self, index):
        img_path, label, glob_idx, crop_dims = self.data[index]
        image = Image.open(os.path.join('./CUB_200_2011/images'))
        


        return 
        
            
    
        