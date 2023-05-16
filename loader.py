import torch
import os
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
# import torch.utils.data.distributed
import torchvision.transforms as T
import torchvision.datasets as datasets

from settings import train_batch_size


class CroppedDataset(Dataset):
    def __init__(self, json_path, sub_data='train', norm=False, augmentation=False, resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.augmentation = augmentation
        self.sub_data = sub_data
        self.mean = mean
        self.std = std
        self.norm = norm
        self.resize = resize
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
        #print(img_path)
        image = Image.open(os.path.join('./CUB_200_2011/images', img_path))
        if image.mode != 'RGB':
            image = image.convert(mode='RGB')
        
        x, y, width, height = crop_dims
        # print("OG")
        # print(np.asarray(image).shape)
    
        img_cropped = image.crop((int(x), int(y), int(x) + int(width), int(y) + int(height)))

        # print("après crop")
        # print(np.asarray(img_cropped).shape)
        if self.augmentation:
             transforms = torch.nn.Sequential(T.RandomHorizontalFlip(p=0.5),
                                              T.RandomAffine(degrees=15, shear=10))
             img_cropped = transforms(img_cropped)
            #  print("après transforms")
            #  print(np.asarray(img_cropped).shape)
        t_tensor = T.ToTensor()
        img_cropped = t_tensor(img_cropped)
        if self.norm:
            normalize = T.Normalize(mean=self.mean, std=self.std)
            img_cropped = normalize(img_cropped)
        #print(img_cropped)
        # print("avant resize")
        # print(np.asarray(img_cropped).shape)
        resize = T.Resize(size=(self.resize, self.resize))
        img_cropped = resize(img_cropped)
        # print("après resize")
        # print(np.asarray(img_cropped).shape)
        return img_cropped, label
    
    def __len__(self):
        return len(self.data)

class UncroppedDataset(Dataset):
    def __init__(self, json_path, augmentation=False, sub_data='train', resize=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.resize = resize
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
        print(np.asarray(image))

        normalize = torch.nn.Sequential(T.Resize(size=(self.resize, self.resize)), 
                                        T.Normalize(mean=self.mean, std=self.std), 
                                        )
        t_tensor = T.ToTensor()
        image = normalize(t_tensor(image))
        return image, label
    
    def __len__(self):
        return len(self.data)
 
class ExampleDataset(Dataset):
    def __init__(self, json_path, sub_data='train', augmentation=False):
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

            
        #print(img_cropped)


        
        return img_cropped, label
    
    def __len__(self):
        return len(self.data)
   
    
# if __name__ == '__main__':
#     path = './split_info.json'

#     mean=(0.485, 0.456, 0.406)
#     std=(0.229, 0.224, 0.225)
#     resize=224
#     norm=True
    
#     training_data = CroppedDataset(path, sub_data='valid',norm=norm, augmentation=True, resize=resize, mean=mean, std=std)
#     train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_batch_size, shuffle=True,
#     num_workers=4, pin_memory=False)
#     #print(len(training_data))
#     x = 0
#     for image, label in train_loader:
#         image = image.cuda()
#         label = label.cuda()
#         print(label)
#         #print(np.asarray(image).shape)
#         #print(np.mean(np.asarray(image)))

#         x += 1


            
    
        