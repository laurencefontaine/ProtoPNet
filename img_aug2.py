#import skimage.transform

import os
import numpy as np
import json

from sklearn import model_selection

from torch.utils.data import Dataset
from os import listdir
from PIL import Image


def augmenter(in_dir, out_dir):
    files = print(listdir(in_dir))

def load_images(path):
    f_test = open("CUB_200_2011/CUB_200_2011/train_test_split.txt", "r")
    f_path = open("CUB_200_2011/CUB_200_2011/images.txt", "r")
    f_crop = open("CUB_200_2011/CUB_200_2011/bounding_boxes.txt", "r")

    for line_test, line_path, line_crop in zip(f_test.readlines(), f_path.readlines(), f_crop.readlines()):
        test = int(line_test.replace("\n", "").split(' ')[1])
        path = line_path.replace("\n", "").split(' ')[1]
        left, top, width, height = line_crop.replace("\n", "").split(' ')[1:]

        im = Image.open("CUB_200_2011/CUB_200_2011/images/" + path)
        im = im.crop((int(float(left)), int(float(top)), int(float(left) + float(width)), int(float(top) + float(height))))
        if test:
            dest_fpath = "./datasets/cub200_whole/test_whole/" + path
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            shutil.copy("CUB_200_2011/CUB_200_2011/images/" + path, dest_fpath)

            dest_fpath = "./datasets/cub200_cropped/test_cropped/" + path
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            im.save(dest_fpath)
        else:
            dest_fpath = "./datasets/cub200_whole/train_whole/" + path
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            shutil.copy("CUB_200_2011/CUB_200_2011/images/" + path, dest_fpath)

            dest_fpath = "./datasets/cub200_cropped/train_cropped/" + path
            os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
            im.save(dest_fpath)

def img_per_classes(path):
    """Donne le nombre d'images par classe dans le directory d'images"""
    x = 0
    for c in listdir(path):
        print(x)
        print(c)
        x=0
        for i in listdir(os.path.join(datasets_root_dir, c)):
            x+=1

def train_valid_test(path):
    """outputs json file {'train': {
                            'path_to_img' : {
                                'classe' : 
                                'index' : index dans total dataset
                                'crop' : [x, y, width, height]
                            }}}"""
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    with open('./CUB_200_2011/images.txt', 'r' ) as file:
        lines = file.readlines()
        file.close()
    with open('./CUB_200_2011/bounding_boxes.txt', 'r' ) as file:
        crops = file.readlines()
        file.close()

    for i, line in enumerate(lines):
        lines[i] =  line.split('/')[-1].split('\n')[0]

    for i, crop in enumerate(crops):
        crops[i] = crop.split('\n')[0].split(' ')[1:]
    
    for n, c in enumerate(listdir(path)):
        
        img_class = listdir(os.path.join(path, c))

        #x_train, x_test_ = model_selection.train_test_split(range(1, len(img_class)+1), test_size=0.5)
        #x_valid, x_test = model_selection.train_test_split(x_test_, test_size=0.5)
        
        x_train, x_test_ = model_selection.train_test_split(img_class, test_size=0.5)
        x_valid, x_test = model_selection.train_test_split(x_test_, test_size=0.5)
        
        for train in x_train:
            idx = lines.index(train)

            train_dict['{}/{}'.format(c, train)] = {'classe' : n+1,
                                                  'index' : idx,
                                                  'crop' : crops[idx-1]}

        for valid in x_valid:
            idx = lines.index(valid)
            valid_dict['{}/{}'.format(c, valid)] = {'classe': n+1,
                                                  'index' : idx,
                                                  'crop' : crops[idx-1]}

        for test in x_test:
            idx = lines.index(test)
            test_dict['{}/{}'.format(c, test)] = {'classe': n+1,
                                                  'index' : idx,
                                                  'crop' : crops[idx-1]}

    split = {'train' : train_dict,
             'valid' : valid_dict,
             'test' : train_dict}
    json_dict = json.dumps(split, indent=4)
    
    with open("train.json", "w") as outfile:
        outfile.write(json_dict)    


def load_dataset(Dataset):
    def __init__(self, file_path, sub_data='test', data_aug=0, **kwargs):
        super(load_dataset, self).__init__()

        self.file_path = file_path
        self.data_aug = data_aug
        self.sub_data = sub_data
    
    def unpack_data(self):
        with open(self.file_path) as file:
            full_data = json.load(file)
        return full_data[self.sub_data]

    def __getitem__(self, index):

        data = unpack_data()
        print(data.keys())
        




if __name__ == '__main__':
    datasets_root_dir = './CUB_200_2011/images'
    classes_path = './CUB_200_2011/classes.txt'
    dir = datasets_root_dir + 'train_cropped/'
    target_dir = datasets_root_dir + 'train_cropped_augmented/'
    load_dataset(file_path='./train.json', sub_data='train')
    #augmenter(dir, target_dir)
    with open('./train.json') as file:
        full_data = json.load(file)
    #print(full_data['train'])
    #train_valid_test(datasets_root_dir)

    
    # for i in range(0, 60):

    #     print(np.random.randint(0,4))

    
    
        







    



