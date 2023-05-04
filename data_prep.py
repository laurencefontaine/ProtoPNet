import os
import shutil

from PIL import Image


def main():
    f_test = open("CUB_200_2011/CUB_200_2011/train_test_split.txt", "r")
    f_path = open("CUB_200_2011/CUB_200_2011/images.txt", "r")
    f_crop = open("CUB_200_2011/CUB_200_2011/bounding_boxes.txt", "r")

    for line_test, line_path, line_crop in zip(f_test.readlines(), f_path.readlines(), f_crop.readlines()):
        test = line_test.replace("\n", "").split(' ')[1]
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


if __name__ == '__main__':
    main()