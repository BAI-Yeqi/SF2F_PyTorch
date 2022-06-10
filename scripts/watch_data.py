import os
import numpy as np
import random
import pickle
import json
import PIL
from PIL import Image

DIR = './data/VoxCeleb/'


def main():
    dir_1 = os.path.join(DIR, 'vox1/masked_faces')
    names = os.listdir(dir_1)
    print("Length of names: ", len(names))  # 1187
    img_num_1 = 0
    all_W1, all_H1 = 0, 0
    num1_50, num1_100, num1_200, num1_300, num1_400, num1_600 = 0, 0, 0, 0, 0, 0
    for i, name in enumerate(names):
        path = os.path.join(dir_1, name)
        images = os.listdir(path)
        for j, image in enumerate(images):
            image_path = os.path.join(path, image)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    img_num_1 += 1
                    all_W1 += WW
                    all_H1 += HH
                    # print("PIL Image:", np.array(image))
                    if (WW >= 50 and WW < 100) or (HH >= 50 and HH < 100):
                        num1_50 += 1
                    elif (WW >= 100 and WW < 200) or (HH >= 100 and HH < 200):
                        num1_100 += 1
                    elif (WW >= 200 and WW < 300) or (HH >= 200 and HH < 300):
                        num1_200 += 1
                    elif (WW >= 300 and WW < 400) or (HH >= 300 and HH < 400):
                        num1_300 += 1
                    elif (WW >= 400 and WW < 600) or (HH >= 400 and HH < 600):
                        num1_400 += 1
                    elif (WW >= 600) or (HH >= 600):
                        num1_600 += 1

    print("======== Vox1 =========")
    print("Number of vox1: ", img_num_1)
    print("Average W: ", all_W1/img_num_1)
    print("Average H: ", all_H1/img_num_1)
    print("50~100: ", num1_50)
    print("100~200: ", num1_100)
    print("200~300: ", num1_200)
    print("300~400: ", num1_300)
    print("400~600: ", num1_400)
    print("600+: ", num1_600)
    print("\n")

    dir_2 = os.path.join(DIR, 'vox2/masked_faces')
    names = os.listdir(dir_2)
    print("Length of names: ", len(names))  # 2850
    img_num_2 = 0
    all_W2, all_H2 = 0, 0
    num2_50, num2_100, num2_200, num2_300, num2_400, num2_600 = 0, 0, 0, 0, 0, 0
    for i, name in enumerate(names):
        path = os.path.join(dir_2, name)
        images = os.listdir(path)
        for j, image in enumerate(images):
            image_path = os.path.join(path, image)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    img_num_2 += 1
                    all_W2 += WW
                    all_H2 += HH
                    # print("PIL Image:", np.array(image))
                    if (WW >= 50 and WW < 100) or (HH >= 50 and HH < 100):
                        num2_50 += 1
                    elif (WW >= 100 and WW < 200) or (HH >= 100 and HH < 200):
                        num2_100 += 1
                    elif (WW >= 200 and WW < 300) or (HH >= 200 and HH < 300):
                        num2_200 += 1
                    elif (WW >= 300 and WW < 400) or (HH >= 300 and HH < 400):
                        num2_300 += 1
                    elif (WW >= 400 and WW < 600) or (HH >= 400 and HH < 600):
                        num2_400 += 1
                    elif (WW >= 600) or (HH >= 600):
                        num2_600 += 1

    print("======== Vox2 =========")
    print("Number of vox2: ", img_num_2)
    print("Average W: ", all_W2/img_num_2)
    print("Average H: ", all_H2/img_num_2)
    print("50~100: ", num2_50)
    print("100~200: ", num2_100)
    print("200~300: ", num2_200)
    print("300~400: ", num2_300)
    print("400~600: ", num2_400)
    print("600+: ", num2_600)
    print("\n")

    print("======== ALL DATASET ========")
    print("Number of images: ", img_num_1 + img_num_2)
    print("Average W: ", (all_W1 + all_W2) / (img_num_1 + img_num_2))
    print("Average H: ", (all_H1 + all_H2) / (img_num_1 + img_num_2))
    print("50~100: ", num1_50 + num2_50)
    print("100~200: ", num1_100 + num2_100)
    print("200~300: ", num1_200 + num2_200)
    print("300~400: ", num1_300 + num2_300)
    print("400~600: ", num1_400 + num2_400)
    print("600+: ", num1_600 + num2_600)

if __name__ == '__main__':
    main()
