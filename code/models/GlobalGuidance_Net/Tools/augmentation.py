# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os.path
from tqdm import tqdm


# 椒盐噪声
def SaltAndPepper(src, percetage):
    SP_NoiseImg = src.copy()
    SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(SP_NoiseNum):
        randR = np.random.randint(0, src.shape[0] - 1)
        randG = np.random.randint(0, src.shape[1] - 1)
        randB = np.random.randint(0, 3)
        if np.random.randint(0, 1) == 0:
            SP_NoiseImg[randR, randG, randB] = 0
        else:
            SP_NoiseImg[randR, randG, randB] = 255
    return SP_NoiseImg


# 高斯噪声
def addGaussianNoise(image, percetage):
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg


# 昏暗
def darker(image, percetage=0.9):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get darker
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = int(image[xj, xi, 0] * percetage)
            image_copy[xj, xi, 1] = int(image[xj, xi, 1] * percetage)
            image_copy[xj, xi, 2] = int(image[xj, xi, 2] * percetage)
    return image_copy


# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    # get brighter
    for xi in range(0, w):
        for xj in range(0, h):
            image_copy[xj, xi, 0] = np.clip(int(image[xj, xi, 0] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 1] = np.clip(int(image[xj, xi, 1] * percetage), a_max=255, a_min=0)
            image_copy[xj, xi, 2] = np.clip(int(image[xj, xi, 2] * percetage), a_max=255, a_min=0)
    return image_copy


# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


# 饱和度
class Saturation:
    def __init__(self, saturation_factor):
        self.saturation_factor = saturation_factor

    def __call__(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 通过cv2.cvtColor把图像从BGR转换到HSV
        colorless_hsv = img_hsv.copy()
        colorless_hsv[:, :, 1] = self.saturation_factor * colorless_hsv[:, :, 1]
        return cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)


# 翻转
class Flip:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, img):
        return cv2.flip(img, self.mode)


# # 翻转
# def flip(image):
#     flipped_image = np.fliplr(image)
#     return flipped_image

if __name__ == '__main__':

    # 图片文件夹路径
    file_dir = './Dataset/image/'
    file_list = os.listdir(file_dir)
    for img_name in tqdm(file_list):
        img_path = file_dir + img_name
        img = cv2.imread(img_path)

        # # Rotate
        # rotated_180 = rotate(img, 180)
        # cv2.imwrite(file_dir + img_name[0:-4] + '_r180.jpg', rotated_180)
        # Add Salt and Pepper effect
        if img_name.startswith('1'):
            print(img_name)
            if not 'salt' in img_name and not 'noise' in img_name and not 'darker' in img_name \
                    and not 'brighter' in img_name and not 'blur' in img_name  and not 'dilate' in img_name \
                    and not 'erosion' in img_name and not 'satup1' in img_name and not 'satlow1' in img_name:
                if not os.path.exists(file_dir + img_name[0:-4] + '_salt.png'):
                    img_salt = SaltAndPepper(img, 0.3)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_salt.png', img_salt)
                # Add Noise
                if not os.path.exists(file_dir + img_name[0:-4] + '_noise.png'):
                    img_gauss = addGaussianNoise(img, 0.3)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_noise.png', img_gauss)
                # Brighter and Darker
                if not os.path.exists(file_dir + img_name[0:-4] + '_darker.png'):
                    img_darker = darker(img)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_darker.png', img_darker)
                if not os.path.exists(file_dir + img_name[0:-4] + '_brighter.png'):
                    img_brighter = brighter(img)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_brighter.png', img_brighter)
                # Blur
                if not os.path.exists(file_dir + img_name[0:-4] + '_blur.png'):
                    blur = cv2.GaussianBlur(img, (7, 7), 1.5)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_blur.png', blur)
                # # Horizontal Mirror
                # flip_horizon = Flip(mode=1)
                # img_horizon = flip_horizon(img)
                # cv2.imwrite(file_dir + img_name[0:-4] + '_horizon.jpg', img_horizon)

                # # Vertical Mirror
                # flip_vertical = Flip(mode=0)
                # img_vertical = flip_vertical(img)
                # cv2.imwrite(file_dir + img_name[0:-4] + '_vertical.jpg', img_vertical)

                # Image erode and dilate
                kernel_dil_ero = np.ones((3, 3), dtype=np.uint8)

                if not os.path.exists(file_dir + img_name[0:-4] + '_dilate.png'):
                    img_dilate = cv2.dilate(img, kernel_dil_ero, 1)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_dilate.png', img_dilate)
                if not os.path.exists(file_dir + img_name[0:-4] + '_erosion.png'):
                    img_erosion = cv2.erode(img, kernel_dil_ero, 1)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_erosion.png', img_erosion)

                # Saturation
                satup1 = Saturation(0.5)
                # satup2 = Saturation(0.7)
                satlow1 = Saturation(1.5)
                # satlow2 = Saturation(1.3)

                # img_satup2 = satup2(img)

                # img_satlow2 = satlow2(img)
                if not os.path.exists(file_dir + img_name[0:-4] + '_satup1.png'):
                    img_satup1 = satup1(img)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_satup1.png', img_satup1)
                # cv2.imwrite(file_dir + img_name[0:-4] + '_satup2.png', img_satup2)
                if not os.path.exists(file_dir + img_name[0:-4] + '_satlow1.png'):
                    img_satlow1 = satlow1(img)
                    cv2.imwrite(file_dir + img_name[0:-4] + '_satlow1.png', img_satlow1)
                # cv2.imwrite(file_dir + img_name[0:-4] + '_satlow2.png', img_satlow2)
                # 9倍

