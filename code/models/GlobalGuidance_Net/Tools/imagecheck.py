import os
from PIL import Image

train_img_benign_adr = '..\\Data\\BUSI\\train\\image\\benign\\'
train_img_malignant_adr = '..\\Data\\BUSI\\train\\image\\malignant\\'
train_img_normal_adr = '..\\Data\\BUSI\\train\\image\\normal\\'
test_img_benign_adr = '..\\Data\\BUSI\\test\\image\\benign\\'
test_img_malignant_adr = '..\\Data\\BUSI\\test\\image\\malignant\\'
test_img_normal_adr = '..\\Data\\BUSI\\test\\image\\normal\\'

train_mask_benign_adr = '..\\Data\\BUSI\\train\\mask\\benign\\'
train_mask_malignant_adr = '..\\Data\\BUSI\\train\\mask\\malignant\\'
train_mask_normal_adr = '..\\Data\\BUSI\\train\\mask\\normal\\'
test_mask_benign_adr = '..\\Data\\BUSI\\test\\mask\\benign\\'
test_mask_malignant_adr = '..\\Data\\BUSI\\test\\mask\\malignant\\'
test_mask_normal_adr = '..\\Data\\BUSI\\test\\mask\\normal\\'

adr_set = [train_img_benign_adr,train_img_malignant_adr,train_img_normal_adr,test_img_benign_adr,test_img_malignant_adr,test_img_normal_adr,
           train_mask_benign_adr,train_mask_malignant_adr,train_mask_normal_adr,test_mask_benign_adr,test_mask_malignant_adr,test_mask_normal_adr]

# 检查文件通道
def image_check():
    error_set = []
    for i in range(12):
        adr = adr_set[i]
        for file in os.listdir(adr):
            image = Image.open(os.path.join(adr, file))
            if 'normal' in file and 'mask' in file:
                nor_image = Image.new('1',(image.size[0],image.size[1]))
                image = nor_image
            if  not 'mask' in file and len(image.split())!=3:
                error_set.append(os.path.join(adr,file))
                print(os.path.join(adr,file), '---', len(image.split()))
            if 'mask' in file and len(image.split())!=1:
                error_set.append(os.path.join(adr, file))
                print(os.path.join(adr,file), '---', len(image.split()))
    return error_set
# ERROR CHANNEL
# ..\Data\BUSI\train\mask\benign\benign (100)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (163)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (173)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (181)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (195)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (25)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (315)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (346)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (4)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (424)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (54)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (58)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (83)_mask.png --- 3
# ..\Data\BUSI\train\mask\benign\benign (92)_mask.png --- 3
# ..\Data\BUSI\train\mask\malignant\malignant (53)_mask.png --- 3
# ..\Data\BUSI\test\mask\benign\benign (93)_mask.png --- 3
# ..\Data\BUSI\test\mask\benign\benign (98)_mask.png --- 3
if __name__ == '__main__':
    error_set = image_check()
    image = Image.open(os.path.join(error_set[0]))
    image = image.convert('1')
    print(image)


