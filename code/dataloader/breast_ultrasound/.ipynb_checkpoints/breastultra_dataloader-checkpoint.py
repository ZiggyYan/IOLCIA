from __future__ import print_function, division
import os
import sys
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import copy
import torch
from torchvision import transforms
import json
import cv2

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

class BreastSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 4 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir='../../../Data/BreastUltrasound',
                 phase='train',
                 splitid=[1,2,3],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        ID - 1 => BUSBRA
        ID - 2 => BUSI
        ID - 3 => USG
        ID - 4 => BUSBset
        ID - 5 => BUV[MAIN TASK]
        """
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        if phase == 'test':
            self.video_list = []
        else:
            self.image_list = []
        self.phase = phase
        self.image_pool = {'BUSBRA':[], 'BUSI':[], 'USG':[], 'BUSB':[], 'BUV':[]}
        self.label_pool = {'BUSBRA':[], 'BUSI':[], 'USG':[], 'BUSB':[], 'BUV':[]}
        self.img_name_pool = {'BUSBRA':[], 'BUSI':[], 'USG':[], 'BUSB':[], 'BUV':[]}

        self.flags_BUSBRA = ['bus']
        self.flags_BUSI = ['ben','mal','nor']
        self.flags_USG = ['case']
        self.flags_BUSB = ['BUSBset']
        self.flags_BUV =['BUV']
        self.splitid = splitid
        SEED = 1212
        random.seed(SEED)
        for id in splitid:
            if id == 1:
                self._image_dir = os.path.join(self._base_dir, 'BUSBRA', phase,'image/')
            elif id == 2:
                self._image_dir = os.path.join(self._base_dir, 'BUSI', phase, 'image/')
            elif id == 3:
                self._image_dir = os.path.join(self._base_dir, 'USG', phase, 'image/')
            elif id == 4:
                self._image_dir = os.path.join(self._base_dir, 'BUSBset', phase, 'image/')
            elif id == 5:
                # AS FOR BUV, READ JSON HERE.
                # HERE THE "TRAIN" AND "VALID" REFERS TO THE DIVIDED PART OF ORIGINAL BUV DATASET
                # DOES NOT REFER TO ACTUAL "TRAIN" AND VALID IN THIS PROJECT
                if phase == 'test':
                    train_json_dir = os.path.join(self._base_dir, 'BUV', 'imagenet_vid_train_15frames.json')
                    valid_json_dir = os.path.join(self._base_dir, 'BUV', 'imagenet_vid_val.json')
                    train_data = read_json_file(train_json_dir)
                    valid_data = read_json_file(valid_json_dir)
                    train_videos = train_data['videos']
                    valid_videos = valid_data['videos']
                    video_paths = []
                    for i in range(len(train_videos)):
                        video_path = train_videos[i]['name']
                        video_paths.append(video_path)
                    for i in range(len(valid_videos)):
                        video_path = valid_videos[i]['name']
                        video_paths.append(video_path)
                elif phase == 'valid':
                    valid_json_dir = os.path.join(self._base_dir, 'BUV', 'imagenet_vid_val.json')
                    valid_data = read_json_file(valid_json_dir)
                    # FETCH THE IMAGE ADR
                    valid_images = valid_data['images']
                    # TO SHRINK THE SIZE OF TESTING DATA
                    # EVERY TIME EXPERIMENT STARTS, PROGRAM SELECTES A RANDOM PART FROM ORIGINAL DATASET
                    valid_image_paths = {}
                    valid_mask_paths = {}
                    for i in range(len(valid_images)):
                        image_name = valid_images[i]['file_name']
                        image_path = os.path.join(self._base_dir, 'BUV','rawframes', image_name)
                        valid_image_paths[str(valid_images[i]['id'])] = image_path
                        mask_path = image_path.replace('.png','_mask.png')
                        valid_mask_paths[str(valid_images[i]['id'])] = mask_path 
                self._image_dir = os.path.join(self._base_dir, 'BUV', 'rawframes/')
            # self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), phase, 'ROIs','image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))
            if id != 5:
                imagelist = glob(self._image_dir + '*.png')
                for image_path in imagelist:
                    gt_path = ''
                    if id == 1:
                        gt_path = image_path.replace('image', 'mask')
                        gt_path = gt_path.replace('bus', 'mask')
                    elif id == 2:
                        gt_path = image_path.replace('image', 'mask')
                        gt_path = gt_path.replace('.png', '_mask.png')
                    elif id == 3:
                        gt_path = image_path.replace('image', 'mask')
                        gt_path = gt_path.replace('.png', '_tumor.png')
                    elif id == 4:
                        gt_path = image_path.replace('image', 'mask')
                    # elif id == 5:
                    #     gt_path = image_path.replace('.png', '_mask.png')
                    # gt_path = image_path.replace('image', 'mask')
                    self.image_list.append({'image': image_path, 'label': gt_path})
            elif id == 5:
                if phase == 'valid':
                    for i in range(len(valid_images)):
                        id = str(valid_images[i]['id'])
                        image_path = valid_image_paths[id]
                        mask_path = valid_mask_paths[id]
                        self.image_list.append({'image': image_path, 'label': mask_path})
                elif phase == 'test':
                    for i in range(len(video_paths)):
                        video_path = video_paths[i]
                        self.video_list.append({'path': video_path})
                
        
        self.transform = transform
        if phase != 'test':
            self._read_img_into_memory()
            for key in self.image_pool:
                if len(self.image_pool[key]) < 1:
                    del self.image_pool[key]
                    del self.label_pool[key]
                    del self.img_name_pool[key]
                    break
            for key in self.image_pool:
                if len(self.image_pool[key]) < 1:
                    del self.image_pool[key]
                    del self.label_pool[key]
                    del self.img_name_pool[key]
                    break
            for key in self.image_pool:
                if len(self.image_pool[key]) < 1:
                    del self.image_pool[key]
                    del self.label_pool[key]
                    del self.img_name_pool[key]
                    break
            # Display stats
            print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))
        else:
            print('-----Total number of videos in {}: {:d}'.format(phase, len(self.video_list)))

    def __len__(self):
        max = -1
        if self.phase != 'test':
            for key in self.image_pool:
                 if len(self.image_pool[key])>max:
                     max = len(self.image_pool[key])
        else:
            max = len(self.video_list)
        return max

    def __getitem__(self, index):
        if self.phase == 'train':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample.append(anco_sample)
        elif self.phase == 'valid':
            sample = []
            for key in self.image_pool:
                try:
                    domain_code = list(self.image_pool.keys()).index(key)
                    _img = self.image_pool[key][index]
                    _target = self.label_pool[key][index]
                    _img_name = self.img_name_pool[key][index]
                    anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                except:
                    continue
                
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                
                sample.append(anco_sample)
        elif self.phase == 'test':
            sample = self.video_list[index]['path']
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:3] in self.flags_BUSBRA:
                Flag = 'BUSBRA'
            elif basename[0:3] in self.flags_BUSI:
                Flag = 'BUSI'
            elif basename[0:4] in self.flags_USG:
                Flag = 'USG'
            elif self.flags_BUSB[0] in self.image_list[index]['image']: 
                Flag = 'BUSB'
            elif self.flags_BUV[0] in self.image_list[index]['image']:
                Flag = 'BUV'
            #!!!
            # elif basename[0] in self.flags_REF_val:
            #     Flag = 'REF_val'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            if self.splitid[0] == '4':
                # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB').crop((144, 144, 144+512, 144+512)).resize((256, 256), Image.LANCZOS))
                _target = np.asarray(Image.open(self.image_list[index]['label']).convert('L'))
                _target = _target[144:144+512, 144:144+512]
                _target = Image.fromarray(_target)
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
                # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB'))
                _target = Image.open(self.image_list[index]['label'])
            # print('----------')
            # print(type(_target))
            # print(_target.mode)  # mode:'RGBA'、'RGB'、'L'
            # to_tensor = transforms.ToTensor()
            # _target = to_tensor(_target)
            # print(_target.shape) # C H W
            # print('----------')
            # check = transforms.functional.pil_to_tensor(_target)
            # check = torch.tensor(check)
            # if torch.all(check == 0):
            #     print('BEFORE PROBLEM')
            # check = transforms.functional.pil_to_tensor(Image.open(self.image_list[index]['image']).convert('RGB').resize((256, 256), Image.LANCZOS))
            # check = torch.tensor(check)
            # if torch.all(check == 0):
            #     print('PREPARE PROBLEM')
                
            if _target.mode != 'L':
                _target = _target.convert('L')
            if self.state != 'prediction':
                _target = _target.resize((256, 256))
                
            # check = transforms.functional.pil_to_tensor(_target)
            # check = torch.tensor(check)
            # if torch.all(check == 0):
            #     print('AFTER PROBLEM')
                
            # print(_target.size)
            # print(_target.mode)
            # print(torch.all(_target== 0))
            self.label_pool[Flag].append(_target)
            # print(self.image_list[index]['image'])
            # if self.split[0:4] in 'test':
            # _img_name = self.image_list[index]['image'].split('/')[-1]
            _img_name = self.image_list[index]['image']
            self.img_name_pool[Flag].append(_img_name)

    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(args.datasetTest[0]) + ')'


if __name__ == '__main__':
    import dataloader.ms_fundus.fundus_transforms as tr
    # from dataloader.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomSizedCrop(512),
        tr.RandomRotate(15),
        tr.Normalize_tf(),
        tr.ToTensor()])

    voc_train = FundusSegmentation(phase='train', splitid=[1],
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample[0]["image"].size()[0]):
            img = sample[0]['image'].numpy()
            gt = sample[0]['label'].numpy()
            segmap = np.transpose(gt[jj], axes=[1, 2, 0]).astype(np.uint8)
            img_tmp = np.transpose((img[jj]+1.0)*128, axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(221)
            plt.imshow(img_tmp)
            plt.subplot(222)
            plt.imshow(segmap[..., 0])
            plt.subplot(223)
            plt.imshow(segmap[..., 1])

            break
    plt.show(block=True)
