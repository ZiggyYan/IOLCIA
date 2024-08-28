#!/usr/bin/env python
import os
import cv2
import sys
from numpy.lib.type_check import iscomplex
import pytz
import tqdm
import torch
import random
import argparse
import numpy as np
import os.path as osp
import torch.nn.functional as F

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from dataloader import utils
from dataloader.breast_ultrasound.breastultra_dataloader import BreastSegmentation
from dataloader.breast_ultrasound import breastultra_transforms as tr
# from scipy.misc import imsave
from utils.utils_breastultrasound import joint_val_image, postprocessing, save_per_img
from utils.losses import *
from datetime import datetime
from models.networks.sdnet import MEncoder, AEncoder, Segmentor, Ada_Decoder
from medpy.metric import binary
torch.set_default_tensor_type('torch.FloatTensor')
from PIL import Image

import pydensecrf.densecrf as dcrf
# from .models.GlobalGuidanceNet.misc import crf_refine

#Global Guidance Net Module
from models.GlobalGuidance_Net.Seg_Module.modeling.deeplab_dense import *

#Sequence Process
from queue import *
from scipy.spatial.distance import cosine

#files
import logging
import yaml
def draw_min_rect_rectangle(mask_array):
    # image = cv2.imread(mask_path)
    # print(image.shape)
    thresh = cv2.Canny(mask_array, 128, 256)
 
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray_image = np.copy(mask_array)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 绘制矩形
        # cv2.rectangle(img,  (x, y+h), (x+w, y), (255,255,255),-1)
        cv2.rectangle(gray_image,  (x, y+h), (x+w, y), (255),-1)

         # 将最小内接矩形填充为白色
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 如果array已经是灰度数据，则不需要转换
    # print(img.shape)
    # print(type(img))
    # cv2.imwrite('z23213.png',img)
    return gray_image


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def invert_transform(tensor):
    ndarry = tensor.numpy().transpose((1, 2, 0)).astype(np.uint8)
    ndarry = ndarry.astype(np.uint8)
    return ndarry

def main():
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=123, help='random seed')
    # dir config
    parser.add_argument('--exp_dir', type=str, default='./output/cddsa/breastultrasound/')
    parser.add_argument('--data_dir', type=str, default='../Data/BreastUltrasound')
    # data config
    parser.add_argument('--data_size', type=int, default=256)
    # GPU file
    parser.add_argument('-g', '--gpu', type=int, default=1)
    # test config
    parser.add_argument('--model-file', type=str, default='train-domain[5]/20240419_212429.686261/models/best_model.pth', help='Model path')
    parser.add_argument('--datasetTest', type=list, default=[5], help='test folder id contain images ROIs to test')
    parser.add_argument('--dataset', type=str, default='test', help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channel', type=int, default=3)
    parser.add_argument('--save_imgs', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--z_length', type=int, default=16)
    parser.add_argument('--anatomy_channel', type=int, default=12)
    parser.add_argument('--strategy', type=bool, default=True)
    parser.add_argument('--confidence_threhold', type=float, default=0.05)
    # parser.add_argument('--similarity_time_threhold', type=float, default=0.75)
    parser.add_argument('--symbolic_similarity_threhold', type=float, default=0.99)
    parser.add_argument('--symbolic_confidence_threhold', type=float, default=0.65)
    parser.add_argument('--symbolic_open', type=bool, default=True)
    parser.add_argument('--time_open', type=bool, default=True)
    parser.add_argument('--extra_info', type=str, default='with_strategy_symbolic_confidence0.65')
    
    args = parser.parse_args()

    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)  # Numpy module.
        random.seed(args.seed)  # Python random module.
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = os.path.join(args.exp_dir, args.model_file)
    # output_path = os.path.join(args.exp_dir, 'test' + str(args.datasetTest[0]), args.model_file.split('/')[1])
    output_path = os.path.join(args.exp_dir, 'test-domain'+str(args.datasetTest),args.model_file.split('/')[1], args.extra_info)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = BreastSegmentation(base_dir=args.data_dir, phase='test', splitid=args.datasetTest,
                                    transform=composed_transforms_test, state='prediction')
    batch_size = args.batch_size
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 2. model
    m_encoder = MEncoder(z_length=args.z_length, in_channel=args.in_channel, img_size=args.data_size)
    a_encoder = AEncoder(in_channel=args.in_channel, width=256, height=256, ndf=16, num_output_channels=args.anatomy_channel, norm='batchnorm', upsample='bilinear')
    segmentor = DeepLab(num_classes=args.num_classes,
                  backbone='daf_ds',
                  output_stride=16,
                  mm='dgnlb',
                  sync_bn='auto',
                  freeze_bn=False).cuda()
    decoder = Ada_Decoder(anatomy_out_channel=args.anatomy_channel, z_length=args.z_length, out_channel=args.in_channel)
    
    if torch.cuda.is_available():
        models = {'M_enc': m_encoder.cuda(), 'A_enc': a_encoder.cuda(), 'Seg': segmentor.cuda(), 'Dec': decoder.cuda()}
    
    print('==> Loading model file: %s' % (model_file))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(model_file)
    for keys, md in models.items():
        pretrained_dict = checkpoint[keys]
        model_dict = md.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        md.load_state_dict(model_dict)
        models[keys] = md
    scores=[]
    Dices=[]
    ACs=[]
    SEs=[]
    SPs=[]
    Precisions=[]
    Adbs=[]
    Confms=[]
    
    # Initialization
    num_symbol = 10
    num_time = 20


    confidence_threhold = args.confidence_threhold
    # similarity_time_threhold = args.similarity_time_threhold
    symbolic_similarity_threhold = args.symbolic_similarity_threhold
    symbolic_confidence_threhold = args.symbolic_confidence_threhold
    symbolic_open = args.symbolic_open
    time_open= args.time_open
    
    # print(confidence_threhold)
    # if time_open:
    #     print('DUMB1')
    # if symbolic_open:
    #     print('DUMB2')

    
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    total_num = 0
    to_pil = transforms.ToPILImage()
    models['M_enc'].eval()
    models['A_enc'].eval()
    models['Seg'].eval()
    models['Dec'].eval()
    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):
        for batch in sample:
            if args.strategy:
                symbol_queue = Queue(maxsize=num_symbol)
                time_queue = Queue(maxsize=num_time)
            # data = batch['image']
            # target = batch['label']
            # img_name = batch['img_name']
            video_path = batch
            video_path = os.path.join(args.data_dir,'BUV','rawframes',video_path)
            filenames = os.listdir(video_path)
            filenames.sort()
            for filename in filenames:
                if 'mask' not in filename:
                    image_path = os.path.join(video_path,filename)
                    label_path = image_path.replace('.png','_mask.png') 
                    
                    frame_sample = {'image':  Image.open(image_path).convert('RGB').resize((256, 256), Image.LANCZOS), 'label': Image.open(label_path).convert('L')}
                    frame_sample = composed_transforms_test(frame_sample)
                    if torch.cuda.is_available():
                        data, target = frame_sample['image'].unsqueeze(0).cuda(), frame_sample['label'].unsqueeze(0).cuda()
                    data, target = Variable(data), Variable(target)
                    
                    with torch.no_grad():
                        a_out = models['A_enc'](data)
                        prediction = models['Seg'](a_out)
                        z_out, mu_out, logvar_out = models['M_enc'](data)
                        # for reconstruction
                        reco = models['Dec'](a_out, z_out)
                    prediction = torch.sigmoid(prediction[0]).data.squeeze(0).cpu()
                    prediction = np.array(to_pil(prediction))
                    image = Image.open(image_path).convert('RGB')
                    # prediction= crf_refine(np.array(data[0].cpu()),prediction)
                    prediction= crf_refine(np.array(image.resize((256,256))),prediction)/255  
                    prediction = torch.tensor(prediction)
                    # confidence = torch.mean(prediction[prediction>confidence_threhold])
                    # print(prediction.shape)
                    # print(type(prediction))
                    # prediction[prediction>=128]=255
                    # prediction[prediction<128]=0
                    # print(prediction[0][0][0].shape)
                    # print(type(prediction[0][0][0]))
                    # print(confidence)
                    # print('Aman ',torch.max(prediction))
                    # print('Amix ',torch.min(prediction))
                    if args.strategy: 
                        #symbol part
                        if symbolic_open:
                            if not symbol_queue.empty():
                                original_prediction = prediction.clone()
                                # prediction = 0.6*prediction
                                for i in range(symbol_queue.qsize()):
                                    item = symbol_queue.get()
                                    symbol_queue.put(item)
                                    processing_prediction = prediction.clone()
                                    processing_prediction[processing_prediction>confidence_threhold]=1
                                    processing_item = item.clone()
                                    processing_item[processing_item>confidence_threhold]=1
                                    similarity = np.dot(processing_item.flatten(), processing_prediction.flatten())/(np.linalg.norm(processing_prediction) * np.linalg.norm(processing_item))
                                    # print(similarity)
                                    # similarity = similarity = 1 - cosine(item, prediction)
                                    # PROCESS
                                    if similarity > symbolic_similarity_threhold:
                                        print('SOMETHING GET INSIDE')
                                        confidence = torch.mean(item[item>confidence_threhold])
                                        weight = confidence*0.4
                                        prediction = weight*item + prediction*(1.0-weight)
                        # confidence = torch.mean(prediction[prediction>confidence_threhold])
                        # print(confidence)
                        # print('Bman ',torch.max(prediction))
                        # print('Bmix ',torch.min(prediction))
                        # time part
                        if time_open:
                            if not time_queue.empty():
                                original_prediction = prediction
                                prediction = 0.6*prediction
                                length = time_queue.qsize()
                                inital_weight = (0.8/length+0.001-0.001*length)/2
                                for i in range(time_queue.qsize()):
                                    item = time_queue.get()
                                    time_queue.put(item)
                                    time_gap = time_queue.qsize() -i -1
                                    # # PROCESS
                                    # if similarity > similarity_time_threhold:
                                    prediction = item*(inital_weight+0.001*time_gap) + prediction
                                prediction[prediction>=1] = 1
                        # AFTER INFERENCE
                        # ADD
                        # Calculate Confidence
                        confidence = torch.mean(prediction[prediction>confidence_threhold])
                        # print(confidence)
                        # print('Cman ',torch.max(prediction))
                        # print('Cmix ',torch.min(prediction))
                        # print('----------------------------')
                        # Put in symbolic sequence
                        if confidence>symbolic_confidence_threhold:
                            # print('High confidence detected')
                            if symbol_queue.full():
                                garbage = symbol_queue.get()
                            symbol_queue.put(prediction)
                        # Put in time sequence
                        if time_queue.full():
                            itm = time_queue.get()
                        time_queue.put(prediction)
         
                    # prediction = torch.sigmoid(prediction[0]).data.squeeze(0).cpu()
                    # prediction = np.array(to_pil(prediction))
                    # # prediction= crf_refine(np.array(data[0].cpu()),prediction)
                    # prediction= crf_refine(np.array(image.resize((256,256))),prediction)
                    # prediction[prediction>=128]=255
                    # prediction[prediction<128]=0
                    prediction = prediction.numpy()
                    
                    prediction = (cv2.resize(prediction,(image.size[0],image.size[1]))*255).astype(np.uint8)
                    prediction[prediction>=100]=255
                    prediction[prediction<100]=0
                    prediction = draw_min_rect_rectangle(prediction)
                    if args.save_imgs and (batch_idx == 1 or batch_idx ==2):
                        if not os.path.exists(os.path.join(output_path,'prediction',video_path.split('/')[-1])):
                            os.makedirs(os.path.join(output_path,'prediction',video_path.split('/')[-1]))
                        (Image.fromarray(prediction,'L')).save(os.path.join(output_path,'prediction', video_path.split('/')[-1], filename))
                    # prediction = prediction.cpu()
                    target = cv2.imread(label_path,0)
                    # target = cv2.imread(os.path.join(batch['img_name'][0].replace('.png','_mask.png')),0)
                    prediction=cv2.resize(prediction,(image.size[0],image.size[1]))/255
                    prediction[prediction>=confidence_threhold]=1
                    prediction[prediction<confidence_threhold]=0
                    target[target>0]=1
                    target[target<=0]=0
                    TP=float(np.sum(np.logical_and(prediction==1,target==1)))
                    TN=float(np.sum(np.logical_and(prediction==0,target==0)))
                    FP=float(np.sum(np.logical_and(prediction==1,target==0)))
                    FN=float(np.sum(np.logical_and(prediction==0,target==1)))
                    JA=TP/((TP+FN+FP)+1e-5)
                    AC=(TP+TN)/((TP+FP+TN+FN+1e-5))
                    DI=2*TP/((2*TP+FN+FP+1e-5))
                    SE=TP/(TP+FN+1e-5)
                    SP=TN/((TN+FP)+1e-5)
                    precision=TP/(TP+FP+1e-5)
                    #     adb=evaluation.asd(pre,gt)
                    #     confm=(3*DI-2)/DI
                    Dices.append(DI)
                    scores.append(JA)
                    ACs.append(AC)
                    SEs.append(SE)
                    SPs.append(SP)
                    Precisions.append(precision)
                    
                    
                    
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data), Variable(target)
            
#             with torch.no_grad():
#                 a_out = models['A_enc'](data)
#                 prediction = models['Seg'](a_out)
#                 z_out, mu_out, logvar_out = models['M_enc'](data)
#                 # for reconstruction
#                 reco = models['Dec'](a_out, z_out)
#             #!!!!!!!!!!!!!
#             # image = Image.open(os.path.join(args.data_dir,'BUV','rawframes',batch['img_name'][0])).convert('RGB')
#             image = Image.open(img_name[0]).convert('RGB')
#             prediction = torch.sigmoid(prediction[0]).data.squeeze(0).cpu()
#             prediction = np.array(to_pil(prediction))
#             # prediction= crf_refine(np.array(data[0].cpu()),prediction)
#             prediction= crf_refine(np.array(image.resize((256,256))),prediction)
            
#             prediction[prediction>=128]=255
#             prediction[prediction<128]=0
#             if not os.path.exists(os.path.join(output_path, 'prediction')):
#                 os.makedirs(os.path.join(output_path, 'prediction'))
#             (Image.fromarray(prediction)).save(os.path.join(output_path, 'prediction', img_name[0].split('/')[-1]))
#             # prediction = prediction.cpu()
#             target = cv2.imread(os.path.join(batch['img_name'][0].replace('.png','_mask.png')),0)
#             prediction=cv2.resize(prediction,(image.size[0],image.size[1]))/255
#             prediction[prediction>=0.1]=1
#             prediction[prediction<0.1]=0
#             target[target>0]=1
#             target[target<0]=0
#             TP=float(np.sum(np.logical_and(prediction==1,target==1)))
#             TN=float(np.sum(np.logical_and(prediction==0,target==0)))
#             FP=float(np.sum(np.logical_and(prediction==1,target==0)))
#             FN=float(np.sum(np.logical_and(prediction==0,target==1)))
#             JA=TP/((TP+FN+FP)+1e-5)
#             AC=(TP+TN)/((TP+FP+TN+FN+1e-5))
#             DI=2*TP/((2*TP+FN+FP+1e-5))
#             SE=TP/(TP+FN+1e-5)
#             SP=TN/((TN+FP)+1e-5)
#             precision=TP/(TP+FP+1e-5)
#             #     adb=evaluation.asd(pre,gt)
#             #     confm=(3*DI-2)/DI
#             Dices.append(DI)
#             scores.append(JA)
#             ACs.append(AC)
#             SEs.append(SE)
#             SPs.append(SP)
#             Precisions.append(precision)
#             # logging.basicConfig(filename=os.path.join(output_path, 'test.log'), level=logging.INFO,
#             #             format='%(asctime)s %(message)s')
#             # logger = logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#             # information = 'JA - '+np.mean(scores)+''
#             # logger.info('')'
    
    print('JA',np.mean(scores),
          'DI',np.mean(Dices),
          'acc',np.mean(ACs),
          'Recall',np.mean(SEs),
          'SP',np.mean(SPs),
          'precision',np.mean(Precisions))
    filename = os.path.join(output_path,'result.txt')
    # 追加写入内容到文件
    with open(filename, "a") as file:
        file.write("Results:\n")
        file.write("JA:"+str(np.mean(scores))+"\n")
        file.write("DI:"+str(np.mean(Dices))+"\n")
        file.write("acc:"+str(np.mean(ACs))+"\n")
        file.write("Recall:"+str(np.mean(SEs))+"\n")
        file.write("SP:"+str(np.mean(SPs))+"\n")
        file.write("precision:"+str(np.mean(Precisions))+"\n")
    return('JA',np.mean(scores),
          'DI',np.mean(Dices),
          'acc',np.mean(ACs),
          'Recall',np.mean(SEs),
          'SP',np.mean(SPs),
          'precision',np.mean(Precisions))
#             target_numpy = target.data.cpu()
#             imgs = data.data.cpu()
#             hd_OC = 100
#             asd_OC = 100
#             hd_OD = 100
#             asd_OD = 100
#             for i in range(prediction.shape[0]):
#                 prediction_post = postprocessing(prediction[i], dataset=args.dataset)
#                 # prediction_post = torch.sigmoid(prediction[i]).data.cpu().numpy()
#                 test_dice = val_dice_class(torch.from_numpy(prediction_post).permute(1,2,0).cuda(), target[i].permute(1,2,0), num_class=args.num_classes)
#                 val_disc_dice.append(test_dice[0].data.cpu().numpy())
#                 # val_disc_dice.append(test_dice[1].data.cpu().numpy())
#                 if np.sum(prediction_post[0, ...]) < 1e-4:
#                     hd_OC = 100
#                     asd_OC = 100
#                 else:
#                     hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
#                                         np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
#                     asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
#                                         np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
# #                 if np.sum(prediction_post[1, ...]) < 1e-4:
# #                     hd_OD = 100
# #                     asd_OD = 100
# #                 else:
# #                     hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
# #                                         np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

# #                     asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
# #                                         np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
#                 total_hd_OC.append(hd_OC)
#                 total_hd_OD.append(hd_OD)
#                 total_asd_OC.append(asd_OC)
#                 total_asd_OD.append(asd_OD)
#                 total_num += 1
#                 if args.save_imgs:
#                     for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
#                         # img, lt = utils.untransform(img, lt)
#                         # img = invert_transform(img)
#                         # lt = invert_transform(lt)
#                         save_per_img(img.numpy().transpose(1, 2, 0),
#                                     os.path.join(output_path,'test_results'),
#                                     img_name[i],
#                                     lp, lt, mask_path=None, ext="bmp")

#     print('OC:', val_disc_dice)
#     # print('OD:', val_disc_dice)
#     import csv
#     with open(output_path+'/Dice_results.csv', 'a+') as result_file:
#         wr = csv.writer(result_file, dialect='excel')
#         wr.writerow(['Result in: '+args.model_file])
#         for index in range(len(val_disc_dice)):
#             wr.writerow([torch.from_numpy(val_disc_dice[index]), torch.from_numpy(val_disc_dice[index])])

#     val_disc_dice_mean = np.mean(val_disc_dice)
#     val_disc_dice_std = np.std(val_disc_dice)
#     # val_disc_dice_mean = np.mean(val_disc_dice)
#     # val_disc_dice_std = np.std(val_disc_dice)
#     # total_dice_mean = np.mean(val_cup_dice+val_disc_dice)
#     # total_dice_std = np.std(val_cup_dice + val_disc_dice)
#     total_dice_mean = np.mean(val_disc_dice)
#     total_dice_std = np.std(val_disc_dice)
#     total_hd_OC_mean = np.mean(total_hd_OC)
#     total_hd_OC_std = np.std(total_hd_OC)
#     total_asd_OC_mean = np.mean(total_asd_OC)
#     total_asd_OC_std = np.std(total_asd_OC)
#     total_hd_OD_mean = np.mean(total_hd_OD)
#     total_hd_OD_std = np.std(total_hd_OD)
#     total_asd_OD_mean = np.mean(total_asd_OD)
#     total_asd_OD_std = np.std(total_asd_OD)

#     print('''\n==>val_disc_dice : {0}-{1}'''.format(val_disc_dice_mean, val_disc_dice_std))
#     # print('''\n==>val_disc_dice : {0}-{1}'''.format(val_disc_dice_mean, val_disc_dice_std))
#     print('''\n==>val_average_dice : {0}-{1}'''.format(total_dice_mean, total_dice_std))
#     print('''\n==>ave_hd_OC : {0}-{1}'''.format(total_hd_OC_mean, total_hd_OC_std))
#     print('''\n==>ave_hd_OD : {0}-{1}'''.format(total_hd_OD_mean, total_hd_OD_std))
#     print('''\n==>ave_asd_OC : {0}-{1}'''.format(total_asd_OC_mean, total_asd_OC_std))
#     print('''\n==>ave_asd_OD : {0}-{1}'''.format(total_asd_OD_mean, total_asd_OD_std))
#     with open(osp.join(output_path, 'test' + str(args.datasetTest[0]) + '_log.csv'), 'a') as f:
#         elapsed_time = (
#                 datetime.now(pytz.timezone('Asia/Hong_Kong')) -
#                 timestamp_start).total_seconds()
#         log = [['batch-size: '] + [batch_size] + [args.model_file] + ['disc dice coefficence: '] + \
#                [val_disc_dice_mean]+['-']+[val_disc_dice_std] + ['total dice coefficence: '] + \
#                [total_dice_mean]+['-']+[total_dice_std] + ['average_hd_OC: '] + \
#                [total_hd_OC_mean]+['-']+[total_hd_OC_std] + ['average_hd_OD: '] + \
#                [total_hd_OD_mean]+['-']+[total_hd_OD_std] + ['ave_asd_OC: '] + \
#                [total_asd_OC_mean]+['-']+[total_asd_OC_std] + ['average_asd_OD: '] + \
#                [total_asd_OD_mean]+['-']+[total_asd_OD_std] + ['elapse time: '] + \
#                [elapsed_time]]
#         log = map(str, log)
#         f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()
