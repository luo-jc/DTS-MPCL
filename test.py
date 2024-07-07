import os
import torch
from time import strftime, localtime
from utils import *
from DTS import *
import time
import math
import numpy as np
import cv2
import sys
from collections import OrderedDict

device = torch.device("cuda")

logger = Logger(os.path.join(r'../log', 'test_log.txt'))
logger('file_directory: {},----test_time: {}'.format(sys.argv[0], strftime("%Y-%m-%d %H:%M:%S", localtime())))
scene_directory = r'../Test'


class PSNR():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        return 20 * math.log10(self.range / math.sqrt(mse))


class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

psnr = PSNR()
ssim = SSIM()

list = os.listdir(scene_directory)

psnr_u_sum = 0
psnr_L_sum = 0
ssim_L_sum = 0
ssim_u_sum = 0

for scene in range(len(list)):
    expoTimes = ReadExpoTimes(os.path.join(scene_directory, list[scene], 'exposure.txt'))
    imgs = ReadImages(list_all_files_sorted(os.path.join(scene_directory, list[scene]), '.tif'))
    label = ReadLabel(os.path.join(scene_directory, list[scene]))
    label_L = label
    label_u = range_compressor(label)
    pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
    pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
    pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
    output0 = np.concatenate((imgs[0], pre_img0), 2)
    output1 = np.concatenate((imgs[1], pre_img1), 2)
    output2 = np.concatenate((imgs[2], pre_img2), 2)

    im1 = torch.Tensor(output0).to(device)
    im1 = torch.unsqueeze(im1, 0).permute(0, 3, 1, 2)

    im2 = torch.Tensor(output1).to(device)
    im2 = torch.unsqueeze(im2, 0).permute(0, 3, 1, 2)

    im3 = torch.Tensor(output2).to(device)
    im3 = torch.unsqueeze(im3, 0).permute(0, 3, 1, 2)

    model = DTS().to(device)
    model.eval()
    state_dict = torch.load('../Model/DTS.pkl')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    _, _, H, W = im1.shape
    H, W = int(H), int(W)
    print(H, W)

    with torch.no_grad():
        im1_11, im1_12, im1_21, im1_22 = im1[:, :, 0:int(H/2), 0:int(W/2)], im1[:, :, 0:int(H/2), int(W/2):W], im1[:, :, int(H/2):H, 0:int(W/2)], im1[:, :, int(H/2):H, int(W/2):W]
        im2_11, im2_12, im2_21, im2_22 = im2[:, :, 0:int(H/2), 0:int(W/2)], im2[:, :, 0:int(H/2), int(W/2):W], im2[:, :, int(H/2):H, 0:int(W/2)], im2[:, :, int(H/2):H, int(W/2):W]
        im3_11, im3_12, im3_21, im3_22 = im3[:, :, 0:int(H/2), 0:int(W/2)], im3[:, :, 0:int(H/2), int(W/2):W], im3[:, :, int(H/2):H, 0:int(W/2)], im3[:, :, int(H/2):H, int(W/2):W]

        pre_single_11, pre_multi_11, out_all_11, _, _, _, _ = model(im1_11, im2_11, im3_11)
        pre_single_12, pre_multi_12, out_all_12, _, _, _, _ = model(im1_12, im2_12, im3_12)
        pre_single_21, pre_multi_21, out_all_21, _, _, _, _ = model(im1_21, im2_21, im3_21)
        pre_single_22, pre_multi_22, out_all_22, _, _, _, _ = model(im1_22, im2_22, im3_22)

        pre_single_1, pre_multi_1, out_all_1 = torch.cat([pre_single_11, pre_single_21], dim=2), torch.cat([pre_multi_11, pre_multi_21], dim=2), torch.cat([out_all_11, out_all_21], dim=2)
        pre_single_2, pre_multi_2, out_all_2 = torch.cat([pre_single_12, pre_single_22], dim=2), torch.cat([pre_multi_12, pre_multi_22], dim=2), torch.cat([out_all_12, out_all_22], dim=2)
        pre_single, pre_multi, out_all = torch.cat([pre_single_1, pre_single_2], dim=3), torch.cat([pre_multi_1, pre_multi_2], dim=3), torch.cat([out_all_1, out_all_2], dim=3)

    pre_total = torch.clamp(out_all, 0., 1.)
    pre_total = pre_total.permute(0, 2, 3, 1)
    pre_total = pre_total.data[0].cpu().numpy()
    output_total = cv2.cvtColor(pre_total, cv2.COLOR_BGR2RGB)
    cv2.imwrite('../test_result/DTS_MPCL_image_{}.hdr'.format(list[scene]), output_total)

    p_total_L = psnr(pre_total, label_L)
    s_total_L = ssim(pre_total, label_L)
    psnr_L_sum += p_total_L
    ssim_L_sum += s_total_L

    pre_total = range_compressor(pre_total)
    p_total_u = psnr(pre_total, label_u)
    s_total_u = ssim(pre_total, label_u)
    psnr_u_sum += p_total_u
    ssim_u_sum += s_total_u

psnr_avg_u = psnr_u_sum / len(list)
psnr_avg_L = psnr_L_sum / len(list)
ssim_avg_L = ssim_L_sum / len(list)
ssim_avg_u = ssim_u_sum / len(list)

logger('psnr_avg_L:{},psnr_avg_u:{},ssim_avg_L:{},ssim_avg_u:{}'.format(psnr_avg_L, psnr_avg_u, ssim_avg_L, ssim_avg_u))

