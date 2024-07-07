import numpy as np
import os, glob
import cv2
from math import log10
import torch
import torch.nn.functional as F
import skimage.metrics as sk
import math
from bisect import bisect_right
from torch import nn
from glob import glob

def js_div(p_output, q_output, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def ReadExpoTimes(fileName):
    return np.power(2, np.loadtxt(fileName))

def ReadExpoTimes_test(fileName):
    return np.power(2, fileName)

def list_all_files_sorted(folderName, extension=""):
    return sorted(glob(os.path.join(folderName, "*" + extension)))


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.imread(imgStr, cv2.IMREAD_UNCHANGED)
        img = np.float32(img)
        img = img / 2 ** 16
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)


def ReadImages_test(fileNames):
    img = cv2.imread(fileNames, cv2.IMREAD_UNCHANGED)
    img = np.float32(img)
    img = img / 2 ** 16
    img.clip(0, 1)
    return np.array(img)

def ReadLabel(fileName):
    label = cv2.imread(os.path.join(fileName, 'HDRImg.hdr'), flags = cv2.IMREAD_UNCHANGED)
    label = label[:, :, [2, 1, 0]]
    return label



def LDR_to_HDR(imgs, expo, gamma):
    return (imgs ** gamma) / expo


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)


def reverse_tonemap(CompressedImage):
    return ((np.power(5001, CompressedImage)) - 1) / 5000

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += sk.peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def batch_SSIM(img, imclean, data_range):
    Img = img.data.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    Iclean = imclean.data.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += sk.structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range, multichannel=True)
    return (SSIM/Img.shape[0])


def print_test_log(epoch, num_epochs, test_psnr, test_ssim, test_psnr_u, test_ssim_u, test_loss, category):
    print(' Epoch [{0}/{1}], Test_PSNR:{2:.2f}, Test_SSIM:{3:.2f},Test_PSNR_u:{4:.2f},Test_SSIM_u:{5:.2f},Test_loss:{6:.3f}'
          .format(epoch, num_epochs, test_psnr, test_ssim,  test_psnr_u, test_ssim_u, test_loss,))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, input):
        input = str(input)
        with open(self.log_file, 'a') as f:
            f.writelines(input+'\n')
        print(input)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class Loss_FDRT(nn.Module):
    def __init__(self):
        super(Loss_FDRT, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input):
        b, c, h, w = input.size()
        feature = input.view(b, c, h * w)
        gram = torch.bmm(feature, feature.permute(0, 2, 1))
        # gram /= (a * b * c * d)
        i_mat = torch.eye(c).to(input.device)
        i_mat_tensor = i_mat.repeat(b, 1, 1)
        loss_fdrt = self.loss(gram, i_mat_tensor) / pow(c, 2)
        return loss_fdrt

GAMMA = 2.2

def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = ReadImages_test(path)
    return img.astype(np.float32)

def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0, j):min(h, j+crop_h), max(0, i):min(w, i+crop_w)], (crop_w, crop_h))

def transform(image, image_size, is_crop):
    if is_crop:
        out = center_crop(image, image_size)
    elif image_size is not None:
        out = cv2.resize(image, image_size)
    else:
        out = image
    return out.astype(np.float32)


def get_image(image_path, image_size=None, is_crop=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(imread(image_path), image_size, is_crop)


def get_input(LDR_path, exp_path, ref_HDR_path):
    in_LDR_paths = sorted(glob(LDR_path))
    ns = len(in_LDR_paths)
    tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
    h, w, c = tmp_img.shape
    h = h // 16 * 16
    w = w // 16 * 16

    in_exps = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
    imgs = []
    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=[h, w], is_crop=True)
        imgs.append(img)
    ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)
    return imgs, in_exps, ref_HDR

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

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
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
