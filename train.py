import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from DTS import *
from datsetprocess import *
from MPCL import *
import random
import albumentations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_h = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).to(device)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).to(device)
        self.weight_h = kernel_h
        self.weight_v = kernel_v

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(20)
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        if classname.find('ConvTranspose2d') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        if classname.find('nn.Linear') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)

    BATCH_SIZE = 3
    BATCH_SIZE_TEST = 15
    Initial_LR = 0.0005

    transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
    ])

    train_data = MyDataset(scene_directory='../Training', transforms=transforms, noise=None, is_training=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)

    DTS = DTS().to(device)
    DTS = nn.DataParallel(DTS, device_ids=[0, 1])

    MPCL = MPCL().to(device)
    MPCL.eval()
    MPCL = nn.DataParallel(MPCL, device_ids=[0, 1])
    MPCL.load_state_dict(torch.load('../Model_MPCL/200.pkl', map_location=lambda storage, loc: storage))

    L1_loss = nn.L1Loss()
    L2_loss = nn.MSELoss()
    DTS.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(DTS.parameters(), lr=Initial_LR)
    scheduler = WarmupMultiStepLR(optimizer=optimizer_G, milestones=(1000, 1500), gamma=0.2, warmup_factor=1.0 / 3, warmup_iters=200, warmup_method="linear")
    counter = 0
    counter_test = 0
    good_number = 0
    Tensor = torch.cuda.FloatTensor

# ----------
#  Training
# ----------
    DTS.load_state_dict(torch.load('../Model/2000.pkl'))
    for epoch in range(2001, 10000):
        for step, sample in enumerate(train_loader, 0):
            a = time.time()
            batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample['label']
            batch_x1, batch_x2, batch_x3, batch_x4 = batch_x1.to(device), batch_x2.to(device), batch_x3.to(device), batch_x4.to(device)

            optimizer_G.zero_grad()

            pre_single, pre_multi, pre_final, gradient_s_1, gradient_m_1, gradient_s_2, gradient_m_2 = DTS(x1=batch_x1, x2=batch_x2, x3=batch_x3)
            gradient = Get_gradient()

            batch_x4_compressed = range_compressor_tensor(batch_x4)

            PSNR_single = batch_PSNR(torch.clamp(pre_single, 0., 1.), batch_x4, 1.)
            SSIM_single = batch_SSIM(torch.clamp(pre_single, 0., 1.), batch_x4, 1.)
            pre_compressed_single = range_compressor_tensor(pre_single)
            PSNR_u_single = batch_PSNR(torch.clamp(pre_compressed_single, 0., 1.), batch_x4_compressed, 1.)
            SSIM_u_single = batch_SSIM(torch.clamp(pre_compressed_single, 0., 1.), batch_x4_compressed, 1.)

            PSNR_multi = batch_PSNR(torch.clamp(pre_multi, 0., 1.), batch_x4, 1.)
            SSIM_multi = batch_SSIM(torch.clamp(pre_multi, 0., 1.), batch_x4, 1.)
            pre_compressed_multi = range_compressor_tensor(pre_multi)
            PSNR_u_multi = batch_PSNR(torch.clamp(pre_compressed_multi, 0., 1.), batch_x4_compressed, 1.)
            SSIM_u_multi = batch_SSIM(torch.clamp(pre_compressed_multi, 0., 1.), batch_x4_compressed, 1.)

            PSNR_final = batch_PSNR(torch.clamp(pre_final, 0., 1.), batch_x4, 1.)
            SSIM_final = batch_SSIM(torch.clamp(pre_final, 0., 1.), batch_x4, 1.)
            pre_compressed_final = range_compressor_tensor(pre_final)
            PSNR_u_final = batch_PSNR(torch.clamp(pre_compressed_final, 0., 1.), batch_x4_compressed, 1.)
            SSIM_u_final = batch_SSIM(torch.clamp(pre_compressed_final, 0., 1.), batch_x4_compressed, 1.)

            out_single = MPCL(pre_single)
            out_multi = MPCL(pre_multi)
            out_final = MPCL(pre_final)
            single_lable = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
            multi_lable = torch.ones(BATCH_SIZE, dtype=torch.long).to(device)
            total_lable = torch.full((BATCH_SIZE,), 2, dtype=torch.long).to(device)
            batch_x4_lable = torch.full((BATCH_SIZE,), 3, dtype=torch.long).to(device)
            batch_x4_lable_2 = torch.full((BATCH_SIZE,), 4, dtype=torch.long).to(device)

            batch_x4_gradient = gradient(batch_x4)
            loss_gradient = L1_loss(gradient_s_1, batch_x4_gradient) + L1_loss(gradient_m_1, batch_x4_gradient) + L1_loss(gradient_s_2, batch_x4_gradient) + L1_loss(gradient_m_2, batch_x4_gradient)

            loss_single = L1_loss(pre_compressed_single, batch_x4_compressed) + 0.1 * (1 - SSIM_u_single)
            loss_multi = L1_loss(pre_compressed_multi, batch_x4_compressed) + 0.1 * (1 - SSIM_u_multi)
            loss_final = L1_loss(pre_compressed_final, batch_x4_compressed) + 0.1 * (1 - SSIM_u_final)
            loss_re = loss_final + loss_single + loss_multi

            loss_single_MPCL = F.cross_entropy(out_single, batch_x4_lable) + F.cross_entropy(out_single, batch_x4_lable_2)
            loss_multi_MPCL = F.cross_entropy(out_multi, batch_x4_lable) + F.cross_entropy(out_multi, batch_x4_lable_2)
            loss_final_MPCL = F.cross_entropy(out_final, batch_x4_lable) + F.cross_entropy(out_final, batch_x4_lable_2)
            loss_classifier = loss_single_MPCL + loss_multi_MPCL + loss_final_MPCL

            print('loss_siongle={}, loss_multi={}, loss_final={}, loss_gradient={}, '.format(loss_single, loss_multi, loss_final, loss_gradient))

            total_loss_train = loss_re + 0.5 * loss_gradient + 0.0001 * loss_classifier

            with torch.autograd.set_detect_anomaly(True):
                total_loss_train.backward()
            optimizer_G.step()

            b = time.time()
            epoch_time = b - a

            counter += 1
        if (epoch % 100 == 0):
            torch.save(DTS.state_dict(), '../Model/%d.pkl' % (epoch))
        scheduler.step()
