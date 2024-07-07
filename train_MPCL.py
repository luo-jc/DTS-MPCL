import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import *
from datsetprocess import *
from DTS import *
from MPCL import *
import random
import albumentations
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    Initial_LR = 0.001
    lambda1 = 1e-4
    transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
    ])

    train_data = MyDataset(scene_directory='../Training', transforms=transforms, noise=None, is_training=True)  # 有getitem变成可迭代的对象
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    generator = DTS().to(device)
    generator.eval()
    state_dict = torch.load('../Model/2000.pkl')

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    generator.load_state_dict(new_state_dict)
    generator = nn.DataParallel(generator, device_ids=[0, 1])

    MPCL = MPCL().to(device)
    MPCL = nn.DataParallel(MPCL, device_ids=[0, 1])
    MPCL.apply(weights_init_normal)

    optimizer_C = torch.optim.Adam(MPCL.parameters(), lr=Initial_LR)
    scheduler = WarmupMultiStepLR(optimizer=optimizer_C, milestones=(150, 250), gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=20, warmup_method="linear")
    counter = 0
    counter_test = 0
    Tensor = torch.cuda.FloatTensor

# ----------
#  Training
# ----------
    for epoch in range(0, 201):
        for step, sample in enumerate(train_loader, 0):
            a = time.time()
            batch_x1, batch_x2, batch_x3, batch_x4 = sample['input1'], sample['input2'], sample['input3'], sample['label']
            batch_x1, batch_x2, batch_x3, batch_x4 = batch_x1.to(device), batch_x2.to(device), batch_x3.to(device), batch_x4.to(device)

            optimizer_C.zero_grad()

            pre_single, pre_multi, out_all, gradient_s_1, gradient_m_1, gradient_s_2, gradient_m_2 = generator(x1=batch_x1, x2=batch_x2, x3=batch_x3)
            out_x4 = MPCL(batch_x4)
            out_single = MPCL(pre_single)
            out_multi = MPCL(pre_multi)
            out_total = MPCL(out_all)

            weight = MPCL.state_dict()['module.liner_5.weight']
            w_g = weight[3, :]
            w_t = weight[4, :]

            single_lable = torch.zeros(BATCH_SIZE, dtype=torch.long).to(device)
            multi_lable = torch.ones(BATCH_SIZE, dtype=torch.long).to(device)
            total_lable = torch.full((BATCH_SIZE,), 2, dtype=torch.long).to(device)
            batch_x4_lable = torch.full((BATCH_SIZE,), 3, dtype=torch.long).to(device)
            batch_x4_lable_2 = torch.full((BATCH_SIZE,), 4, dtype=torch.long).to(device)

            similarity = torch.cosine_similarity(w_g, w_t, dim=0)

            loss_x4 = F.cross_entropy(out_x4, batch_x4_lable) + F.cross_entropy(out_x4, batch_x4_lable_2)
            loss_single = F.cross_entropy(out_single, single_lable)
            loss_multi = F.cross_entropy(out_multi, multi_lable)
            loss_total = F.cross_entropy(out_total, total_lable)
            loss_clc = loss_total + loss_single + loss_multi + loss_x4

            total_loss_train = loss_clc + similarity

            print("epoch:{} ;loss:{} ;".format(epoch, total_loss_train))
            print('')

            with torch.autograd.set_detect_anomaly(True):
                total_loss_train.backward()
            optimizer_C.step()

            b = time.time()
            epoch_time = b - a

            counter += 1

        if (epoch % 100 == 0):
            torch.save(MPCL.state_dict(), '../Model_MPCL/%d.pkl' % (epoch))
        scheduler.step()