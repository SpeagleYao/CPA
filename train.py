from models import *
# from loss import *
from img_aug import data_generator
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=2500, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=16, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率大小")
parser.add_argument("--version", type=int, default=1, help="学习率大小")
args = parser.parse_args()


def train(model):

    model = model.cuda()
    best_model = None
    best_epoch = 0
    best_loss = 99999.0
    vv = str(args.version)

    g_train = data_generator('./data/img_train.npy', './data/tar_train.npy', args.batch_size, train=True)
    # g_train = data_generator('./data/img_resample.npy', './data/tar_resample.npy', args.batch_size, train=True)
    g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', 200, train=False)
 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scaler = GradScaler()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True, threshold=1e-6)
    criterion = nn.MSELoss().cuda()
    # criterion = nn.L1Loss().cuda()
    # criterion = nn.HuberLoss().cuda()
    since = time.time()

    for epoch in tqdm(range(args.epoch)):
        model.train()
        img, tar = g_train.gen()
        img = img.cuda()
        tar = tar.cuda()

        optimizer.zero_grad()
        with autocast():
            out = model(img)
            loss = criterion(out, tar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.no_grad():
            img, tar = g_val.gen()
            img = img.cuda()
            tar = tar.cuda()
            out = model(img)
            loss_val = criterion(out, tar)

        if loss_val < best_loss:
            best_loss = loss_val
            best_model = model
            best_epoch = epoch
        
        # if epoch > 250:
        # scheduler.step(loss)

        if epoch == int(args.epoch * 0.6):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            tqdm.write("Learning rate reduced.")

        if epoch % 25 == 0:
            tqdm.write("Epoch:{0} || Loss_train:{1} Loss_val:{2}".format(epoch, format(loss, ".6f"), format(loss_val, ".6f")))
            # tqdm.write("Epoch:{0} || Loss_train:{1} Loss_val:{2}".format(epoch, format(loss, ".6f"), format(loss_val, ".6f")))
            # tqdm.write("Epoch:{0} || Loss_train:{1}".format(epoch, format(loss, ".6f")))

        # loss = loss.detach().cpu().numpy()
        # loss_val = loss_val.detach().cpu().numpy()
        # if not 'np_loss' in dir():
        #     np_loss = loss
        #     np_loss_val = loss_val
        # else:
        #     np_loss = np.append(np_loss, loss)
        #     np_loss_val = np.append(np_loss_val, loss_val)

        # if (optimizer.state_dict()['param_groups'][0]['lr'] < args.learning_rate/100): break

    model.eval()
    with torch.no_grad():
        best_model.eval()
        img, tar = g_val.gen()
        img = img.cuda()
        tar = tar.cuda()
        out = best_model(img)
        loss_val = criterion(out, tar)
    tqdm.write("Best Epoch:{0} || Loss_val:{1}".format(best_epoch, format(loss_val, ".6f")))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if not os.path.exists('./pth_BS16/'):
        os.mkdir('./pth_BS16/')
    # if not os.path.exists('./loss/'):
        # os.mkdir('./loss/')
    torch.save(best_model.state_dict(), "./pth_BS16/VGG16_" + vv + ".pth")
    # np.save('./loss/loss.npy', np_loss)
    # np.save('./loss/loss_val.npy', np_loss_val)

if __name__ == '__main__':
    model = VGG16()
    train(model)