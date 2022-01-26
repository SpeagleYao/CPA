from models import *
# from loss import *
from img_aug import data_generator
import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import VGG
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1000, help="训练迭代次数")
parser.add_argument("--batch_size", type=int, default=16, help="批训练大小")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="学习率大小")
args = parser.parse_args()

def train(model):

    model = model.cuda()
    best_model = None
    best_epoch = 0
    best_val = 100.0

    g_train = data_generator('./data/img_train.npy', './data/tar_train.npy', args.batch_size, train=True)
    g_val = data_generator('./data/img_val.npy', './data/tar_val.npy', args.batch_size, train=False)
 
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=100, verbose=True)
    criterion = nn.MSELoss().cuda()
    
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
        img, tar = g_val.gen()
        img = img.cuda()
        tar = tar.cuda()
        out = model(img)
        loss_val = criterion(out, tar)
        if loss_val + loss < best_val and epoch > 50:
            best_val = loss_val + loss
            best_model = model
            best_epoch = epoch
        if epoch > 100:
            scheduler.step(loss_val)

        if epoch % 20 == 0:
            tqdm.write("Epoch:{0} || Loss_train:{1}".format(epoch, format(loss, ".4f")))
            tqdm.write("Epoch:{0} || Loss_val:{1}".format(epoch, format(loss_val, ".4f")))

        if (optimizer.state_dict()['param_groups'][0]['lr'] < args.learning_rate/100): break

    best_model.eval()
    img, tar = g_val.gen()
    img = img.cuda()
    tar = tar.cuda()
    out = best_model(img)
    loss_val = criterion(out, tar)
    tqdm.write("Best Epoch:{0} || Loss_val:{1}".format(best_epoch, format(loss_val, ".4f")))
    torch.save(best_model.state_dict(), "./pth/VGG16.pth")

if __name__ == '__main__':
    model = VGG16()
    train(model)
