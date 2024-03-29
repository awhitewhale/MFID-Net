import sys
import os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/.."+"/..")
import torch
#from src.model import dyconv
#from src.model import doconv
#from src.model import dyrelu
#from src.model import dehaze4K
#from src.model import dyConvdyrelu
from src.model import DRAGN
#from src.model import dyL1Loss
model_this_time = 'DRAGN'
print(model_this_time)
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 3, 4, 10, 11, 12, 13, 14, 15, 16'
from src.preprocess import imagepreprocess
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torchvision.utils import save_image
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2

def train(args):
    result = pd.DataFrame(columns=('idx', 'epoch', 'total_iter', 'loss', 'avgloss', 'PSNR', 'avgPSNR', 'SSIM', 'avgSSIM'))
    avgloss, avgPSNR, avgSSIM, f,s = 0, 0, 0, 0, 1
    model = eval(model_this_time).dynamic_transformer()
    model = nn.DataParallel(model, device_ids=[0, 1, 3, 4, 10, 11, 12, 13, 14, 15, 16])
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    #dyloss = dyL1Loss().cuda()
    loss = nn.L1Loss().cuda()
    hazy_folder1 = '/home/data/RESIDE/ITS/train/ITS_haze/'
    gt_folder = '/home/data/RESIDE/ITS/train/ITS_clear/'
    ITS_hazy_train_folder = '/home/data/RESIDE/ITS/train/ITS_haze/'
    ITS_gt_train_folder = '/home/data/RESIDE/ITS/train/ITS_clear/'
    ITS_hazy_val_folder = '/home/data/RESIDE/ITS/val/haze/'
    ITS_gt_val_folder = '/home/data/RESIDE/ITS/val/clear/'
    OTS_hazy_folder = '/home/data/RESIDE/OTS/'
    OTS_gt_folder = '/home/data/RESIDE/clear_images/'
    train_loader= imagepreprocess.reside_dataset_loader(ITS_hazy_train_folder, ITS_gt_train_folder, ITS_hazy_val_folder, ITS_gt_val_folder,OTS_hazy_folder, OTS_gt_folder,args.size, args.batch_size)
    # train_loader = imagepreprocess.Data_Loader(hazy_folder1, gt_folder, args.size, args.batch_size)
    num_batch = len(train_loader)
    for epoch in range(args.epoch):
        for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            total_iter = epoch*num_batch + idx
            hazy = batch[f].float().cuda()
            gt = batch[s].float().cuda()
            optimizer.zero_grad()
            output = model(hazy)
            total_loss = dyloss(output, gt)
            total_loss.backward()
            optimizer.step()
            save_image(output[f],'output.jpg')
            save_image(gt[f], 'gt.jpg')
            PSNR = peak_signal_noise_ratio(cv2.imread('output.jpg'), cv2.imread('gt.jpg'))
            avgPSNR = (avgPSNR + PSNR)/(2)
            SSIM = structural_similarity(cv2.imread('output.jpg'), cv2.imread('gt.jpg'), multichannel=True)
            avgSSIM = (avgSSIM + SSIM)/(2)
            avgloss = (avgloss + total_loss.item()) / (2)
            df = [idx+1, epoch+1, total_iter+1, total_loss.item(), avgloss, PSNR, avgPSNR, SSIM, avgSSIM]
            result.loc[total_iter] = df
            result.to_csv('result.csv')
            if np.mod(total_iter+1, 1) == 0:
                pass
                print('{}\nEpoch:{}, Iter:{}\n total loss: {}'.format(args.save_dir, epoch, total_iter, total_loss.item()))
            if not os.path.exists(args.save_dir+'/image'):
                os.mkdir(args.save_dir+'/image')
            pass
        if epoch % 20 ==0:
            torch.save(model.state_dict(), 'model/' +model_this_time+'_{}.pth'.format(epoch))
    result.to_csv('result.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=7, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    train(args)
