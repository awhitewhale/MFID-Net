import os
import cv2
import numpy as np
import yaml
from yaml import Loader
from natsort import ns, natsorted
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import joblib

class Haze4KProcess(Dataset):
    def __init__(self, config):
        '''
        确定Haze4K数据集存放位置
        '''
        if not (os.path.exists(config['haze_dir']) or os.path.exists(config['gt_dir'])):
            raise Exception('{} or {}is not exists'.format(config['haze_dir'], config['gt_dir']))
        self.haze_dir = config['haze_dir']
        self.gt_dir = config['gt_dir']
        haze_files = []
        for path, dir_list, file_list in os.walk(self.haze_dir):
            for file_name in file_list:
                if '.DS_Store' not in file_name:
                    haze_files.append(os.path.join(path, file_name))
        gt_files = []
        for path, dir_list, file_list in os.walk(self.gt_dir):
            for file_name in file_list:
                if '.DS_Store' not in file_name:
                    gt_files.append(os.path.join(path, file_name))
        self.haze_files = natsorted(haze_files, alg=ns.PATH) #按windows资源管理器的排序顺序进行排序
        self.gt_files = natsorted(gt_files, alg=ns.PATH)
        self.img_size = config['img_size']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.len = len(haze_files)

    def __getitem__(self, key):
        '''
        建立索引并归一化
        :param key:
        :return:
        '''
        one_haze_pic = cv2.resize(cv2.imread(self.haze_files[key]), (self.img_size, self.img_size))
        one_gt_pic = cv2.resize(cv2.imread(self.gt_files[key]), (self.img_size, self.img_size))
        normalized_haze = (one_haze_pic - np.min(one_haze_pic)) / (np.max(one_haze_pic) - np.min(one_haze_pic))
        normalized_gt = (one_gt_pic - np.min(one_gt_pic)) / (np.max(one_gt_pic) - np.min(one_gt_pic))
        return [normalized_haze, normalized_gt]

    def __len__(self):
        return self.len

    def LoadData(self):
        dataset = Haze4KProcess(config["Haze4KConfig"])
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_dataloader

if __name__ == "__main__":
    with open("../../config/config.yaml",encoding="utf-8") as f:
        file_data = f.read()
        config = yaml.load(file_data, Loader=Loader)
        dataset = Haze4KProcess(config["Haze4KConfig"])
        joblib.dump(dataset, 'dataset.pkl')

