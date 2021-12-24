import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from natsort import ns, natsorted
'''
读取文件夹内所有图片，并以列表形式返回，通过natsort排序为windows下的排序方法而不是linux
'''
def load_dataset_pathlist(src_path):
    name_list = []
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = natsorted([name for name in name_list if '.jpg' or '.JPG' or '.png' or '.PNG' in name], alg=ns.PATH)
    return name_list

class DehazeDataset(Dataset):
    def __init__(self, hazy, gt, size):
        self.hazy_list = hazy
        self.gt = gt
        self.size = 512
        self.len = len(self.hazy_list)
    def __getitem__(self, index):
        hazy_path = self.hazy_list[index]
        gt_path = self.gt[index]
        hazy = cv2.imread(hazy_path)[:, :, ::-1]
        gt = cv2.imread(gt_path)[:, :, ::-1]
        hazy = cv2.resize(hazy, (512, 512))
        gt = cv2.resize(gt, (512, 512))
        hazy = hazy.transpose((2, 0, 1)) / 255.0
        gt = gt.transpose((2, 0, 1)) / 255.0
        return hazy, gt
    def __len__(self):
        return self.len

def Data_Loader(hazy_folder, gt_folder, size, batch_size):
    hazy_list = load_dataset_pathlist(hazy_folder)
    gt_list = load_dataset_pathlist(gt_folder)
    dataset = DehazeDataset(hazy_list, gt_list, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def reside_dataset_loader(ITS_hazy_train_folder, ITS_gt_train_folder, ITS_hazy_val_folder, ITS_gt_val_folder, OTS_hazy_folder, OTS_gt_folder, size, batch_size):
    '''
    :param ITS_hazy_train_folder:
    :param ITS_gt_train_folder:
    :param ITS_hazy_val_folder:
    :param ITS_gt_val_folder:
    :param OTS_hazy_train_folder:
    :param OTS_gt_train_folder:
    :param OTS_hazy_val_folder:
    :param OTS_gt_val_folder:
    :param size:
    :param batch_size:
    :return:
    '''
    hazy_list1 = load_dataset_pathlist(ITS_hazy_train_folder)
    hazy_list2 = load_dataset_pathlist(ITS_hazy_val_folder)
    hazy_list3 = load_dataset_pathlist(OTS_hazy_folder)
    hazy_list = hazy_list1+hazy_list2+hazy_list3
    gt_list1 = load_dataset_pathlist(ITS_gt_train_folder)
    gt_list2 = load_dataset_pathlist(ITS_gt_val_folder)
    gt_list3 = load_dataset_pathlist(OTS_gt_folder)
    gt_list = gt_list1 * 10 + gt_list2 * 10 + gt_list3*35
    hazy_list = natsorted(hazy_list, alg=ns.PATH)
    gt_list = natsorted(gt_list, alg=ns.PATH)
    dataset = DehazeDataset(hazy_list, gt_list, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



