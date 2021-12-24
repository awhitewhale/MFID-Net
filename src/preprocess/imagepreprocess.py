import os
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from natsort import ns, natsorted

def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' or '.JPG' or '.png' or '.PNG' in name]
    name_list = natsorted(name_list, alg=ns.PATH)
    return name_list

class StyleDataset(Dataset):
    def __init__(self, content1, information, size):
        self.content1_list = content1
        self.label = information
        self.size = 512
        self.len = len(self.content1_list)

    def __getitem__(self, index):
        c1_path = self.content1_list[index]
        f_path = self.label[index]
        content = cv2.imread(c1_path)[:, :, ::-1]
        information = cv2.imread(f_path)[:, :, ::-1]
        try:
            content = cv2.resize(content, (512, 512))
        except:
            content = cv2.resize(content, (512, 512))
        try:
            information = cv2.resize(information, (512, 512))
        except:
            information = cv2.resize(information, (512, 512))
        content = content.transpose((2, 0, 1)) / 255.0
        information = information.transpose((2, 0, 1)) / 255.0

        return content, information

    def __len__(self):
        return self.len

def style_loader(hazy_folder, gt_folder, size, batch_size):
    hazy_list = load_simple_list(hazy_folder)
    gt_list = load_simple_list(gt_folder)
    dataset = StyleDataset(hazy_list, gt_list, size)
    num_workers = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
    hazy_list1 = load_simple_list(ITS_hazy_train_folder)
    hazy_list2 = load_simple_list(ITS_hazy_val_folder)
    hazy_list3 = load_simple_list(OTS_hazy_folder)
    hazy_list = hazy_list1+hazy_list2+hazy_list3
    gt_list1 = load_simple_list(ITS_gt_train_folder)
    gt_list2 = load_simple_list(ITS_gt_val_folder)
    gt_list3 = load_simple_list(OTS_gt_folder)
    gt_list = gt_list1 * 10 + gt_list2 * 10 + gt_list3*35
    hazy_list = natsorted(hazy_list, alg=ns.PATH)
    gt_list = natsorted(gt_list, alg=ns.PATH)
    dataset = StyleDataset(hazy_list, gt_list, size)
    num_workers = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader



