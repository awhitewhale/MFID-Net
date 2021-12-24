import os
from torchvision import transforms
from PIL import Image
import torch
from natsort import ns, natsorted
from torchvision.utils import save_image
import pandas as pd
from src.model import dehaze4K
from src.model import doconv
from src.model import dyconv
from src.model import dyrelu
from src.model import dyConvdyrelu
model_this_time = 'DRADN'
print(model_this_time)
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 3, 4, 10, 11, 12, 13, 14, 15, 16'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

temporary_DRADN = eval(model_this_time).dynamic_transformer().to(device)
temporary_DRADN.eval()
temporary_DRADN.to(device)
temporary_DRADN.load_state_dict(torch.load("../../checkpoint/DRADN.pth"))
input_image = transforms.Compose([transforms.ToTensor()])


def load_dataset_pathlist(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.png' in name]
    name_list = natsorted(name_list, alg=ns.PATH)
    return name_list


list_s = load_dataset_pathlist('../../dataset/Haze4K/test/haze/')
list_g = load_dataset_pathlist('../../dataset/Haze4K/test/gt/')
result = pd.DataFrame(columns=('INDEX', 'PSNR', 'SSIM'))
for idx in range(len(list_s)):
    hazy_image = Image.open(list_s[idx]).convert('RGB')
    my_little_pony = input_image(hazy_image).unsqueeze(0).to(device)
    output = temporary_DRADN(my_little_pony)
    save_image(output[0], '../../result/{}.png'.format(idx))
    PSNR = peak_signal_noise_ratio(cv2.imread('../../result/{}.png'.format(idx)), cv2.imread(list_g[idx]))
    SSIM = structural_similarity(cv2.imread('../../result/{}.png'.format(idx)), cv2.imread(list_g[idx]), multichannel=True)
    df = [idx + 1, PSNR, SSIM]
    result.loc[idx] = df
    result.to_csv(model_this_time + '_TEST_result.csv')
    print('{}LEFT'.format(len(list_s)-idx))
    print(list_g[idx])
    print('PSNR:{},SSIM:{}'.format(PSNR,SSIM))