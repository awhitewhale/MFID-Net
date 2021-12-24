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
modelname = 'dyrelu'
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = dyrelu.B_transformer().to(device)
my_model.eval()
my_model.to(device)

my_model.load_state_dict(torch.load("../../model/{}.pth".format(modelname)))
to_pil_image = transforms.ToPILImage()

tfs_full = transforms.Compose([
    # transforms.Resize(1080),
    transforms.ToTensor()
])


def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.png' or '.jpg' in name]
    name_list = natsorted(name_list, alg=ns.PATH)
    return name_list


list_s = load_simple_list('/home/liuyifan/project/TSIDNET/paper/hazy/')
sot_gt = load_simple_list('/home/liuyifan/project/TSIDNET/paper/gt/')
sot_gt = natsorted(sot_gt, alg=ns.PATH)
result = pd.DataFrame(columns=('idx', 'PSNR', 'SSIM'))

for idx in range(len(list_s)):
    image_in = Image.open(list_s[idx]).convert('RGB')
    full = tfs_full(image_in).unsqueeze(0).to(device)
    output = my_model(full)
    save_image(output[0], '/home/liuyifan/project/TSIDNET/paper/{}out/{}.png'.format(modelname,idx))
    PSNR = peak_signal_noise_ratio(cv2.imread('/home/liuyifan/project/TSIDNET/paper/{}out/{}.png'.format(modelname,idx)), cv2.imread(sot_gt[idx]))
    SSIM = structural_similarity(cv2.imread('/home/liuyifan/project/TSIDNET/paper/{}out/{}.png'.format(modelname,idx)), cv2.imread(sot_gt[idx]), multichannel=True)
    df = [idx, PSNR, SSIM]
    print(df)
