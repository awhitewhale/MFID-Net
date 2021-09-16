import torch
import torch.nn.functional as F
from torch import nn, einsum
from src.preprocess.image_preprocess import Haze4KProcess
from torch.nn.modules.activation import PReLU, Sigmoid
import yaml
from yaml import Loader

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.pre = nn.Conv2d(64, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))
        return x

class DynamicAttention2d(nn.Module):
    def __init__(self, in_channels, K):
        with open("../../config/config.yaml", encoding="utf-8") as f:
            file_data = f.read()
            config = yaml.load(file_data, Loader=Loader)
        super(DynamicAttention2d, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(config['DynamicAttention2d']['adaptive_avg_pool'])
        self.fc1 = nn.Conv2d(in_channels, K, config['DynamicAttention2d']['fc1'])
        self.fc2 = nn.Conv2d(K, K, config['DynamicAttention2d']['fc2'])
    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x, 1)

class DynamicConvBlock(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation, groups=1, bias, K, activation, batch_norm):
    def __init__(self, in_channels, out_channels):
        with open("../../config/config.yaml", encoding="utf-8") as f:
            file_data = f.read()
            config = yaml.load(file_data, Loader=Loader)
        super(DynamicConvBlock, self).__init__()
        self.groups = config['DynamicConvBlock']['groups']
        assert in_channels % self.groups == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = config['DynamicConvBlock']['kernel_size']
        self.stride = config['DynamicConvBlock']['stride']
        self.padding = config['DynamicConvBlock']['padding']
        self.dilation = config['DynamicConvBlock']['dilation']
        self.bias = config['DynamicConvBlock']['bias']
        self.K = config['DynamicConvBlock']['K']
        self.batch_norm = config['DynamicConvBlock']['batch_norm']
        self.attention = DynamicAttention2d(in_channels, self.K)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), self.kernel_size, padding=self.padding, stride=self.stride, bias=self.bias)
        self.activation = DynamicAttention2d() if self.activation else None
        self.bn = nn.BatchNorm2d(out_channels) if self.batch_norm else None
        self.weight = nn.Parameter(torch.Tensor(self.K, out_channels, in_channels//self.groups, self.kernel_size, self.kernel_size), requires_grad=True)
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.K, out_channels))
        else:
            self.bias = None

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)
        return result

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = DynamicConvBlock(1, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = DynamicConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = DynamicConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)

        return output


class B_transformer(nn.Module):
    def __init__(self):
        super(B_transformer, self).__init__()

        self.guide_r = GuideNN()
        self.guide_g = GuideNN()
        self.guide_b = GuideNN()

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.u_net = UNet(n_channels=3)
        self.u_net_mini = UNet(n_channels=3)
        # self.u_net_mini = UNet_mini(n_channels=3)
        self.smooth = nn.PReLU()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

        self.x_r_fusion = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.downsample = nn.AdaptiveAvgPool2d((64, 256))
        self.p = nn.PReLU()

        self.r_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.g_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.b_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_u = F.interpolate(x, (320, 320), mode='bicubic', align_corners=True)

        x_r = F.interpolate(x, (256, 256), mode='bicubic', align_corners=True)
        coeff = self.downsample(self.u_net(x_r)).reshape(-1, 12, 16, 16, 16)

        guidance_r = self.guide_r(x[:, 0:1, :, :])
        guidance_g = self.guide_g(x[:, 1:2, :, :])
        guidance_b = self.guide_b(x[:, 2:3, :, :])

        slice_coeffs_r = self.slice(coeff, guidance_r)
        slice_coeffs_g = self.slice(coeff, guidance_g)
        slice_coeffs_b = self.slice(coeff, guidance_b)

        x_u = self.u_net_mini(x_u)
        x_u = F.interpolate(x_u, (x.shape[2], x.shape[3]), mode='bicubic', align_corners=True)

        output_r = self.apply_coeffs(slice_coeffs_r, self.p(self.r_point(x_u)))
        output_g = self.apply_coeffs(slice_coeffs_g, self.p(self.g_point(x_u)))
        output_b = self.apply_coeffs(slice_coeffs_b, self.p(self.b_point(x_u)))

        output = torch.cat((output_r, output_g, output_b), dim=1)
        output = self.fusion(output)
        output = self.p(self.x_r_fusion(output) * x - output + 1)

        return output