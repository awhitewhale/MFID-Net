import torch
import torch.nn as nn
import torch.nn.functional as F

class image_h_attention(nn.Module):
    def __init__(self, in_features, ratios, K, temperature, init_weight=True):
        super(image_h_attention, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_features!=3:
            hidden_features = int(in_features*ratios)+1
        else:
            hidden_features = K
        self.fullyconnected1 = nn.Conv2d(in_features, hidden_features, 1, bias=False)
        self.fullyconnected2 = nn.Conv2d(hidden_features, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fullyconnected1(x)
        x = F.relu(x)
        x = self.fullyconnected2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_features%groups==0
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = image_h_attention(in_features, ratio, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_features, in_features//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_features))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_features, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_features, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_features, output.size(-2), output.size(-1))
        return output

class DyReLU(nn.Module):
    def __init__(self, in_features, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.in_features = in_features
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']
        self.fullyconnected1 = nn.Linear(in_features, in_features // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fullyconnected2 = nn.Linear(in_features // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def dynamicrelu(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta.unsqueeze(1)
        theta = self.fullyconnected1(theta)
        theta = self.relu(theta)
        theta = self.fullyconnected2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError

class DyReLU(DyReLU):
    def __init__(self, in_features, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__(in_features, reduction, k, conv_type)
        self.fullyconnected2 = nn.Linear(in_features // reduction, 2*k*in_features)

    def forward(self, x):
        assert x.shape[1] == self.in_features
        theta = self.dynamicrelu(x)
        relu_coefs = theta.view(-1, self.in_features, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result


class DoubleConv(nn.Module):
    def __init__(self, in_features, out_features, mid_features=None):
        super().__init__()
        if not mid_features:
            mid_features = out_features
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_features, out_features)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_features, out_features, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_features, out_features, in_features // 2)
        else:
            self.up = nn.ConvTranspose2d(in_features , in_features // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_features, out_features)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DRADNnn(nn.Module):
    def __init__(self, n_features, bilinear=True):
        super(DRADNnn, self).__init__()
        self.n_features = n_features
        self.bilinear = bilinear
        self.inc = DoubleConv(n_features, 64)
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

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class haze_strong_features(nn.Module):
    def __init__(self):
        super(haze_strong_features, self).__init__()

    def forward(self, input_hazy_features, haze_weak_features):
        device = input_hazy_features.get_device()
        N, _, H, W = haze_weak_features.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)
        hg, wg = hg * 2 - 1, wg * 2 - 1
        haze_weak_features = haze_weak_features.permute(0, 2, 3, 1).contiguous()
        haze_weak_features_guide = torch.cat([wg, hg, haze_weak_features], dim=3).unsqueeze(1)
        dyparameter = F.grid_sample(input_hazy_features, haze_weak_features_guide, align_corners=True)
        return dyparameter.squeeze(2)

class initialize_hazy_apply(nn.Module):
    def __init__(self):
        super(initialize_hazy_apply, self).__init__()
        self.degree = 3

    def forward(self, dyparameter, full_res_input):
        R = torch.sum(full_res_input * dyparameter[:, 0:3, :, :], dim=1, keepdim=True) + dyparameter[:, 3:4, :, :]
        G = torch.sum(full_res_input * dyparameter[:, 4:7, :, :], dim=1, keepdim=True) + dyparameter[:, 7:8, :, :]
        B = torch.sum(full_res_input * dyparameter[:, 8:11, :, :], dim=1, keepdim=True) + dyparameter[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)
        return result


class initialize_dynet(nn.Module):
    def __init__(self, bn=True):
        super(initialize_dynet, self).__init__()
        self.conv1 = ConvBlock(1, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = ConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)
        return output

class dynamic_transformer(nn.Module):
    def __init__(self):
        super(dynamic_transformer, self).__init__()
        self.initialize_dynet_r = initialize_dynet()
        self.initialize_dynet_g = initialize_dynet()
        self.initialize_dynet_b = initialize_dynet()
        self.haze_strong_features = haze_strong_features()
        self.apply_dyparameters = initialize_hazy_apply()
        self.DRADNnn = DRADNnn(n_features=3)
        self.DRADNnn_backup = DRADNnn(n_features=3)
        self.smooth = nn.PReLU()
        self.fusion = nn.Sequential(
            Dynamic_conv2d(in_features=9, out_features=16, kernel_size=3, stride=1, padding=1),
            DyReLU(in_features=16),
            Dynamic_conv2d(in_features=16, out_features=3, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

        self.features_fusion = nn.Sequential(
            Dynamic_conv2d(in_features=3, out_features=8, kernel_size=3, stride=1, padding=1),
            DyReLU(in_features=8),
            Dynamic_conv2d(in_features=8, out_features=3, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.downsample = nn.AdaptiveAvgPool2d((64, 256))
        self.p = nn.PReLU()
        self.r_channel_feature = Dynamic_conv2d(in_features=3, out_features=3, kernel_size=1, stride=1, padding=0)
        self.g_channel_feature = Dynamic_conv2d(in_features=3, out_features=3, kernel_size=1, stride=1, padding=0)
        self.b_channel_feature = Dynamic_conv2d(in_features=3, out_features=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        first_upsample = F.interpolate(x, (320, 320), mode='bicubic', align_corners=True)
        second_upsample = F.interpolate(x, (256, 256), mode='bicubic', align_corners=True)
        dyparameter = self.downsample(self.u_net(second_upsample)).reshape(-1, 12, 16, 16, 16)
        hazef_r = self.initialize_dynet_r(x[:, 0:1, :, :])
        hazef_g = self.initialize_dynet_g(x[:, 1:2, :, :])
        hazef_b = self.initialize_dynet_b(x[:, 2:3, :, :])
        haze_strong_features_dyparameters_r = self.haze_strong_features(dyparameter, hazef_r)
        haze_strong_features_dyparameters_g = self.haze_strong_features(dyparameter, hazef_g)
        haze_strong_features_dyparameters_b = self.haze_strong_features(dyparameter, hazef_b)
        first_upsample = self.u_net_mini(first_upsample)
        first_upsample = F.interpolate(first_upsample, (x.shape[2], x.shape[3]), mode='bicubic', align_corners=True)
        output_r = self.apply_dyparameters(haze_strong_features_dyparameters_r, self.p(self.r_channel_feature(first_upsample)))
        output_g = self.apply_dyparameters(haze_strong_features_dyparameters_g, self.p(self.g_channel_feature(first_upsample)))
        output_b = self.apply_dyparameters(haze_strong_features_dyparameters_b, self.p(self.b_channel_feature(first_upsample)))
        output = torch.cat((output_r, output_g, output_b), dim=1)
        output = self.fusion(output)
        output = self.p(self.features_fusion(output) * x - output + 1)
        return output
