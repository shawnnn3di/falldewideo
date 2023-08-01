import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torch.optim import Adam, lr_scheduler


resizer = nn.AdaptiveAvgPool3d((10, 32, 64))
def forward_func(batch, device, model, half, resizer=resizer):
        
    x = batch['csi'].to(device)
    x = resizer(x.permute(0, 2, 1, 3, 4))
    
    y_sm = resizer(batch['mask'].float().to(device))
    y_jhm = resizer(batch['aff'].to(device).permute(0, 2, 1, 3, 4))
    y_paf = resizer(batch['kpt'].to(device).permute(0, 2, 1, 3, 4))

    if not half:
        x = x.float()
        y_sm = y_sm.float().to(device)
        y_jhm = batch['aff'].to(device).float().permute(0, 2, 1, 3, 4)
        y_paf = batch['kpt'].to(device).float().permute(0, 2, 1, 3, 4)
        
    y_sm = resizer(y_sm)
    y_jhm = resizer(y_jhm) * y_sm.unsqueeze(1)
    y_paf = resizer(y_paf) * y_sm.unsqueeze(1)
    
    jhm, paf = model(x)
    jhm = resizer(jhm)
    paf = resizer(paf)
    sm = resizer(paf[:, :18, ...].sum(1))
    
    loss_sm = (
        ((sm - y_sm) ** 2 * (1 + 1 * y_sm.abs())).sum() * 0.05
    )
    loss_jhm = ((jhm - y_jhm) ** 2 * (1 + 1 * y_jhm.abs())).sum()
    loss_paf = ((paf - y_paf) ** 2 * (1 + 0.3 * y_paf.abs())).sum()
    
    return loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, batch['img']


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    

class bi_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels_0=19, out_channels_1=38):
        super().__init__()
        
        c = [2 ** _ for _ in range(5, 12)]
        
        self.startconv = double_conv(in_channels, c[0])
        
        self.dconv_down0_0 = double_conv(c[0], c[1])
        self.dconv_down1_0 = double_conv(c[1], c[2])
        self.dconv_down2_0 = double_conv(c[2], c[3])
        self.dconv_down3_0 = double_conv(c[3], c[4])
        self.dconv_down4_0 = double_conv(c[4], c[5])
        
        self.dconv_down0_1 = double_conv(c[0], c[1])
        self.dconv_down1_1 = double_conv(c[1], c[2])
        self.dconv_down2_1 = double_conv(c[2], c[3])
        self.dconv_down3_1 = double_conv(c[3], c[4])
        self.dconv_down4_1 = double_conv(c[4], c[5])
        
        self.avgpool = nn.AvgPool3d((2, 2, 2))
        
        self.upsample4_0 = nn.ConvTranspose3d(c[5], c[4], 3, stride=2, padding=1, output_padding=1)
        self.upsample3_0 = nn.ConvTranspose3d(c[4], c[3], 3, stride=2, padding=1, output_padding=1)
        self.upsample2_0 = nn.ConvTranspose3d(c[3], c[2], 3, stride=2, padding=1, output_padding=1)
        self.upsample1_0 = nn.ConvTranspose3d(c[2], c[1], 3, stride=2, padding=1, output_padding=1)
        self.upsample0_0 = nn.ConvTranspose3d(c[1], c[0], 3, stride=2, padding=1, output_padding=1)
        
        self.upsample4_1 = nn.ConvTranspose3d(c[5], c[4], 3, stride=2, padding=1, output_padding=1)
        self.upsample3_1 = nn.ConvTranspose3d(c[4], c[3], 3, stride=2, padding=1, output_padding=1)
        self.upsample2_1 = nn.ConvTranspose3d(c[3], c[2], 3, stride=2, padding=1, output_padding=1)
        self.upsample1_1 = nn.ConvTranspose3d(c[2], c[1], 3, stride=2, padding=1, output_padding=1)
        self.upsample0_1 = nn.ConvTranspose3d(c[1], c[0], 3, stride=2, padding=1, output_padding=1)
        
        self.dconv_up3_0 = double_conv(c[5], c[4])
        self.dconv_up2_0 = double_conv(c[4], c[3])
        self.dconv_up1_0 = double_conv(c[3], c[2])
        self.dconv_up0_0 = double_conv(c[2], c[1])
        
        self.dconv_up3_1 = double_conv(c[5], c[4])
        self.dconv_up2_1 = double_conv(c[4], c[3])
        self.dconv_up1_1 = double_conv(c[3], c[2])
        self.dconv_up0_1 = double_conv(c[2], c[1])
        
        self.endconv_0 = nn.Conv3d(c[0], c[0], 1)
        self.endconv_1 = nn.Conv3d(c[0], c[0], 1)
        
        # self.resizer1 = nn.AdaptiveAvgPool3d((10, 72, 128))
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv3d(c[0], c[0], 1, 1),
            nn.BatchNorm3d(c[0]),
            nn.ReLU(),
            nn.Conv3d(c[0], out_channels_0, 1, 1),
            nn.BatchNorm3d(out_channels_0),
            nn.ReLU(),
        )
        
        self.postprocess_1 = nn.Sequential(
            nn.Conv3d(c[0], c[0], 1, 1),
            nn.BatchNorm3d(c[0]),
            nn.ReLU(),
            nn.Conv3d(c[0], out_channels_1, 1, 1),
            nn.BatchNorm3d(out_channels_1),
            nn.ReLU(),
        )

        self.aa = nn.AdaptiveAvgPool3d((16, 64, 128))
        
    def forward(self, x):
        x = self.aa(x)
        
        x = self.startconv(x)
        
        # encode
        conv0_0 = self.dconv_down0_0(x)
        x_0 = self.avgpool(conv0_0)
        
        conv1_0 = self.dconv_down1_0(x_0)
        x_0 = self.avgpool(conv1_0)
        
        conv2_0 = self.dconv_down2_0(x_0)
        x_0 = self.avgpool(conv2_0)

        conv3_0 = self.dconv_down3_0(x_0)
        x_0 = self.avgpool(conv3_0)

        x_0 = self.dconv_down4_0(x_0)
        
        # encode
        conv0_1 = self.dconv_down0_1(x)
        x_1 = self.avgpool(conv0_1)
        
        conv1_1 = self.dconv_down1_1(x_1)
        x_1 = self.avgpool(conv1_1)
        
        conv2_1 = self.dconv_down2_1(x_1)
        x_1= self.avgpool(conv2_1)

        conv3_1 = self.dconv_down3_1(x_1)
        x_1 = self.avgpool(conv3_1)

        x_1 = self.dconv_down4_1(x_1)
        
        # decode_0
        x_0 = self.upsample4_0(x_0)
        x_0 = torch.cat([x_0, conv3_0], dim=1)

        x_0 = self.dconv_up3_0(x_0)
        x_0 = self.upsample3_0(x_0)
        x_0 = torch.cat([x_0, conv2_0], dim=1)

        x_0 = self.dconv_up2_0(x_0)
        x_0 = self.upsample2_0(x_0)
        x_0 = torch.cat([x_0, conv1_0], dim=1)

        x_0 = self.dconv_up1_0(x_0)
        x_0 = self.upsample1_0(x_0)
        x_0 = torch.cat([x_0, conv0_0], dim=1)

        x_0 = self.dconv_up0_0(x_0)
        x_0 = self.upsample0_0(x_0)

        out_0 = self.endconv_0(x_0)
        
        out_0 = self.postprocess_0(out_0)
        # out_0 = self.resizer1(out_0)
        
        # decode_1
        x_1 = self.upsample4_0(x_1)
        x_1 = torch.cat([x_1, conv3_1], dim=1)
        
        x_1 = self.dconv_up3_1(x_1)
        x_1 = self.upsample3_1(x_1)
        x_1 = torch.cat([x_1, conv2_1], dim=1)

        x_1 = self.dconv_up2_1(x_1)
        x_1 = self.upsample2_1(x_1)
        x_1 = torch.cat([x_1, conv1_1], dim=1)

        x_1 = self.dconv_up1_1(x_1)
        x_1 = self.upsample1_1(x_1)
        x_1 = torch.cat([x_1, conv0_1], dim=1)

        x_1 = self.dconv_up0_1(x_1)
        x_1 = self.upsample0_1(x_1)

        out_1 = self.endconv_1(x_1)
        
        out_1 = self.postprocess_1(out_1)
        # out_1 = self.resizer1(out_1)
        
        return out_0, out_1
    

def build_model(device, checkpoint=None, lr=1e-4, wd=1e-5, gamma=0.99):
    nview = 1
    
    if checkpoint is None:
        print('initializing new model..')
        net = bi_unet(9 * 2 * 3, 19 * nview).to(device)
    
    else:
        print('loading checkpoint from %s' % checkpoint)
        net = torch.load(checkpoint).to(device)
            
    model = net
    optimizer = Adam(
        [
            {'params': net.parameters()},
        ],
        lr=lr,
        weight_decay=wd,
    )
    
    scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer, 
        gamma=0.99,
    )
    forward = forward_func
        
    return model, optimizer, scheduler, forward