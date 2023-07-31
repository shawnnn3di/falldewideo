import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torch.optim import Adam, lr_scheduler


resizer = Resize((32, 64))
def forward_func(batch, device, model, half, resizer=resizer):
        
    x = batch['csi'].to(device)
    x = resizer(x)
    
    y_sm = batch['mask'].to(device)
    y_jhm = batch['kpt'].to(device)
    y_paf = batch['aff'].to(device)

    if not half:
        x = x.float()
        y_sm = y_sm.float()
        y_jhm = batch['kpt'].to(device).float()
        y_paf = batch['aff'].to(device).float()
        
    y_sm = resizer(y_sm)
    y_jhm = resizer(y_jhm) * y_sm.unsqueeze(1)
    y_paf = resizer(y_paf) * y_sm.unsqueeze(1)
    
    paf = resizer(model[0](x))
    jhm = resizer(model[1](x))
    sm = resizer(paf[:, :18, ...].sum(1))
    
    loss_sm = (
        ((sm - y_sm) ** 2 * (1 + 1 * y_sm.abs())).sum() * 0.05
    )
    loss_jhm = ((jhm - y_jhm) ** 2 * (1 + 1 * y_jhm.abs())).sum()
    loss_paf = ((paf - y_paf) ** 2 * (1 + 0.3 * y_paf.abs())).sum()
    
    return loss_sm, loss_jhm, loss_paf, sm, jhm, paf, y_sm, y_jhm, y_paf, batch['img']


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=19):
        super().__init__()
        
        c = [2 ** _ for _ in range(4, 13)]
        
        self.startconv = double_conv(in_channels, c[0])
        
        self.dconv_down0 = double_conv(c[0], c[1])
        self.dconv_down1 = double_conv(c[1], c[2])
        self.dconv_down2 = double_conv(c[2], c[3])
        self.dconv_down3 = double_conv(c[3], c[4])
        self.dconv_down4 = double_conv(c[4], c[5])
        
        self.avgpool = nn.AvgPool2d(2)
        self.upsample4 = nn.ConvTranspose2d(c[5], c[4], 3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(c[4], c[3], 3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(c[3], c[2], 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(c[2], c[1], 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(c[1], c[0], 3, stride=2, padding=1, output_padding=1)
        
        self.dconv_up3 = double_conv(c[5], c[4])
        self.dconv_up2 = double_conv(c[4], c[3])
        self.dconv_up1 = double_conv(c[3], c[2])
        self.dconv_up0 = double_conv(c[2], c[1])
        
        self.endconv = nn.Conv2d(c[0], c[0], 1)
        
        self.postprocess_0 = nn.Sequential(
            nn.Conv2d(c[0], c[0], 1, 1),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(),
            nn.Conv2d(c[0], out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.startconv(x)
        
        # encode
        conv0 = self.dconv_down0(x)
        x = self.avgpool(conv0)
        
        conv1 = self.dconv_down1(x)
        x = self.avgpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.avgpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.avgpool(conv3)

        x = self.dconv_down4(x)
        
        # decode
        x = self.upsample4(x)

        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv0], dim=1)

        x = self.dconv_up0(x)
        x = self.upsample0(x)

        out = self.endconv(x)
        
        out = self.postprocess_0(out)
        
        return out


def build_model(device, checkpoint=None, lr=1e-4, wd=1e-5, gamma=0.99):
    nview = 1
    
    if checkpoint is None:
        print('initializing new model..')
        student_jhm = unet(9 * 2 * 3, 19 * nview).to(device)
        student_paf = unet(9 * 2 * 3, 38 * nview).to(device)
    
    else:
        print('loading checkpoint from %s' % checkpoint)
        [student_jhm, student_paf] = [_.to(device) for _ in torch.load(checkpoint)]
            
    model = [student_jhm, student_paf]
    optimizer = Adam(
        [
            {'params': student_jhm.parameters()},
            {'params': student_paf.parameters()},
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