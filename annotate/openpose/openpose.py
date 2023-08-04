# %%
lookto = 'annotate/csi/data/full.pk'

import pickle as pk
import pandas as pd
from torchvision.transforms import Resize

df = pk.load(open(lookto, 'rb'))

# %%
import cv2
import matplotlib.pyplot as plt

# %%
openposepath = 'external/pytorch-openpose'

import sys
sys.path.append(openposepath)
from src import util

import cv2
import torch
import numpy as np

class prep:
    def __init__(self):
        self.scale_search = [.5,]       # scale=1 will run out of memory
        self.boxsize = 1080
        self.stride = 8
        self.padValue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
    
    def __call__(self, oriImg):
        self.multiplier = [x * self.boxsize / oriImg.shape[0] for x in self.scale_search]
        self.heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        self.paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(len(self.multiplier)):
            scale = self.multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.stride, self.padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()

        return data


# %%
from torch.utils.data import Dataset, DataLoader

import cv2
import torch
import tqdm
import numpy as np

class prep:
    def __init__(self):
        self.scale_search = [.5,]       # scale=1 will run out of memory
        self.boxsize = 1080
        self.stride = 8
        self.padValue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05
    
    def __call__(self, oriImg):
        self.multiplier = [x * self.boxsize / oriImg.shape[0] for x in self.scale_search]
        self.heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        self.paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(len(self.multiplier)):
            scale = self.multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.stride, self.padValue)
            im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()

        return data
    

prepper = prep()
class _dataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, index):
        return self.df.iloc[index]['pic']
    
    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    imgs = [prepper(cv2.imread(_)) for _ in batch]
    paths = [_ for _ in batch]
    return np.stack(imgs), paths
    

from src.model import bodypose_model

print('setting cuda devices..')
gpuid = 1
device = 'cuda:%d' % gpuid

print('waking up openpose...')
model_path = openposepath + '/model/body_pose_model.pth'
teacher = bodypose_model()
model_dict = util.transfer(teacher, torch.load(model_path))
teacher.load_state_dict(model_dict)
teacher.to(device)
teacher.eval()
teacher.half()

# %%
# plt.imshow(arr[..., ::-1])

# %%
# testtensor = arr.half().to(device)
# out1, out2 = teacher(testtensor[..., ::2, ::2])

# %%
import colorsys

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    j = 0
    step = 360.0 / num
    while j < num:
        h = i
        s = 90 #+ random.random() * 10
        l = 50 #+ random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
        j += 1
    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors, hls_colors

colors = ncolors(19)[0]
colors = np.array(np.stack(colors) / 256)
colors[18] = np.array([0, 0, 0])

# plt.imshow(np.matmul(out1[0, 1::2, ...].permute(1, 2, 0).squeeze().cpu().detach().numpy(), colors))
# plt.imshow(np.matmul(out2[0, ...].permute(1, 2, 0).squeeze().cpu().detach().numpy(), colors))

# %%
import torch.nn.functional as F

n = 10
resizer = Resize((720, 1280))

for k in range(n):

    dataset = _dataset(df[k * int(len(df) / n) : (k+1) * (int(len(df) / n))])
    inference_loader = DataLoader(dataset, 4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    kptlist = list()
    afflist = list()
    namelist = list()

    for i, batch in tqdm.tqdm(enumerate(inference_loader), total=len(inference_loader)):
        kpt, aff = teacher(torch.tensor(batch[0]).squeeze().to(device).half()[..., ::2, ::2])
        kpt = F.interpolate(kpt, (36, 64))
        aff = F.interpolate(aff, (36, 64))
        kptlist.extend(kpt.detach().cpu().numpy())
        afflist.extend(aff.detach().cpu().numpy())
        namelist.extend(batch[1])
        
                
    df_ = df[k * int(len(df) / n) : (k+1) * (int(len(df) / n))]
    df_['kpt'] = kptlist
    df_['aff'] = afflist
    
    pk.dump(df_[['kpt', 'aff', 'pic']], open('annotate/openpose/data/full_%d.pk' % k, 'wb'))

# %%



