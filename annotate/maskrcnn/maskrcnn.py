# %%
# lookto = '/home/lscsc/caizhijie/0420-wamera-benchmark/annotate_29/3c.pk'
lookto = '/home/lscsc/caizhijie/0710-falldewideo/123c.pk'

import pickle as pk

df = pk.load(open(lookto, 'rb'))

# %%
# df

# # %%
# df.iloc[0]['pic'].replace('sensing', 'lscsc').replace('0702-falldewideo', '0420-wamera-benchmark')

# # %%
# import cv2

# arr = cv2.imread(df.iloc[0]['pic'].replace('sensing', 'lscsc').replace('0702-falldewideo', '0420-wamera-benchmark'))

# # %%
# import matplotlib.pyplot as plt

# plt.imshow(arr[..., ::-1])

# %%
# %%
# import some common detectron2 utilities

import torch

import os

import sys

sys.path.insert(0, '/home/lscsc/caizhijie/0710-falldewideo/external')

import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library

yamlfile = '/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/maskrcnn/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml'
# yamlfile = 'configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
# yamlfile = '/home/lscsc/caizhijie/0420-wamera-benchmark/annotate/maskrcnn/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'

cfg.merge_from_file(yamlfile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yamlfile[11:])
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yamlfile[70:])
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yamlfile)

# %%
from torch.utils.data import Dataset, DataLoader

import cv2
import torch
import tqdm
import numpy as np

class _dataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __getitem__(self, index):
        return self.df.iloc[index]['pic'].replace('sensing', 'lscsc').replace('0702-falldewideo', '0420-wamera-benchmark').replace('data_unzip', 'data')
    
    def __len__(self):
        return len(self.df)
    
def collate_fn(batch):
    imgs = [cv2.imread(_) for _ in batch]
    paths = [_ for _ in batch]
    return np.stack(imgs), paths


class packPredictor(DefaultPredictor):
    def __call__(self, pack):
        # 'NHWC'
        assert len(pack.shape) == 4, 'A default predictor is qualified'
        with torch.no_grad():
            packimage = list()
            packheight = list()
            packwidth = list()
            
            for i in range(pack.shape[0]):
                thisimage = pack[i, ...]
                if self.input_format == 'RGB':
                    thisimage = thisimage[..., ::-1]
                    
                # print(thisimage.shape)
                height, width = thisimage.shape[:2]
                image = self.aug.get_transform(thisimage).apply_image(thisimage)
                image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
                
                packimage.append(image)
                packheight.append(height)
                packwidth.append(width)
            
            inputs = [{'image': packimage[_], 'height': packheight[_], 'width': packwidth[_]} for _ in range(len(packimage))]
            predictions = self.model(inputs)
            return predictions

# %%
n = 1

for k in range(n):

    dataset = _dataset(df[k * int(len(df) / n) : (k+1) * (int(len(df) / n))])
    inference_loader = DataLoader(dataset, 4, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    ppredictor = packPredictor(cfg)

    outputlist = list()

    for i, batch in tqdm.tqdm(enumerate(inference_loader), total=len(inference_loader)):
        output = ppredictor(batch[0][:, ::5, ::5, :])
        
        dump = [_['instances'].to('cpu').get_fields() for _ in output]

        for _ in dump:
            try:
                outputlist.append(_['pred_masks'][torch.where(_['pred_classes'] == 0)[0]][0, ::4, ::4].cpu().detach().numpy())
            except IndexError:
                outputlist.append((torch.zeros((54, 96)) == 1).cpu().detach().numpy())
    
    df_ = df[k * int(len(df) / n) : (k+1) * (int(len(df) / n))]
    df_['maskrcnn'] = outputlist
    pk.dump(df_[['maskrcnn', 'pic']], open('/home/lscsc/caizhijie/0710-falldewideo/annotate/maskrcnn/data/3c_%d.pk' % k, 'wb'))

# %%



