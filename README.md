# falldewideo

0. Preprocess the raw dataset using scripts in annotate (raw dataset is not provided)
1. Train a model by simply ```python train.py```
2. Evaluate your models using ```evaluate.ipynb```

### Notes

0. Raw data is not provided, however we provide rendered data packs, with 
    i.      raw CSI readings, 
    ii.     OpenPose outputs of the original pictures filmed by cameras.
    iii.    MaskRCNN outputs of the original pictures filmed by cameras.
    They will be made accessible ASAP.

1. All of the data are packed into ```pandas``` DataFrames. For i, with columns ['csi0', 'csi1', 'csi2', 'pic'], corresponding to CSI readings from three receivers and the original directories of the raw picture. For ii, with columns ['jhm', 'paf', 'pic'], corresponding to OpenPose output of JHM, PAF and the original directories of the raw picture. For iii, with columns ['mask', 'pic'], corresponding to the mask corresponds to the human objects on the picture and the original directories of the raw picture. The whole dataset is sliced into numerous parts to prevent huge memory consumption. When in use, download the data packs to any directory and guide the script to where you put the packs by passing arguments.

### Dataset is available in:

0. BaiduYun:   https://pan.baidu.com/s/1zTAsQUIbvNPvESXZ7Hu22w?pwd=osrf 

1. Mega:       https://mega.nz/folder/l6cgwbgQ#dFSRER1vkXobu-uaXOam5g (partially, due to size limit. Working on solutions)
