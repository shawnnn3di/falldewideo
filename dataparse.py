import numpy as np
import pandas as pd
import pickle as pk
import torch

from torch.utils.data import Dataset, DataLoader

class framewise(Dataset):
    
    def __init__(self, dfcsi, dfmask, dfpose):
        self.dfcsi = dfcsi
        self.dfmask = dfmask
        self.dfpose = dfpose
        self.k = self.scidx(40, 4)
        
    def scidx(self, bw, ng, standard='n'):
        """subcarriers index

        Args:
            bw: bandwitdh(20, 40, 80)
            ng: grouping(1, 2, 4)
            standard: 'n' - 802.11n， 'ac' - 802.11ac.
        Ref:
            1. 802.11n-2016: IEEE Standard for Information technology—Telecommunications
            and information exchange between systems Local and metropolitan area
            networks—Specific requirements - Part 11: Wireless LAN Medium Access
            Control (MAC) and Physical Layer (PHY) Specifications, in
            IEEE Std 802.11-2016 (Revision of IEEE Std 802.11-2012), vol., no.,
            pp.1-3534, 14 Dec. 2016, doi: 10.1109/IEEESTD.2016.7786995.
            2. 802.11ac-2013 Part 11: ["IEEE Standard for Information technology--
            Telecommunications and information exchange between systemsLocal and
            metropolitan area networks-- Specific requirements--Part 11: Wireless
            LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications
            --Amendment 4: Enhancements for Very High Throughput for Operation in 
            Bands below 6 GHz.," in IEEE Std 802.11ac-2013 (Amendment to IEEE Std
            802.11-2012, as amended by IEEE Std 802.11ae-2012, IEEE Std 802.11aa-2012,
            and IEEE Std 802.11ad-2012) , vol., no., pp.1-425, 18 Dec. 2013,
            doi: 10.1109/IEEESTD.2013.6687187.](https://www.academia.edu/19690308/802_11ac_2013)
        """

        PILOT_AC = {
            20: [-21, -7, 7, 21],
            40: [-53, -25, -11, 11, 25, 53],
            80: [-103, -75, -39, -11, 11, 39, 75, 103],
            160: [-231, -203, -167, -139, -117, -89, -53, -25, 25, 53, 89, 117, 139, 167, 203, 231]
        }
        SKIP_AC_160 = {1: [-129, -128, -127, 127, 128, 129], 2: [-128, 128], 4: []}
        AB = {20: [28, 1], 40: [58, 2], 80: [122, 2], 160: [250, 6]}
        a, b = AB[bw]

        if standard == 'n':
            if bw not in [20, 40] or ng not in [1, 2, 4]:
                raise ValueError("bw should be [20, 40] and ng should be [1, 2, 4]")
            k = np.r_[-a:-b:ng, -b, b:a:ng, a]
        if standard == 'ac':
            if bw not in [20, 40, 80] or ng not in [1, 2, 4]:
                raise ValueError("bw should be [20, 40, 80] and ng should be [1, 2, 4]")

            g = np.r_[-a:-b:ng, -b]
            k = np.r_[g, -g[::-1]]

            if ng == 1:
                index = np.searchsorted(k, PILOT_AC[bw])
                k = np.delete(k, index)
            if bw == 160:
                index = np.searchsorted(k, SKIP_AC_160[ng])
                k = np.delete(k, index)
        return k
        
    def calib(self, phase, k, axis=1):
        """Phase calibration

        Args:
            phase (ndarray): Unwrapped phase of CSI.
            k (ndarray): Subcarriers index
            axis (int): Axis along which is subcarrier. Default: 1

        Returns:
            ndarray: Phase calibrated

        ref:
            [Enabling Contactless Detection of Moving Humans with Dynamic Speeds Using CSI]
            (http://tns.thss.tsinghua.edu.cn/wifiradar/papers/QianKun-TECS2017.pdf)
        """
        p = np.asarray(phase)
        k = np.asarray(k)

        slice1 = [slice(None, None)] * p.ndim
        slice1[axis] = slice(-1, None)
        slice1 = tuple(slice1)
        slice2 = [slice(None, None)] * p.ndim
        slice2[axis] = slice(None, 1)
        slice2 = tuple(slice2)
        shape1 = [1] * p.ndim
        shape1[axis] = k.shape[0]
        shape1 = tuple(shape1)

        k_n, k_1 = k[-1], k[1]
        a = (p[slice1] - p[slice2]) / (k_n - k_1)
        b = p.mean(axis=axis, keepdims=True)
        k = k.reshape(shape1)

        phase_calib = p - a * k - b
        return phase_calib
        
    def __getitem__(self, idx):
        mask = self.dfmask.iloc[idx]['maskrcnn']
        aff = self.dfpose.iloc[idx]['aff']
        kpt = self.dfpose.iloc[idx]['kpt']
        name = self.dfcsi.iloc[idx]['pic']
        
        csiamp0 = np.abs(self.dfcsi.iloc[idx]['csi0']).reshape(50, 30, 9).transpose((2, 1, 0))
        csipha0 = self.calib(np.unwrap(np.angle(self.dfcsi.iloc[idx]['csi0'])), self.k).reshape(50, 30, 9).transpose((2, 1, 0))
        csiamp1 = np.abs(self.dfcsi.iloc[idx]['csi1']).reshape(50, 30, 9).transpose((2, 1, 0))
        csipha1 = self.calib(np.unwrap(np.angle(self.dfcsi.iloc[idx]['csi1'])), self.k).reshape(50, 30, 9).transpose((2, 1, 0))
        csiamp2 = np.abs(self.dfcsi.iloc[idx]['csi2']).reshape(50, 30, 9).transpose((2, 1, 0))
        csipha2 = self.calib(np.unwrap(np.angle(self.dfcsi.iloc[idx]['csi2'])), self.k).reshape(50, 30, 9).transpose((2, 1, 0))
    
        csi = np.vstack([csiamp0, csipha0, csiamp1, csipha1, csiamp2, csipha2])
        
        return mask, aff, kpt, csi, name
    
    def __len__(self):
        return len(self.dfcsi)


class eventwise(framewise):
    
    def __init__(self, dfcsi, dfmask, dfpose, samplelength=10, eventlength=10):
        
        self.k = self.scidx(40, 4)
        
        self.dfcsi = dfcsi
        self.dfmask = dfmask
        self.dfpose = dfpose
        
        self.samplelength = samplelength
        
        idxmaplist = list(range(len(dfcsi)))
        idxmap = pd.DataFrame.from_dict({'samplable_idx': idxmaplist})
        idxmap = idxmap[idxmap['samplable_idx'].apply(lambda x: x % eventlength > (samplelength + 0))]
        self.idxmap = idxmap.sample(frac=0.25)
        
        print('processing...')
        self.dfcsi.loc[:, 'csiamp0'] = self.dfcsi.loc[:, 'csi0'].apply(lambda x: np.abs(x))
        self.dfcsi.loc[:, 'csiamp1'] = self.dfcsi.loc[:, 'csi1'].apply(lambda x: np.abs(x))
        self.dfcsi.loc[:, 'csiamp2'] = self.dfcsi.loc[:, 'csi2'].apply(lambda x: np.abs(x))
        
        self.dfcsi.loc[:, 'csipha0'] = self.dfcsi.loc[:, 'csi0'].apply(lambda x: self.calib(np.unwrap(np.angle(x)), self.k).reshape(50, 30, 9).transpose((2, 1, 0)))
        self.dfcsi.loc[:, 'csipha1'] = self.dfcsi.loc[:, 'csi1'].apply(lambda x: self.calib(np.unwrap(np.angle(x)), self.k).reshape(50, 30, 9).transpose((2, 1, 0)))
        self.dfcsi.loc[:, 'csipha2'] = self.dfcsi.loc[:, 'csi2'].apply(lambda x: self.calib(np.unwrap(np.angle(x)), self.k).reshape(50, 30, 9).transpose((2, 1, 0)))
        
    def __getitem__(self, index):
        idx = self.idxmap.iloc[index]['samplable_idx']
        mask = self.dfmask.iloc[idx - self.samplelength: idx]['maskrcnn']
        aff = self.dfpose.iloc[idx - self.samplelength: idx]['aff']
        kpt = self.dfpose.iloc[idx - self.samplelength: idx]['kpt']
        name = self.dfcsi.iloc[idx - self.samplelength: idx]['pic']

        csiamp0 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csiamp0'])).reshape(self.samplelength, 50, 30, 9)
        csiamp1 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csiamp1'])).reshape(self.samplelength, 50, 30, 9)
        csiamp2 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csiamp2'])).reshape(self.samplelength, 50, 30, 9)
        csipha0 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csipha0'])).reshape(self.samplelength, 50, 30, 9)
        csipha1 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csipha1'])).reshape(self.samplelength, 50, 30, 9)
        csipha2 = np.stack(list(self.dfcsi.iloc[idx - self.samplelength : idx]['csipha2'])).reshape(self.samplelength, 50, 30, 9)

        csi = np.concatenate([csiamp0, csipha0, csiamp1, csipha1, csiamp2, csipha2], axis=3).transpose((0, 3, 2, 1))
        
        mask = np.stack(list(dict(mask).values()))
        aff = np.stack(list(dict(aff).values()))
        kpt = np.stack(list(dict(kpt).values()))
        name = list(dict(name).values())
        
        return mask, aff, kpt, csi, name
        
    def __len__(self):
        return len(self.idxmap)


def collate_fn(batch):
    
    mask = list()
    aff = list()
    kpt = list()
    csi = list()
    name = list()
    
    for b in batch:
        m, a, k, c, n = b
        mask.append(m)
        aff.append(a)
        kpt.append(k)
        csi.append(c)
        name.append(n)
        
    mask = torch.tensor(np.stack(mask))
    aff = torch.tensor(np.stack(aff))
    kpt = torch.tensor(np.stack(kpt))
    csi = torch.tensor(np.abs(np.stack(csi)))
    pic = [np.zeros((960, 540, 3), dtype=np.uint8) - 0.5 for _ in range(len(aff))]
    
    return dict(zip(['mask', 'aff', 'kpt', 'csi', 'img', 'name'], [mask, aff, kpt, csi, pic, name]))


def build_loader(pathcsi, pathmask, pathpose, eventlength=1, testfrac=0.05, trainfrac=0.8, bs=32, nw=12, mode='framewise'):
    
    # read the data pack
    dfcsi, dfmask, dfpose = (
        pk.load(open(pathcsi, 'rb')),
        pk.load(open(pathmask, 'rb')),
        pk.load(open(pathpose, 'rb'))
    )
    
    '''
    split train/valid/test dataset, in eventwise or framewise manner. eventlength == 1 is simply
    frame-level separate scheme. In event-level split scheme, dataset is first sliced into samples
    of a fixed length. Randomly select the samples, instead of frames.
    '''
    
    n = len(dfcsi)
    
    rows = pd.DataFrame({'idx': range(n)})
    
    samples = pd.DataFrame({'idx': list(range(int(n / eventlength)))}).sample(frac=1.0)
    ns = len(samples)
    
    splits = [int(testfrac * ns), int((testfrac + trainfrac) * ns)]
    
    testsamples = samples.iloc[:splits[0]]
    trainsamples = samples.iloc[splits[0]: splits[1]]
    validsamples = samples.iloc[splits[1]:]
    
    if eventlength != 1:
        
        testrows = pd.concat([rows.iloc[_ * eventlength : (_ + 1) * eventlength] for _ in testsamples['idx']])
        trainrows = pd.concat([rows.iloc[_ * eventlength : (_ + 1) * eventlength] for _ in trainsamples['idx']])
        validrows = pd.concat([rows.iloc[_ * eventlength : (_ + 1) * eventlength] for _ in validsamples['idx']])
        
    else:
        testrows = rows.iloc[testsamples]
        trainrows = rows.iloc[trainsamples]
        validrows = rows.iloc[validsamples]
        
    dfcsitrain = dfcsi.iloc[trainrows['idx']]
    dfmasktrain = dfmask.iloc[trainrows['idx']]
    dfposetrain = dfpose.iloc[trainrows['idx']]
    
    dfcsivalid = dfcsi.iloc[validrows['idx']]
    dfmaskvalid = dfmask.iloc[validrows['idx']]
    dfposevalid = dfpose.iloc[validrows['idx']]
    
    if mode == 'eventwise':
        trainset = eventwise(dfcsitrain, dfmasktrain, dfposetrain)
        validset = eventwise(dfcsivalid, dfmaskvalid, dfposevalid)
    
    elif mode == 'framewise':
        trainset = framewise(dfcsitrain, dfmasktrain, dfposetrain)
        validset = framewise(dfcsivalid, dfmaskvalid, dfposevalid)
    
    trainloader = DataLoader(
        trainset, 
        batch_size=bs, 
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    validloader = DataLoader(
        validset, 
        batch_size=bs, 
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return trainloader, validloader