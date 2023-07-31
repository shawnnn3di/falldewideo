import sys

openposepath = '/home/sensing/caizhijie/ref-rep/pytorch-openpose'
sys.path.append(openposepath)

from src import util

import tqdm
import cv2
import torch
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
import math

def _preview(oriImg, Mconv7_stage6_L1, Mconv7_stage6_L2):
    oriImg = np.zeros((3, 960, 540))
    scale_search = [.5]
    scale = 1.
    boxsize = 540
    stride = 8
    padValue = 128
    thre1 = 0.1
    thre2 = 0.05
    
    x = 960
    y = 540
    
    Mconv7_stage6_L1 = np.array(Mconv7_stage6_L1, dtype=float)
    Mconv7_stage6_L2 = np.array(Mconv7_stage6_L2, dtype=float)

    oriImg = np.ascontiguousarray(oriImg).transpose(1, 2, 0)

    multiplier = [x * boxsize / oriImg.shape[1] for x in scale_search]
    heatmap_avg = np.zeros((y, x, 19))
    paf_avg = np.zeros((y, x, 38))

    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
    
    heatmap = np.squeeze(Mconv7_stage6_L2).transpose(1, 2, 0)  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    heatmap = cv2.resize(heatmap, (oriImg.shape[0], oriImg.shape[1]), interpolation=cv2.INTER_CUBIC)

    paf = np.squeeze(Mconv7_stage6_L1).transpose(1, 2, 0)  # output 0 is PAFs
    paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[0], oriImg.shape[1]), interpolation=cv2.INTER_CUBIC)

    heatmap_avg += heatmap_avg + heatmap / len(multiplier)
    paf_avg += + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        one_heatmap = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(one_heatmap.shape)
        map_left[1:, :] = one_heatmap[:-1, :]
        map_right = np.zeros(one_heatmap.shape)
        map_right[:-1, :] = one_heatmap[1:, :]
        map_up = np.zeros(one_heatmap.shape)
        map_up[:, 1:] = one_heatmap[:, :-1]
        map_down = np.zeros(one_heatmap.shape)
        map_down[:, :-1] = one_heatmap[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (one_heatmap >= map_left, one_heatmap >= map_right, one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        peak_id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                [55, 56], [37, 38], [45, 46]]

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    norm = max(0.001, norm)
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                        for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                        for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append(
                            [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    return candidate, subset


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


def getnormdis(arr, w=64, h=36, thres=0.4, thres_pd=0.1):

    dislist = list()
    allpdxlist = list()
    allpdylist = list()
    allgtxlist = list()
    allgtylist = list()

    for idx in range(arr[2].shape[0]):
        
        pdxlist = list()
        pdylist = list()
        gtxlist = list()
        gtylist = list()
        
        for kptidx in range(arr[2].shape[1] - 1):
            
            dis = list()
            # print(arr[3][idx, kptidx, ...].max())
            if arr[3][idx, kptidx, ...].max() > thres:
                
                if arr[2][idx, kptidx, ...].max() > thres_pd:
                    pdx = ((arr[2][idx, kptidx, ...] ** 2).sum(0) / (arr[2][idx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                    pdy = ((arr[2][idx, kptidx, ...] ** 2).sum(1) / (arr[2][idx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
                else:
                    pdx = -1
                    pdy = -1

                gtx = ((arr[3][idx, kptidx, ...] ** 2).sum(0) / (arr[3][idx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                gty = ((arr[3][idx, kptidx, ...] ** 2).sum(1) / (arr[3][idx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
                
                pdxlist.append(pdx)
                pdylist.append(pdy)
                gtxlist.append(gtx)
                gtylist.append(gty)
                
            pdxarr = np.array(pdxlist)
            pdyarr = np.array(pdylist)
            gtxarr = np.array(gtxlist)
            gtyarr = np.array(gtylist)
            
            allpdxlist.append(pdxarr)
            allpdylist.append(pdyarr)
            allgtxlist.append(gtxarr)
            allgtylist.append(gtyarr)
            
            
            if len(gtxarr) > 1:
                normdis = ((gtxarr.max() - gtxarr.min()) ** 2 + (gtyarr.max() - gtyarr.min()) ** 2) ** 0.5
                dis = list(((gtxarr - pdxarr) ** 2 + (gtyarr - pdyarr) ** 2) ** 0.5 / normdis)
        
        dislist.extend(dis)
        
    return dislist


def getkpt(arr, idx, timeidx, w, h, long, thres=0.2):

    realkptslist = list()
    fakekptslist = list()

    for kptidx in range(18):

        if long:
            if arr[1][idx, timeidx, kptidx, ...].max() > thres:
                gtx = ((arr[1][idx, timeidx, kptidx, ...] ** 2).sum(0) / (arr[1][idx, timeidx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                gty = ((arr[1][idx, timeidx, kptidx, ...] ** 2).sum(1) / (arr[1][idx, timeidx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
            else:
                gtx = -1
                gty = -1
            if arr[0][idx, timeidx, kptidx, ...].max() > thres:
                pdx = ((arr[0][idx, timeidx, kptidx, ...] ** 2).sum(0) / (arr[0][idx, timeidx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                pdy = ((arr[0][idx, timeidx, kptidx, ...] ** 2).sum(1) / (arr[0][idx, timeidx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
            else:
                pdx = -1
                pdy = -1
        
        else:
            if arr[3][idx, kptidx, ...].max() > thres:
                gtx = ((arr[3][idx, kptidx, ...] ** 2).sum(0) / (arr[3][idx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                gty = ((arr[3][idx, kptidx, ...] ** 2).sum(1) / (arr[3][idx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
            else:
                gtx = -1
                gty = -1
            if arr[2][idx, kptidx, ...].max() > thres:
                pdx = ((arr[2][idx, kptidx, ...] ** 2).sum(0) / (arr[2][idx, kptidx, ...] ** 2).sum(0).sum() * np.array(range(w))).sum() / w
                pdy = ((arr[2][idx, kptidx, ...] ** 2).sum(1) / (arr[2][idx, kptidx, ...] ** 2).sum(1).sum() * np.array(range(h))).sum() / h
            else:
                pdx = -1
                pdy = -1

        realkptslist.append(np.array([gtx, gty]))
        fakekptslist.append(np.array([pdx, pdy]))
        
    realkpts = np.stack(realkptslist)
    fakekpts = np.stack(fakekptslist)
    
    return realkpts, fakekpts


def pose(canvas, kpts, w=960, h=540, stickwidth=4, colors=None):
    kpts[:, 0] = kpts[:, 0] * w
    kpts[:, 1] = kpts[:, 1] * h
    
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    
    for i in range(18):
        x, y = kpts[i][0:2]
        try:
            if (x > 0) and (y > 0):
                cv2.circle(canvas, (int(y), int(x)), 4, colors[i], thickness=4)
        except ValueError:
            pass
    
    cur_canvas = canvas.copy()
    for i in range(len(limbSeq) - 2):
        try:
            Y = kpts[[limbSeq[i][0] - 1, limbSeq[i][1] - 1], 1]
            X = kpts[[limbSeq[i][0] - 1, limbSeq[i][1] - 1], 0]
            if (X.min() > 0) and (Y.min() > 0):
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        except ValueError:
            pass

    
    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas