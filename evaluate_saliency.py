import os
import timeit
import cv2
import numpy as np
from skimage.color import gray2rgb, rgb2lab
from skimage.io import imread, imsave
from skimage.transform import rescale
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =========================
# Metric functions
# =========================
def auc_judd(saliency, fixation):
    S = saliency.flatten()
    F = fixation.flatten().astype(bool)
    
    Sth = S[F]  # values at fixation locations
    Nfix = len(Sth)
    Npixels = len(S)
    
    all_thresh = np.sort(Sth)[::-1]  # descending
    tp = [0]
    fp = [0]
    
    for thresh in all_thresh:
        tp.append(np.sum(Sth >= thresh)/Nfix)
        fp.append(np.sum(S >= thresh) - np.sum(Sth >= thresh))  # non-fixation above thresh
    fp = np.array(fp)/(Npixels - Nfix)
    
    tp.append(1)
    fp = np.append(fp,1)
    
    return np.trapz(tp, fp)


def auc_borji(saliency, fixation, Nsplits=100):
    S = saliency.flatten()
    F = fixation.flatten().astype(bool)
    
    Sth = S[F]
    Nfix = len(Sth)
    Npixels = len(S)
    
    aucs = []
    for _ in range(Nsplits):
        rand_inds = np.random.choice(Npixels, Nfix, replace=False)
        curfix = S[rand_inds]
        
        all_thresh = np.linspace(0, max(np.max(Sth), np.max(curfix)), 100)
        tp = [0]
        fp = [0]
        
        for t in all_thresh:
            tp.append(np.sum(Sth >= t)/Nfix)
            fp.append(np.sum(curfix >= t)/Nfix)
        
        tp.append(1)
        fp.append(1)
        
        aucs.append(np.trapz(tp, fp))
    return np.mean(aucs)


def nss(saliency, fixation):
    S = saliency.astype(np.float32)
    S = (S - S.mean()) / (S.std() + 1e-8)
    return S[fixation.astype(bool)].mean()


def kl_div(saliency, fixation):
    S = saliency.astype(np.float32)
    F = fixation.astype(np.float32)
    
    S = S / (S.sum() + 1e-8)
    F = F / (F.sum() + 1e-8)
    
    return np.sum(F * np.log((F + 1e-8)/(S + 1e-8)))

def cc(saliency, fixation):
    S = saliency.astype(np.float32)
    F = fixation.astype(np.float32)
    
    S = (S - S.mean()) / (S.std() + 1e-8)
    F = (F - F.mean()) / (F.std() + 1e-8)
    
    return np.sum(S*F) / (np.sqrt(np.sum(S*S)*np.sum(F*F)) + 1e-8)


def sim(saliency, fixation):
    S = saliency.astype(np.float32)
    F = fixation.astype(np.float32)
    
    S = S / (S.sum() + 1e-8)
    F = F / (F.sum() + 1e-8)
    
    return np.sum(np.minimum(S, F))
