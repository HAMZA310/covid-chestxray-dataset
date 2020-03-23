from PIL import Image
from os.path import join
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pickle
import skimage
import glob
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import skimage.transform
import warnings
import tarfile

default_pathologies = [  'Atelectasis',
                 'Consolidation',
                 'Infiltration',
                 'Pneumothorax',
                 'Edema',
                 'Emphysema',
                 'Fibrosis',
                 'Effusion',
                 'Pneumonia',
                 'Pleural_Thickening',
                 'Cardiomegaly',
                 'Nodule',
                 'Mass',
                 'Hernia',
                 'Lung Lesion',
                 'Fracture',
                 'Lung Opacity',
                 'Enlarged Cardiomediastinum'
                ]

thispath = os.path.dirname(os.path.realpath(__file__))

def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

def relabel_dataset(pathologies, dataset):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:,pathology_idx])
        else:
            print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T
    
    dataset.labels = new_labels
    dataset.pathologies = pathologies

class Dataset():
    def __init__(self):
        pass
    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies,counts))
    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")
        
    
class COVID19_Dataset(Dataset):
    """
    COVID-19 image data collection
    https://github.com/ieee8023/covid-chestxray-dataset
    """
    
    def __init__(self, 
                 imgpath= "/covid-chestxray-dataset/images",
                 csvpath="/covid-chestxray-dataset/metadata.csv", 
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True):

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        
        # defined here to make the code easier to read
        pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
        
        self.pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
        self.pathologies = sorted(self.pathologies)

        mapping = dict()
        mapping["Pneumonia"] = pneumonias
        mapping["Viral Pneumonia"] = ["COVID-19", "SARS", "MERS"]
        mapping["Bacterial Pneumonia"] = ["Streptococcus"]
        
        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the frontal views.
        #idx_pa = self.csv["view"].isin(["PA", "AP", "AP Supine"])
        idx_pa = self.csv["view"].isin(["PA"])
        self.csv = self.csv[idx_pa]
        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["finding"].str.contains(syn)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
            
        return {"PA":img, "lab":self.labels[idx], "idx":idx}
    
    

    
