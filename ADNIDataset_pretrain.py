import csv
import numpy as np
import torch
import math
import time
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import os
import torch.nn.functional as F
from PIL import Image
import imageio
import torchvision.datasets as datasets
from torchvision import transforms
from skimage.transform import resize
import nibabel as nib
import random
labels_name = ['CN','MCI','AD']
class ADNIDataset(Dataset):
    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        imgs = torch.tensor(imgs)
        p = 4
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *1)
        imgs: (N, 1, H, W)
        """
        p = 4
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p,1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
       # imgs = imgs.reshape(shape=(x.shape[0], h * p, h * p))
        return imgs

    def get_top(self,x):
        k=10000
        values,indices = x.topk(k,dim=0,largest=True,sorted=True)
        print('values = ',values)
        print('indices = ',indices)
        result = torch.zeros_like(x)
        result.scatter_(0,indices,values)
        return result

    def __init__(self, root,label_tsv,transform,sample_hard=False,sample_hard_level=0,augmentation=False):
        self.root = root
        self.basis = '*'
        self.augmentation = augmentation
        self.sample_hard = sample_hard
        self.sample_hard_level = sample_hard_level
        self.transform = transform
        self.softmax = torch.nn.Softmax(dim=1)
        name = []
        basiss = []
        basis = []
        labels = {}
        mul_labels = {}
        date  = []
        for dirpath, dirnames, filenames in os.walk(root):

            for filename in filenames:
                if filename.endswith(".png"):
                    name.append(os.path.join(dirpath, filename))
        f = open(label_tsv,'r')
        rdr = csv.reader(f)
        self.name = name
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        path = os.path.join(self.root,self.name[index])
        import skimage.io as io
        img = io.imread(path)
        #print('img = ',img)
        #data = resize(img,(256,256))
        data = img
        #print('data = ',data)
        #print('data change = ',Image.fromarray(data))
        data = (1.0 * data / np.max(data) * 255.0).astype(np.uint8)
        #print('data transform = ',self.transform(Image.fromarray(data)))
        imgs = []
        imgs.append(self.transform(Image.fromarray(data)))

        img = torch.cat(imgs, 0)

        target = [1]
        weight = torch.ones_like(img)
        #if False:
        #    pixel_array = np.zeros((img[0].shape[0], img[0].shape[1]), dtype=np.uint8)
        #    for j in range(img.shape[0]):
        #        pixel_array = (img[j].cpu().clone().detach().numpy() / np.max(img.cpu().clone().detach().numpy()) * 255.0).astype(np.uint8)
        #        pil_image = Image.fromarray(pixel_array)
        #        index = random.randint(1, 500000)
        #        pil_image.save(os.path.join('./output/figures_adni/','{}.png'.format(index)))

        return (img,torch.tensor(target).repeat(img.shape[0]),weight)

def make_dataset(data_dir,label_csv,transform,sample_hard,sample_hard_level, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = ADNIDataset(data_dir,label_csv,transform,sample_hard,sample_hard_level)
    return dataset
