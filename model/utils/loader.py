import numpy as np
import torch
import nibabel as nib
import imutils
import operator
import os
from torch.utils.data import Dataset, DataLoader, Subset

class img_dataset(Dataset):
    # Begin the initialization of the datasets. Creates dataset iterativey for each subject and
    # concatenates them together for both training and testing datasets (implements img_dataset class).
    def __init__(self, name, view, size: int = 192, highres=False, horizontal_flip: bool = False, 
                 vertical_flip: bool = False, rotation_angle: int = None):
        self.root_dir = './Data/'
        self.highres = highres
        self.srctrgt = ['source', 'target']
        self.name = name
        self.view = view
        self.horizontal = horizontal_flip
        self.vertical = vertical_flip
        self.angle = rotation_angle
        self.size = size if not highres else size * 2

    def __len__(self):
        if self.highres:
            path = self.root_dir+'high_res'+'/'+self.name
        else:
            path = self.root_dir+'source'+'/'+self.name
        raw = nib.load(path).get_fdata()

        if self.view == 'L':
            return raw.shape[0]   
        elif self.view == 'A':
            return raw.shape[1]
        else:
            return raw.shape[2]
    
    def rotation(self, x, alpha):
        y = x.astype(np.uint8)
        y_rot = imutils.rotate(y, angle = alpha)
        return y_rot.astype(np.float64)
    
    def resizing(self, img, n):
        target = (n, n)
        if (img.shape > np.array(target)).any():
            target_shape2 = np.min([target, img.shape],axis=0)
            start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
            end = tuple(map(operator.add, start, target_shape2))
            slices = tuple(map(slice, start, end))
            img = img[tuple(slices)]
        offset = tuple(map(lambda a, da: a//2-da//2, target, img.shape))
        slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
        result = np.zeros(target)
        result[tuple(slices)] = img
        return result

    def __getitem__(self, idx):

        if self.highres:
            raw = nib.load(self.root_dir+'high_res/'+self.name).get_fdata()
            if self.view == 'L':
                n_img = self.resizing(raw[idx,:,:], self.size)    
            elif self.view == 'A':
                n_img = self.resizing(raw[:,idx,:], self.size)
            else:
                n_img = self.resizing(raw[:,:,idx], self.size)
        
            if self.horizontal == True:
                n_img = np.flip(n_img,axis=0)

            if self.vertical == True:
                n_img = np.flip(n_img, axis=1)

            if self.angle is not None:
                n_img = self.rotation(n_img, self.angle)

            n_img = np.expand_dims(n_img,axis=-1)
            img_torch = torch.from_numpy(n_img.copy()).type(torch.float)

            dataset = img_torch
        else:
            dataset = dict()
            for class_name in self.srctrgt:
                raw = nib.load(self.root_dir+class_name+'/'+self.name).get_fdata()
                if self.view == 'L':
                    n_img = self.resizing(raw[idx,:,:], self.size)    
                elif self.view == 'A':
                    n_img = self.resizing(raw[:,idx,:], self.size)
                else:
                    n_img = self.resizing(raw[:,:,idx], self.size)
            
                if self.horizontal == True:
                    n_img = np.flip(n_img,axis=0)

                if self.vertical == True:
                    n_img = np.flip(n_img, axis=1)

                if self.angle is not None:
                    n_img = self.rotation(n_img, self.angle)

                n_img = np.expand_dims(n_img,axis=-1)
                img_torch = torch.from_numpy(n_img.copy()).type(torch.float)

                dataset[class_name] = img_torch

        return dataset

def data_augmentation(base_set, path, view, h, ids):
    transformations = {1: (True, None),
                       2: (False, -10), 3: (True, -10),
                       4: (False, -5), 5: (True, -5),
                       6: (False, 5), 7: (True, 5),
                       8: (False, 10), 9: (True, 10)}
    
    for x, specs in transformations.items():
        aug = img_dataset(path, view, size = h, horizontal_flip = specs[0], rotation_angle = specs[1])
        aug = Subset(aug,ids)
        base_set = torch.utils.data.ConcatDataset([base_set, aug])
    return base_set

def loader(source_path, view, batch_size, h):
    low_res = os.listdir(source_path+'source')
    l_n = int(len(low_res)*.8)
    high_res = os.listdir(source_path+'high_res')[:-50]
    h_n = int(len(high_res)*.8)

    tr_lowres_set = img_dataset(low_res[0], view, size = h)
    tr_highres_set = img_dataset(high_res[0], view, size = h, highres = True)

    ts_lowres_set = img_dataset(low_res[l_n], view, size = h)
    ts_highres_set = img_dataset(high_res[h_n], view, size = h, highres = True)

    for idx,image in enumerate(low_res):
        if idx != 0 and idx<l_n:
            low_set = img_dataset(image, view, size = h)
            tr_lowres_set = torch.utils.data.ConcatDataset([tr_lowres_set, low_set])  
        elif idx != 0:
            low_set = img_dataset(image, view, size = h)
            ts_lowres_set = torch.utils.data.ConcatDataset([ts_lowres_set, low_set]) 

    for idx,image in enumerate(high_res):
        if idx != 0 and idx<h_n:
            high_set = img_dataset(image, view, size = h, highres = True)
            tr_highres_set = torch.utils.data.ConcatDataset([tr_highres_set, high_set])
        elif idx != 0:
            high_set = img_dataset(image, view, size = h, highres = True)
            ts_highres_set = torch.utils.data.ConcatDataset([ts_highres_set, high_set])

# Dataloaders generated from datasets 
    tr_lowres_final = DataLoader(tr_lowres_set, shuffle=True, batch_size=batch_size,num_workers=12)
    tr_highres_final = DataLoader(tr_highres_set, shuffle=True, batch_size=batch_size,num_workers=12)
    ts_lowres_final = DataLoader(ts_lowres_set, shuffle=True, batch_size=batch_size,num_workers=12)
    ts_highres_final = DataLoader(ts_highres_set, shuffle=True, batch_size=batch_size,num_workers=12)
    return tr_lowres_final, tr_highres_final, ts_lowres_final, ts_highres_final