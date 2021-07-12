import os
import glob
import h5py
import numpy as np

from torch.utils.data import Dataset

def load_point_cloud(partition, dataset_dir):
    all_data = []
    all_label = []
    
    for h5_name in glob.glob(os.path.join(dataset_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        split = h5_name[-4]
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32') # (N, 2048, 3)
        label = f['label'][:].astype('int64') # 0-39
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

class PointDataset(Dataset):
    def __init__(self, dataset_dir, num_points=2048, num_classes=40, partition='test'):
        self.dataset_dir = dataset_dir
        self.partition = partition
        self.data, self.label = load_point_cloud(self.partition, self.dataset_dir)
        self.num_points = num_points
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]
