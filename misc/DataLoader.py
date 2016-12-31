import json 
import numpy as np 
import os 
from scipy.misc import imread 

class DataLoader():

    def __init__(self, batch_size, data_path, info_path, val_split, test=False):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test = test 

        self.data_path = data_path
        if self.test:
            self.img_names = os.listdir(data_path)
            self.total_imgs = len(self.img_names)
        else:
            with open(info_path, 'r') as f:
                self.im_info = json.load(f)
            self.im2class = self.im_info['im_class_map']
            self.img_names = self.im2class.keys()
            self.total_imgs = self.im_info['total_images']

        self.split_size = [0, int(self.total_imgs*val_split)]
        self.split_size[0] = self.total_imgs - self.split_size[1]
        self.train_size = self.split_size[0]
        self.val_size = self.split_size[1]

        self.perm = np.random.permutation(self.total_imgs)
        self.perm_split = [self.perm[:self.split_size[0]], self.perm[self.split_size[0]:]]

        train_classes = np.zeros(8)
        for i in range(self.split_size[0]):
            train_classes[self.im2class[self.img_names[self.perm_split[0][i]]]] += 1
        print(train_classes)
        print(train_classes / self.train_size)

        val_classes = np.zeros(8)
        for i in range(self.split_size[1]):
            val_classes[self.im2class[self.img_names[self.perm_split[1][i]]]] += 1
        print(val_classes)
        print(val_classes / self.val_size)

        self.iter = [0, 0]

    def next_batch(self, split=0):
        imgs = np.zeros([self.batch_size, 224, 224, 3])
        labels = np.zeros([self.batch_size])
        wrap = False
        batch_names = []
        for i in range(self.batch_size):
            ix = self.iter[split]
            img_base = self.img_names[self.perm_split[split][ix]]
            batch_names.append(img_base)
            img_path = os.path.join(self.data_path, img_base)
            imgs[i, :, :, :] = imread(img_path)

            if not self.test:
                labels[i] = self.im2class[img_base]

            self.iter[split] = self.iter[split] + 1
            if self.iter[split] == self.split_size[split]:
                wrap = True
                self.iter[split] = 0

        if self.test:
            return imgs, None, wrap, batch_names
        else:
            return imgs, labels, wrap

if __name__ == '__main__':

    loader = DataLoader(16, '/scratch/cluster/vsub/ssayed/NCFMKaggle/test_proc/', info_path=None, val_split=0, test=True)
    imgs, _, wrap, names = loader.next_batch()
    print(names)

