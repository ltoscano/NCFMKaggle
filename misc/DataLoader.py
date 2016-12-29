import json 
import numpy as np 
import os 
from scipy.misc import imread 

class DataLoader():

    def __init__(self, batch_size, data_path, info_path):
        self.batch_size = batch_size
        with open(info_path, 'r') as f:
            self.im_info = json.load(f)

        self.num_imgs = self.im_info['total_images']
        self.data_path = data_path
        self.im2class = self.im_info['im_class_map']
        self.img_names = self.im2class.keys()

        self.perm = np.random.permutation(self.num_imgs)
        self.iter = 0

    def next_batch(self):
        imgs = np.zeros([self.batch_size, 224, 224, 3])
        labels = np.zeros([self.batch_size])
        for i in range(self.batch_size):
            img_base = self.img_names[self.perm[self.iter]]
            img_path = os.path.join(self.data_path, img_base)
            imgs[i, :, :, :] = imread(img_path)
            labels[i] = self.im2class[img_base]

            self.iter = self.iter + 1
            if self.iter == self.num_imgs:
                self.iter = 0

        return imgs, labels

if __name__ == '__main__':

    loader = DataLoader(16, 'train_proc/', 'fish.json')
    imgs, labels = loader.next_batch()

