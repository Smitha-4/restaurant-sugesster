import io
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torch

class text2ImageDataset(Dataset):
    def __init__(self, datasetfile, transform =None, split=0):
        self.datasetfile = datasetfile
        self.transform =transform
        self.dataset =None
        self.dataset_keys = None
        self.split = 'train' if split == 0 else 'valid' if split == 1 else 'test'
        self.h5py2int = lambda x:int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetfile,'r')
        self.dataset_keys = [str(k) for k in f[self.split].keys()]
        length = len(f[self.split])
        f.close()
        return length
    
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetfile, mode ='r')
        example_name =self.dataset_keys[index]
        example = self.dataset[self.split][example_name]
        correct_image =bytes(np.array(example['img']))
        correct_embed = np.array(example['embeddings'], dtype=float)
        wrong_image =bytes(np.array(self.find_wrong_image(example['class'])))
        inter_embed = np.array(self.find_inter_embed())
        correct_image =Image.open(io.BytesIO(correct_image)).resize((64,64))
        wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64,64))
        correct_image = self.validate_image(correct_image)
        wrong_image = self.validate_image(wrong_image)

        try:
            txt = np.array(example['txt']).astype(str)
        except:
            txt = np.array([example['text'][()].decode('utf-8', errors='replcae')])
            txt = np.char.replace(txt, '', ' ').astype(str)
        sample = {
            'correct_images': torch.FloatTensor(correct_image),
            'correct_embed':torch.FloatTensor(correct_embed),
            'wrong_images': torch.FloatTensor(wrong_image),
            'inter_embed': torch.FloatTensor(inter_embed),
            'txt':str(txt)
        }
        sample['correct_image'] = sample['correct_image'].sub_(127.5).div_(127.5)
        sample['wrong-images'] = sample['wrong-images'].sub_(127.5).div_(127.5)

        return sample
    
    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_inter_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.split][example_name]
        return example['embeddings']

    def validate_image(self, img):
        img = np.array(img, dtype=float)
        if len(img.shape) < 3:
            rgb = np.empty((64, 64, 3), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb

        return img.transpose(2, 0, 1)
        