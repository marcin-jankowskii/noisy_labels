import torch
import yaml
import glob
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize

#path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
path_to_config = '/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml'


class ProcessData:
    def __init__(self, config_path=path_to_config, mode = 'full',annotator = 1):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
            self.mode = mode
            self.annotator = annotator

    def process_dataset(self, dataset_name):
        dataset_path = self.config['dataset_path']
        dataset_path = dataset_path + dataset_name
        print(dataset_path)

        if self.annotator == 1:
            name = '/GT1_'
        elif self.annotator == 2:
            name = '/GT2_'
        if self.mode == 'full':
            segment = 'full/'
        elif self.mode == 'head':
            segment = 'head/'
        elif self.mode == 'tail':
            segment = 'tail/'

        gt_path = dataset_path + name + segment
        images = sorted(glob.glob(f"{dataset_path}/images/*"))
        masks = sorted(glob.glob(f"{gt_path}*.png"))

        X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
        y = np.zeros((len(masks), self.config['image_height'], self.config['image_width'], 1), dtype=np.float32)

        for n, (img, mimg) in enumerate(zip(images, masks)):
            # Load images
            img = cv2.imread(img)
            x_img = img.astype(np.float32)
            x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
            # Load masks
            mask = cv2.imread(mimg)
            mask = mask.astype(np.float32)
            mask = resize(mask, (self.config['image_height'], self.config['image_width'], 1), mode='constant', preserve_range=True)
            # Save images
            X[n] = x_img / 255.0
            y[n] = mask / 255.0

        return X, y


class BatchMaker:
    def __init__(self, config_path=path_to_config, batch_size=6, mode = 'all',segment = 'full' ,annotator = 1):
        
    
        self.process_data = ProcessData(config_path=config_path,mode = segment,annotator = annotator)
        self.batch_size = batch_size
        if mode == 'all':
            x_train, y_train = self.process_data.process_dataset('/train')
            x_val, y_val = self.process_data.process_dataset('/test_small')
            x_test, y_test = self.process_data.process_dataset('/test')
            self.train_loader = self.create_loader(x_train, y_train, shuffle=True)
            self.val_loader = self.create_loader(x_val, y_val, shuffle=True)
            self.test_loader = self.create_loader(x_test, y_test, shuffle=False)
        elif mode == 'train':
            x_train, y_train = self.process_data.process_dataset('/train')
            x_val, y_val = self.process_data.process_dataset('/test_small')
            self.train_loader = self.create_loader(x_train, y_train, shuffle=True)
            self.val_loader = self.create_loader(x_val, y_val, shuffle=True)
        elif mode == 'test':
            x_test, y_test = self.process_data.process_dataset('/test')
            self.test_loader = self.create_loader(x_test, y_test, shuffle=False)
        

    def create_loader(self, x, y, shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)