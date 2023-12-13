import torch
import yaml
import glob
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize
import torch.nn.functional as F



def rgb_to_class_id(mask_rgb, class_colors):
        mask_id = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        for class_id, color in enumerate(class_colors):
            mask_id[(mask_rgb == color).all(axis=2)] = class_id
        return mask_id



path_to_config = '/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
#path_to_config = '/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml'
#path_to_config = '/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml'

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
        elif self.mode == 'mixed':
            segment = 'mixed/'    
        
        gt_path = dataset_path + name + segment
        images = sorted(glob.glob(f"{dataset_path}/images/*"))
        masks = sorted(glob.glob(f"{gt_path}*.png"))
        class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]  # tło, wić, główka

        X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
        y = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],3), dtype=np.float32)
        y_id = np.zeros((len(masks),  self.config['image_height'], self.config['image_width']), dtype=np.float32)

        for n, (img, mimg) in enumerate(zip(images, masks)):
            # Load images
            img = cv2.imread(img)
            x_img = img.astype(np.float32)
            x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
            # Load masks
            mask = cv2.imread(mimg)
            mask = mask.astype(np.float32)
            mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
            mask_id = rgb_to_class_id(mask, class_colors)

            # Convert mask_id to tensor and then to one-hot format
            mask = torch.from_numpy(mask_id)
            mask = F.one_hot(mask, num_classes=len(class_colors))

            # Save images and masks
            X[n] = x_img/255.0 
            y[n] = mask
            y_id[n] = mask_id

        return X, y, y_id


class BatchMaker:
    def __init__(self, config_path=path_to_config, batch_size=6, mode = 'all',segment = 'full' ,annotator = 1):
        
    
        self.process_data = ProcessData(config_path=config_path,mode = segment,annotator = annotator)
        self.batch_size = batch_size
        if mode == 'all':
            x_train, y_train,yid_train = self.process_data.process_dataset('/train')
            x_val, y_val,yid_val = self.process_data.process_dataset('/test_small')
            x_test, y_test,yid_test = self.process_data.process_dataset('/test')
            self.train_loader = self.create_loader(x_train, y_train,yid_train,shuffle=True)
            self.val_loader = self.create_loader(x_val, y_val,yid_val, shuffle=True)
            self.test_loader = self.create_loader(x_test, y_test,yid_test ,shuffle=False)
        elif mode == 'train':
            x_train, y_train,yid_train = self.process_data.process_dataset('/train')
            x_val, y_val,yid_val = self.process_data.process_dataset('/test_small')
            self.train_loader = self.create_loader(x_train, y_train,yid_train, shuffle=True)
            self.val_loader = self.create_loader(x_val, y_val,yid_val, shuffle=True)
        elif mode == 'test':
            x_test, y_test,yid_test = self.process_data.process_dataset('/test')
            self.test_loader = self.create_loader(x_test, y_test,yid_test, shuffle=False)
        

    def create_loader(self, x, y, id, shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        y= np.transpose(y, (0, 3, 1, 2))
        y_id = id
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        id_tensor = torch.from_numpy(y_id).type(torch.LongTensor)
        dataset = TensorDataset(x_tensor, y_tensor, id_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)