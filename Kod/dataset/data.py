import torch
import yaml
import glob
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from skimage.transform import resize
import torch.nn.functional as F
import random

path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            }

path_config = {"place": "lab"
               }



def rgb_to_class_id(mask_rgb, class_colors):
        mask_id = np.zeros(mask_rgb.shape[:2], dtype=np.float32)
        for class_id, color in enumerate(class_colors):
            idx = class_id
            if class_id == 3:
                idx = 1
            mask_id[(mask_rgb == color).all(axis=2)] = idx
        return mask_id

class ProcessData:
    def __init__(self, config_path=path_dict[path_config['place']], mode = 'full',annotator = 1):
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
        
        class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]  # tło, wić, główka

        images = sorted(glob.glob(f"{dataset_path}/images/*"))

        if self.mode == 'intersection_and_union' or self.mode == 'intersection':
            gt_path1 = dataset_path + '/GT1_' + 'mixed/'
            gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            masks = sorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = sorted(glob.glob(f"{gt_path2}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            intersections = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],), dtype=np.float32)
            unions = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],), dtype=np.float32)


            for n, (img, mimg,mimg2) in enumerate(zip(images, masks, masks2)):
                # Load images
                img = cv2.imread(img)
                x_img = img.astype(np.float32)
                x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                # Normalize images
                min_val = np.min(x_img)
                max_val = np.max(x_img)
                x_img = (x_img - min_val) / (max_val - min_val)

                # Load masks
                mask = cv2.imread(mimg)
                mask = mask.astype(np.float32)
                mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)

                mask2 = cv2.imread(mimg2)
                mask2 = mask2.astype(np.float32)
                mask2 = resize(mask2, (512, 512, 3), mode='constant', preserve_range=True)

                intersection = cv2.bitwise_and(mask, mask2)
                union = cv2.bitwise_or(mask, mask2)
                #union = mask2
                #intersection = mask2

                intersection_id = rgb_to_class_id(intersection, class_colors)
                union_id = rgb_to_class_id(union, class_colors)

                # Normalize intersection
                min_val = np.min(intersections)
                max_val = np.max(intersections)

                if (max_val - min_val) > 0:
                    intersection = (intersection - min_val) / (max_val - min_val)
                else:
                    intersection = intersection / 255

                if self.mode == 'intersection_and_union':
                    X[n] = intersection
                else:
                    X[n] = x_img
                intersections[n] = intersection_id
                unions[n] = union_id

            return X, intersections, unions


        else:
            gt_path = dataset_path + name + segment
            masks = sorted(glob.glob(f"{gt_path}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            y = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],), dtype=np.float32)


            for n, (img, mimg) in enumerate(zip(images, masks)):
                # Load images
                img = cv2.imread(img)
                x_img = img.astype(np.float32)
                x_img = resize(x_img, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                # Normalize images
                min_val = np.min(x_img)
                max_val = np.max(x_img)
                x_img = (x_img - min_val) / (max_val - min_val)

                # Load masks
                mask = cv2.imread(mimg)
                mask = mask.astype(np.float32)
                mask = resize(mask, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                mask_id = rgb_to_class_id(mask, class_colors)
                # Save images and masks

                X[n] = x_img
                y[n] = mask_id

            return X, y

class BatchMaker:
    def __init__(self, config_path=path_dict[path_config['place']], batch_size=6, mode = 'all',segment = 'full' ,annotator = 1):
        
    
        self.process_data = ProcessData(config_path=config_path,mode = segment,annotator = annotator)
        self.batch_size = batch_size
        if segment == 'intersection_and_union' or segment == 'intersection':
            if mode == 'all':
                x_train, int_train,un_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val = self.process_data.process_dataset('/test_small')
                x_test, int_test,un_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader2(x_train, int_train,un_train,shuffle=False)
                self.val_loader = self.create_loader2(x_val, int_val,un_val, shuffle=False)
                self.test_loader = self.create_loader2(x_test, int_test,un_test ,shuffle=False)
            elif mode == 'train':
                x_train, int_train,un_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val = self.process_data.process_dataset('/test_small')
                self.train_loader = self.create_loader2(x_train, int_train,un_train,shuffle=True)
                self.val_loader = self.create_loader2(x_val, int_val,un_val, shuffle=True)
            elif mode == 'test':
                x_test, int_test,un_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader2(x_test, int_test,un_test ,shuffle=False)
        else:
            if mode == 'all':
                x_train, y_train = self.process_data.process_dataset('/train')
                x_val, y_val = self.process_data.process_dataset('/test_small')
                x_test, y_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader(x_train, y_train,shuffle=False)
                self.val_loader = self.create_loader(x_val, y_val, shuffle=False)
                self.test_loader = self.create_loader(x_test, y_test ,shuffle=False)
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
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y).type(torch.float64)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    

    def create_loader2(self, x, intersection,union,shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)