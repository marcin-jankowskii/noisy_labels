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
        mask_id = np.zeros((*mask_rgb.shape[:2], len(class_colors)), dtype=np.float32)
        for class_id, color in enumerate(class_colors):
            idx = class_id
            if class_id == 3:
                idx = 1
                mask = (mask_rgb == color).all(axis=2).astype(np.float32)
                mask_id[:, :, -1] = np.logical_or(mask_id[:, :, -1], mask)
            else:
                mask = (mask_rgb == color).all(axis=2).astype(np.float32)
                mask_id[:, :, idx] = mask

            if idx == 1 or idx == 2:
                mask_id[:, :, -1] = np.logical_or(mask_id[:, :, -1], mask)

        return mask_id


def feeling_lucky(mask1, mask2):
    # Get the dimensions of the masks
    height, width, num_classes = mask1.shape

    # Randomly choose which annotator's mask to use for each pixel and class
    random_indices = np.random.randint(2, size=(height, width, num_classes))
    result_mask = np.zeros((height, width, num_classes), dtype=mask1.dtype)

    for i in range(height):
        for j in range(width):
            for k in range(num_classes):
                # Choose the mask based on the random index for the given pixel and class
                chosen_mask_value = mask1[i, j, k] if random_indices[i, j, k] == 0 else mask2[i, j, k]
                result_mask[i, j, k] = chosen_mask_value

    return result_mask


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

        if self.mode == 'intersection_and_union' or self.mode == 'intersection' or self.mode == 'intersection_and_union_inference' or self.mode == 'intersection_inference' or self.mode == 'feeling_lucky' or self.mode == 'union':
            gt_path1 = dataset_path + '/GT1_' + 'mixed/'
            gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            masks = sorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = sorted(glob.glob(f"{gt_path2}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            intersections = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            unions = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            feelings = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            y1 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            y2 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)


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
                mask_id = rgb_to_class_id(mask, class_colors)

                mask2 = cv2.imread(mimg2)
                mask2 = mask2.astype(np.float32)
                mask2 = resize(mask2, (512, 512, 3), mode='constant', preserve_range=True)
                mask2_id = rgb_to_class_id(mask2, class_colors)

                intersection = cv2.bitwise_and(mask, mask2)

                intersection_id = cv2.bitwise_and(mask_id, mask2_id)
                union_id = cv2.bitwise_or(mask_id, mask2_id)
                #feeling_id = feeling_lucky(mask_id, mask2_id)
                feeling_id = np.zeros_like(mask_id)

                # Normalize intersection
                min_val = np.min(intersection)
                max_val = np.max(intersection)

                if (max_val - min_val) > 0:
                    intersection = (intersection - min_val) / (max_val - min_val)
                else:
                    intersection = intersection / 255

                if self.mode == 'intersection_and_union' or self.mode == 'intersection_and_union_inference':
                    X[n] = intersection
                else:
                    X[n] = x_img
                intersections[n] = intersection_id
                unions[n] = union_id
                feelings[n] = feeling_id
                y1[n] = mask_id
                y2[n] = mask2_id

            
            if self.mode == 'intersection_and_union_inference' or self.mode == 'intersection_inference':
                return X, intersections, unions,feelings,y1, y2
            elif self.mode == 'intersection_and_union' or self.mode == 'intersection':
                return X, y2, intersections
            elif self.mode == 'feeling_lucky':
                return X, feelings, unions
            elif self.mode == 'union':
                return X, unions,intersections
            

        elif self.mode == 'both':
            gt_path1 = dataset_path + '/GT1_' + 'mixed/'
            gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            masks = sorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = sorted(glob.glob(f"{gt_path2}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            y1 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            y2 = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)


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
                mask_id = rgb_to_class_id(mask, class_colors)
              
                mask2 = cv2.imread(mimg2)
                mask2 = mask2.astype(np.float32)
                mask2 = resize(mask2, (self.config['image_height'], self.config['image_width'], 3), mode='constant', preserve_range=True)
                mask2_id = rgb_to_class_id(mask2, class_colors)
                # Save images and masks

                X[n] = x_img
                y1[n] = mask_id
                y2[n] = mask2_id

            return X, y1,y2
        


        else:
            gt_path = dataset_path + name + segment
            masks = sorted(glob.glob(f"{gt_path}*.png"))

            X = np.zeros((len(images), self.config['image_height'], self.config['image_width'], 3), dtype=np.float32)
            y = np.zeros((len(masks),  self.config['image_height'], self.config['image_width'],4), dtype=np.float32)
            



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
        if segment == 'intersection_and_union' or segment == 'intersection' or segment == 'both' or segment == 'feeling_lucky' or segment == 'union':
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


        elif segment == 'intersection_and_union_inference' or segment == 'intersection_inference':
            if mode == 'all':
                x_train, int_train,un_train,fl_train,y1_train,y2_train = self.process_data.process_dataset('/train')
                x_val, int_val,un_val,y1_val,fl_val,y2_val = self.process_data.process_dataset('/test_small')
                x_test, int_test,un_test,y1_test,fl_test,y2_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader3(x_train, int_train,un_train,fl_train,y1_train,y2_train,shuffle=False)
                self.val_loader = self.create_loader3(x_val, int_val,un_val,fl_val,y1_val,y2_val, shuffle=False)
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test ,shuffle=False)
            elif mode == 'train':
                x_train, int_train,un_train,fl_train,y1_train,y2_train = self.process_data.process_dataset('/train')
                x_test, int_test,un_test,y1_test,fl_test,y2_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader3(x_train, int_train,un_train,fl_train,y1_train,y2_train,shuffle=True)
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test ,shuffle=True)
            elif mode == 'test':
                x_test, int_test,un_test,fl_test,y1_test,y2_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader3(x_test, int_test,un_test,fl_test,y1_test,y2_test ,shuffle=False)


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
                x_test, y_test = self.process_data.process_dataset('/test')
                self.train_loader = self.create_loader(x_train, y_train, shuffle=True)
                self.test_loader = self.create_loader(x_test, y_test, shuffle=True)
            elif mode == 'test':
                x_test, y_test = self.process_data.process_dataset('/test')
                self.test_loader = self.create_loader(x_test, y_test, shuffle=False)
        

    def create_loader(self, x, y, shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        y = np.transpose(y, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y).type(torch.float64)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    

    def create_loader2(self, x, intersection,union,shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        intersection = np.transpose(intersection, (0, 3, 1, 2))
        union = np.transpose(union, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def create_loader3(self,x,intersection,union,feeling,y1,y2,shuffle):
        x = np.transpose(x,(0,3,1,2))
        intersection = np.transpose(intersection, (0, 3, 1, 2))
        union = np.transpose(union, (0, 3, 1, 2))
        feeling = np.transpose(feeling, (0, 3, 1, 2))
        y1 = np.transpose(y1, (0, 3, 1, 2))
        y2 = np.transpose(y2, (0, 3, 1, 2))   
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        feeling_tensor = torch.from_numpy(feeling).type(torch.float64)
        y1_tensor = torch.from_numpy(y1).type(torch.float64)
        y2_tensor = torch.from_numpy(y2).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor,feeling_tensor,y1_tensor,y2_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        