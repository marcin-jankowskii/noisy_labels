import cv2
from skimage.transform import resize
from models.Unet import UNet
from dataset.data import BatchMaker
from utils.metrics import SegmentationMetrics
from utils.better_aug import BetterAugmentation
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random


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


class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]]  # tło, wić, główka
number = random.randint(0, 431)
print(f'Number = {number}')
number = 14

img = cv2.imread(f'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Dataset/SegSperm/test/images/{number}.png')
x_img = img.astype(np.float32)
x_img = resize(x_img, (512, 512, 3), mode='constant', preserve_range=True)

# Load masks
mask = cv2.imread(f'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Dataset/SegSperm/test/GT1_mixed/{number}.png')
mask = mask.astype(np.float32)
mask = resize(mask, (512, 512, 3), mode='constant', preserve_range=True)
mask_id = rgb_to_class_id(mask, class_colors)

hmask = cv2.imread(f'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Dataset/SegSperm/test/GT1_head/{number}.png')
hmask = mask.astype(np.float32)
hmask = resize(hmask, (512, 512, 1), mode='constant', preserve_range=True)
# hmask_id = rgb_to_class_id(hmask, class_colors)
min_val = np.min(x_img)
max_val = np.max(x_img)
x_img = (x_img - min_val) / (max_val - min_val)

x_img = x_img.transpose(2, 0, 1)
mask_id = mask_id.transpose(2, 0, 1)
x_img = torch.from_numpy(x_img)
mask_id = torch.from_numpy(mask_id)

augmentation = BetterAugmentation()
input = x_img.unsqueeze(0)
x_img, mask_id = augmentation(x_img, mask_id)

mask_id = mask_id.unsqueeze(0)
mask_id = mask_id.numpy().transpose(0, 2, 3, 1)
mask_to_display = mask_id[0]

colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka

# Utwórz obraz RGB z maski
mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
for class_id, color in enumerate(colors):
    mask_rgb[mask_to_display[:, :, class_id] == 1] = color

kernel = np.ones((3, 3), np.uint8)
mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(5, 5))
plt.imshow(x_img[0].permute(1, 2, 0))
plt.axis('off')
plt.show()
output = x_img[0].permute(1, 2, 0).numpy()

plt.figure(figsize=(5, 5))
plt.imshow(input[0].permute(1, 2, 0))
plt.axis('off')

plt.figure(figsize=(50, 50))
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(x_img[0].permute(1, 2, 0))
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(mask_rgb)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(mask)
plt.show()

cv2.imwrite('/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/example.png', output * 255)
