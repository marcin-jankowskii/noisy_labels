import kornia as K
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class MyAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = K.augmentation.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0.25)
        #self.k2 = K.augmentation.RandomAffine([0, 0], [0, 0], [0, 0], [0, 0])

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    
        img_out = self.add_lines_along_tail(img,mask)
        img_out = self.add_circles(img_out,mask)
        img_out = self.k1(img_out)
        #mask_out = self.k2(mask, self.k2._params)

        return img_out


    def find_contours(self,mask: torch.Tensor):
        # Konwersja maski do formatu używanego przez OpenCV
        mask_np = mask.numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def calculate_endpoint(self,start_point, direction, img_size):
        dir_x, dir_y = direction[0].item(), direction[1].item()
        ratios = [float('inf')] * 4

        if dir_x > 0:
            ratios[2] = (img_size - 1 - start_point[0]) / dir_x
        elif dir_x < 0:
            ratios[3] = -start_point[0] / dir_x

        if dir_y > 0:
            ratios[0] = (img_size - 1 - start_point[1]) / dir_y
        elif dir_y < 0:
            ratios[1] = -start_point[1] / dir_y

        min_ratio = min(ratios)
        end_point = start_point + min_ratio * direction

        return end_point

    def add_lines_along_tail(self,img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tail_contours = self.find_contours(mask == 1)

        for contour in tail_contours:
            if contour.shape[0] > 10 and not None:
                idx = contour.shape[0] // 8
                start_point = torch.tensor(contour[idx].squeeze(), dtype=torch.float32)
                start_point[0] = start_point[0] -5
                if start_point[0] > 511:
                    start_point[0] = 511
                elif start_point[0] < 0:
                    start_point[0] = 0
                if start_point[1] > 511:
                    start_point[1] = 511
                elif start_point[1] < 0:
                    start_point[1] = 0
                # Obliczanie kierunku linii
                direction = torch.tensor(contour[idx + 1] - contour[idx - 1], dtype=torch.float32).squeeze()
                direction = direction / torch.norm(direction)  # Normalizacja do długości jednostkowej

                end_point1 = self.calculate_endpoint(start_point, direction, 511)
                if end_point1[0] > 511:
                    end_point1[0] = 511
                elif end_point1[0] < 0:
                    end_point1[0] = 0
                if end_point1[1] > 511:
                    end_point1[1] = 511
                elif end_point1[1] < 0:
                    end_point1[1] = 0
                
                end_point2 = self.calculate_endpoint(start_point, -direction, 511)
                if end_point2[0] > 511:
                    end_point2[0] = 511
                elif end_point2[0] < 0:
                    end_point2[0] = 0
                if end_point2[1] > 511:
                    end_point2[1] = 511
                elif end_point2[1] < 0:
                    end_point2[1] = 0

                color = torch.tensor([0.2, 0.2, 0.2])

                if not math.isnan(end_point1[0]) and not math.isnan(end_point1[1]) and not math.isnan(end_point2[0]) and not math.isnan(end_point2[1]):
                    img = K.utils.draw_line(img, start_point, end_point1, color)
                    img = K.utils.draw_line(img, start_point, end_point2, color)

        return img

    def add_circles(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for _ in range(10):
            overlap = True
            while overlap:
                center_coordinates = torch.randint(0, img.shape[1], (2,)).float()
                radius_x = torch.tensor(random.randint(5, 15)).float()
                radius_y = torch.tensor(random.randint(5, 15)).float()
                num_sides = 100
                theta = torch.linspace(0, 2 * np.pi, num_sides)
                x = center_coordinates[0] + radius_x * torch.cos(theta)
                y = center_coordinates[1] + radius_y * torch.sin(theta)

                # Sprawdzanie, czy okrąg nakłada się na plemniki
                x_clamped = torch.clamp(x.long(), 0, mask.shape[0] - 1)
                y_clamped = torch.clamp(y.long(), 0, mask.shape[1] - 1)
                overlap = torch.any(mask[y_clamped, x_clamped] > 0)

            # Rysowanie okręgu, jeśli nie nakłada się na plemniki
            polygon = torch.stack((x, y), dim=-1).unsqueeze(0)
            img = img.unsqueeze(0)
            color = torch.tensor([0.2, 0.2, 0.2])
            img = K.utils.draw_convex_polygon(img, polygon, color)
            img = img.squeeze(0)

        return img

     