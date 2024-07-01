import kornia as K
import kornia.augmentation as Ka
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import elasticdeform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class BetterAugmentation(nn.Module):
    def __init__(self):
        super(BetterAugmentation,self).__init__()
        self.k1 = Ka.ColorJitter(brightness=0.2, contrast=0.3, p=0.5)
        self.k2 = Ka.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k3 = Ka.RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k6 = Ka.RandomRotation(45.0, same_on_batch=False, align_corners=True, p=0.5, keepdim=False,
                                    resample='nearest')
        self.k5 = Ka.RandomGaussianBlur((3, 9), (0.1, 2.0), p=0.5)

        self.k4 = Ka.RandomResizedCrop((512, 512), scale=(0.67, 0.67), ratio=(0.75, 1.333), same_on_batch=False,
                                       resample='bilinear', p=0.5, align_corners=True)
        self.resize = Ka.Resize((512, 512))

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:


        img_out, mask_out = self.copy_paste_sperm(img, mask)
        img_out = self.add_lines_along_tail(img_out, mask)
        img_out = self.add_circles(img_out, mask)
        img_out = self.add_blur_along_tail(img_out, mask)
        img_out = self.k1(img_out)
        img_out = self.k2(img_out)
        img_out = self.k3(img_out)
        img_out = self.k4(img_out)
        img_out = self.k5(img_out)
        img_out = self.k6(img_out)
        img_out = self.resize(img_out)
        mask_out = self.k2(mask_out, self.k2._params)
        mask_out = self.k3(mask_out, self.k3._params)
        mask_out = self.k4(mask_out, self.k4._params)
        mask_out = self.k6(mask_out, self.k6._params)
        mask_out = self.resize(mask_out)
        # mask_out = self.k5(mask_out, self.k5._params)
        # img_out, mask_out, mhead_out = self.randomDeform3(img_out, mask_out, mhead)

        return img_out, mask_out

    def find_contours(self, mask: torch.Tensor):
        # Konwersja maski do formatu używanego przez OpenCV
        mask_np = mask.numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def calculate_endpoint(self, start_point, direction, img_size):
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

        if end_point[0] > 511:
            end_point[0] = 511
        elif end_point[0] < 0:
            end_point[0] = 0
        if end_point[1] > 511:
            end_point[1] = 511
        elif end_point[1] < 0:
            end_point[1] = 0

        return end_point

    def calculate_startpoint(self, contour):
        idx = contour.shape[0] // 2
        start_point = torch.tensor(contour[idx].squeeze(), dtype=torch.float32)
        if start_point[0] > 511:
            start_point[0] = 511
        elif start_point[0] < 0:
            start_point[0] = 0
        if start_point[1] > 511:
            start_point[1] = 511
        elif start_point[1] < 0:
            start_point[1] = 0
        return start_point

    def add_lines_along_tail(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        tail_contours = self.find_contours(mask[1, :, :])
        head_contours = self.find_contours(mask[2, :, :])

        # print(tail_contours)

        linesNumber = random.randint(0, 3)
        number = 1

        for contour, t_contour in zip(head_contours, tail_contours):
            if contour.shape[0] > 10 and not None and number <= linesNumber:

                start_point = self.calculate_startpoint(contour)
                idx2 = t_contour.shape[0] // 8
                if idx2 + 1 >= t_contour.shape[0]:
                    max = idx2
                else:
                    max = idx2 + 1
                if idx2 - 1 <= 0:
                    min = 0
                else:
                    min = idx2 - 1
                direction = torch.tensor(t_contour[max] - t_contour[min], dtype=torch.float32).squeeze()
                direction = direction / torch.norm(direction)  # Normalizacja do długości jednostkowej
                end_point1 = self.calculate_endpoint(start_point, direction, 511)
                end_point2 = self.calculate_endpoint(start_point, -direction, 511)
                end_point3 = self.calculate_endpoint(start_point - 1, direction, 511)
                end_point4 = self.calculate_endpoint(start_point - 1, -direction, 511)
                end_point5 = self.calculate_endpoint(start_point - 2, direction, 511)
                end_point6 = self.calculate_endpoint(start_point - 2, -direction, 511)

                color = torch.tensor([0.3, 0.3, 0.3])
                mask = torch.zeros_like(img)
                if not math.isnan(end_point1[0]) and not math.isnan(end_point1[1]) and not math.isnan(
                        end_point2[0]) and not math.isnan(end_point2[1]):
                    mask = K.utils.draw_line(mask, start_point, end_point1, color)
                    mask = K.utils.draw_line(mask, start_point, end_point2, color)
                    mask = K.utils.draw_line(mask, start_point - 1, end_point3, color)
                    mask = K.utils.draw_line(mask, start_point - 1, end_point4, color)
                    mask = K.utils.draw_line(mask, start_point - 2, end_point5, color)
                    mask = K.utils.draw_line(mask, start_point - 2, end_point6, color)

                number += 1

                max = 35
                min = 20
                kernel = random.randint(min, max)
                if kernel % 2 == 0:
                    kernel += 1

                # Convert the mask to a numpy array and blur it
                mask_np = mask.numpy()
                mask_np = cv2.GaussianBlur(mask_np, (kernel, kernel), 0)

                # Convert the blurred mask back to a tensor
                mask_blurred = torch.from_numpy(mask_np)

                # Add the blurred mask to the image
                img = img - mask_blurred

        return img

    def add_circles(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        circleNumber = random.randint(0, 5)
        for _ in range(circleNumber):
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
                sperm_mask = mask[3, :, :]
                overlap = torch.any(sperm_mask[y_clamped, x_clamped] > 0)

            # Rysowanie okręgu, jeśli nie nakłada się na plemniki

            polygon = torch.stack((x, y), dim=-1).unsqueeze(0)
            polygon2 = torch.stack((x - 1, y - 1), dim=-1).unsqueeze(0)
            mean_color = img.float().mean(dim=(1, 2))
            img = img.unsqueeze(0)

            color = mean_color + torch.tensor([0.45, 0.45, 0.45])
            color2 = torch.tensor([1, 1, 1])
            # img = K.utils.draw_convex_polygon(img, polygon2, color2)
            # img = K.utils.draw_convex_polygon(img, polygon, color)

            max = 17
            min = 9
            kernel = random.randint(min, max)
            if kernel % 2 == 0:
                kernel += 1

            # Initialize a new mask with the same size as the image
            mask1 = torch.zeros_like(img)
            mask2 = torch.zeros_like(img)
            mask2 = K.utils.draw_convex_polygon(mask1, polygon2, color2)
            mask1 = K.utils.draw_convex_polygon(mask1, polygon, color)
            # plt.figure(figsize=(50, 50))
            # plt.subplot(1,2,1)
            # plt.imshow(mask1[0].permute(1,2,0))
            mask1_np = mask1.numpy()
            mask1_np = mask1.squeeze().numpy()
            mask1_np = cv2.GaussianBlur(mask1_np, (kernel, kernel), 0)

            mask2_np = mask2.numpy()
            mask2_np = mask2.squeeze().numpy()
            mask2_np = cv2.GaussianBlur(mask2_np, (kernel, kernel), 0)

            mask1 = torch.from_numpy(mask1_np)
            mask2 = torch.from_numpy(mask2_np)

            # plt.subplot(1,2,2)
            # plt.imshow(mask1.permute(1,2,0))

            # Subtract the mask from the image
            img = img + mask1 - mask2

            img = img.squeeze(0)

        return img

    def add_blur_along_tail(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tail_contours = self.find_contours(mask[1, :, :])

        for contour in tail_contours:
            if contour.shape[0] > 10 and not None:

                segments = np.array_split(contour, 20)
                idx = random.randint(1, 4)
                segments_to_blur = random.sample(segments, k=idx)
                expand_by = 5
                for segment in segments_to_blur:
                    if segment.size > 0:
                        start_point = (np.min(segment[:, 0, 0] - expand_by), np.min(segment[:, 0, 1] - expand_by))
                        end_point = (np.max(segment[:, 0, 0] + expand_by), np.max(segment[:, 0, 1] + expand_by))

                        # color = torch.tensor([0, 1, 0])  # Green color in RGB
                        # color = color.view(3, 1, 1).expand(-1, end_point[1] - start_point[1], end_point[0] - start_point[0])
                        # img[:, start_point[1]:end_point[1], start_point[0]:end_point[0]] = color

                        img = img.unsqueeze(0)
                        segment_img = img[:, :, start_point[1]:end_point[1], start_point[0]:end_point[0]].clone()

                        if min(segment_img.shape[2], segment_img.shape[3]) >= 5:
                            blurred_segment = K.filters.box_blur(segment_img, (5, 5))
                            # print("bluruje")
                        else:
                            blurred_segment = segment_img
                        img[:, :, start_point[1]:end_point[1], start_point[0]:end_point[0]] = blurred_segment
                        img = img.squeeze(0)

        return img

    import torch
    import numpy as np
    import cv2
    from sklearn.decomposition import PCA

    def copy_paste_sperm(self, img: torch.Tensor, mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        new_img = img.clone()
        new_mask = mask.clone()
        img_numpy = img.permute(1, 2, 0).numpy()  # Przeniesienie kanałów na ostatni wymiar

        # Ustalenie wektora translacji na podstawie warstwy 3
        translation_vector = None
        sperm_contours = self.find_contours(mask[3, :, :])
        eligible_contours = [contour for contour in sperm_contours if cv2.contourArea(contour) > 10]

        number_to_process = random.randint(0, min(2, len(eligible_contours)))


        for contour in sperm_contours:
            if contour.shape[0] > 10:  # Rozważ kontury o wystarczającej wielkości
                pca = PCA(n_components=2)
                data = contour[:, 0, :]  # Jeśli potrzeba, dostosuj kształt danych
                pca.fit(data)
                main_axis = pca.components_[0]  # Główny kierunek konturu
                translation_vector = np.array([-main_axis[1], -main_axis[0]])  # Obrót o 90 stopni
                break  # Wykorzystaj tylko pierwszy znaleziony kierunek

        if translation_vector is None:
            return img, mask  # Nie znaleziono konturu spełniającego kryteria

        translation_distance = 20  # Dystans translacji
        translation_vector *= translation_distance


        # Przesunięcie konturów na wszystkich warstwach
        for layer in [1, 2, 3]:
            sperm_contours = self.find_contours(mask[layer, :, :])
            i = 0
            for contour in sperm_contours:
                if contour.shape[0] > 10:
                    if i < number_to_process:
                        translated_contour = contour + translation_vector.astype(int)

                        # Tworzenie nowej maski dla przesuniętego konturu
                        translated_mask = np.zeros_like(mask[layer, :, :].numpy())
                        cv2.drawContours(translated_mask, [translated_contour], -1, 255, thickness=cv2.FILLED)

                        # Kopiowanie pikseli z obrazu źródłowego do nowego obrazu
                        original_mask = np.zeros_like(mask[layer, :, :].numpy())
                        cv2.drawContours(original_mask, [contour], -1, 255, thickness=cv2.FILLED)

                        # Iteracja przez piksele w oryginalnej masce
                        for y in range(original_mask.shape[0]):
                            for x in range(original_mask.shape[1]):
                                if original_mask[y, x] == 255:
                                    new_y = y + int(translation_vector[1])
                                    new_x = x + int(translation_vector[0])
                                    if 0 <= new_y < img.shape[1] and 0 <= new_x < img.shape[2]:
                                        new_img[:, new_y, new_x] = img[:, y, x]
                                        new_mask[layer, new_y, new_x] = 1  # Aktualizacja nowej maski
                        i += 1

        return new_img, new_mask
