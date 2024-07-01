from models.Unet import UNet
from dataset.data import BatchMaker
import pandas as pd
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
import datetime
from utils.metrics import SegmentationMetrics
from utils.metrics2 import calculate_iou, calculate_ap_for_segmentation
import segmentation_models_pytorch as smp
import cv2
import os


def transform_mask(mask):
    new_mask = np.zeros((512, 512))
    for i in range(3):
        new_mask[mask[i] == 1] = i

    return new_mask

def transform_mask2(mask):
    new_mask = np.zeros((512, 512))
    for i in range(4):
        if i == 0:
            new_mask[mask[i] == 1] = i
        if i == 3:
            new_mask[mask[i] == 1] = 1
    return new_mask

def transform_batch(batch):
    new_batch = np.zeros((batch.shape[0], 512, 512))

    # Przypisanie wartości 1, 2, 3 do odpowiednich warstw dla każdej maski w batchu
    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask(batch[i])

    return new_batch

def transform_batch2(batch):
    # Tworzenie nowego batcha o wymiarach (500, 512, 512)
    new_batch = np.zeros((batch.shape[0], 512, 512))

    # Przypisanie wartości 1, 2, 3 do odpowiednich warstw dla każdej maski w batchu
    for i in range(batch.shape[0]):
        new_batch[i] = transform_mask2(batch[i])

    return new_batch
def plot_sample(X, y, preds, ix=None, number = '1'):

    """Function to plot the results"""
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 5,figsize=(20, 10))
    ax[0].imshow(X[ix])
    #if has_mask:
        #ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sperm Image')
    ax[0].set_axis_off()


    mask_to_display = y[ix]

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color


    ax[1].imshow(mask_rgb)
    ax[1].set_title('Sperm Mask Image')
    ax[1].set_axis_off()

    mask_to_display = preds[ix]

    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color
 

    ax[2].imshow(mask_rgb)
    #if has_mask:
        #ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Sperm Image Predicted')
    ax[2].set_axis_off()


    # Utwórz obraz RGB z maski
    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colorsV2):
        if class_id == 0 or class_id == 3:
            mask_rgb2[mask_to_display[:, :, class_id] == 1] = color


    ax[3].imshow(mask_rgb2)
    ax[3].set_title('Sperm Image Predicted (class3)')
    ax[3].set_axis_off()


    # Utwórz obraz RGB z maski
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        if class_id == 1 or class_id == 2:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = [0,255,255]
        else:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    ax[4].imshow(mask_rgb-mask_rgb2)
    ax[4].set_title('Diff (class1,2 - class3)')
    ax[4].set_axis_off()

    
    plt.savefig(yaml_config['save_inf_fig_path']+'/{}.png'.format(ix))
    if number == '1':
        wandb.log({"inference/plot": wandb.Image(fig)})
    else:
        wandb.log({"inference/plot"+number: wandb.Image(fig)})
    plt.close()

def video_sample(true,pred,ix=None):
    if ix is None:
        ix = random.randint(0, len(true))

    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    
    # mask - pred
    mask_to_display = true[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_to_display = pred[ix]

    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb2[mask_to_display[:, :, class_id] == 1] = color

    union = cv2.bitwise_or(mask_rgb, mask_rgb2)
    diff = (union - mask_rgb2).transpose((2, 0, 1))


    # Róznica pomiędzy klasami 1,2 a 3
    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        if class_id == 1 or class_id == 2:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = [0,255,255]
        else:
            mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_rgb2 = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colorsV2):
        if class_id == 0 or class_id == 3:
            mask_rgb2[mask_to_display[:, :, class_id] == 1] = color

    union = cv2.bitwise_or(mask_rgb, mask_rgb2)
    diff2 = (union - mask_rgb2).transpose((2, 0, 1))


    
    
    return diff,diff2
    
def video_sample2(intersection,union,pred,ix=None):
    if ix is None:
        ix = random.randint(0, len(intersection))

    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka
    colorsV2 = [[0, 0, 0], [0, 255, 0], [255, 0, 0],[0,255,255]]  # tło, wić, główka, główka+wić

    ## (Union - Intersection) - (pred - Intersection)
    mask_to_display = intersection[ix]

    mask_rgb = np.zeros((mask_to_display.shape[0], mask_to_display.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb[mask_to_display[:, :, class_id] == 1] = color

    mask_to_display2 = union[ix]

    mask_rgb2 = np.zeros((mask_to_display2.shape[0], mask_to_display2.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb2[mask_to_display2[:, :, class_id] == 1] = color


    mask_to_display3 = pred[ix]

    mask_rgb3 = np.zeros((mask_to_display3.shape[0], mask_to_display3.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        mask_rgb3[mask_to_display3[:, :, class_id] == 1] = color

    union1 = cv2.bitwise_or(mask_rgb, mask_rgb2)
    union2 = cv2.bitwise_or(mask_rgb3, mask_rgb)
    udiff = union1 - mask_rgb
    pdiff = union2 - mask_rgb
    union3 = cv2.bitwise_or(udiff, pdiff)
    diff = (union3 - pdiff).transpose((2, 0, 1))

    return diff

def save_numpy_files(folder_name, file1, file2):
    folder_name = "/media/cal314-1/9E044F59044F3415/Marcin/Data/Optimal Treshold/" + folder_name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    np.save(os.path.join(folder_name, 'file1.npy'), file1)
    np.save(os.path.join(folder_name, 'file2.npy'), file2)


def inference(ximages,mask_numpy,mask_plot,pred_numpy,pred_plot,number='1',mask_name='mask',video_name='video',class_diff=False):
    #pred numpy - predykcja multiclass
    #pred plot - predykcja one class

    ids1 = transform_batch(mask_numpy)
    ids2 = transform_batch2(mask_numpy)

    unique_values = np.unique(pred_numpy)
    multiclass = len(unique_values[unique_values != 0]) > 1

    ious_multiclass = []
    ious_oneclass = []



    if multiclass:
        mean_iou_multiclass, iou_per_class_multiclass = calculate_iou(ids1,pred_numpy,3)
        for i in range(ids1.shape[0]):
            iou_multiclass, iou_multiclasssss = calculate_iou(ids1[i], pred_numpy[i], 3)
            iou = iou_multiclass
            ious_multiclass.append(iou)


    else:
        mean_iou_multiclass = 0
        iou_per_class_multiclass = 0
        ious_multiclass = 0

    mean_iou_oneclass, iou_per_class_oneclass = calculate_iou(ids2, pred_plot, 2)
    for i in range(ids2.shape[0]):
        iou_oneclass, iou_oneclasssss = calculate_iou(ids2[i], pred_plot[i], 2)
        ious_oneclass.append(iou_oneclass)


    #mean_iou, iou_per_class = calculate_iou(mask_numpy, pred_numpy)
    print(f"Mean IoU oneclass ({mask_name}):", mean_iou_oneclass)
    print(f"IoU per class oneclass ({mask_name}):", iou_per_class_oneclass)
    print(f"Mean IoU multiclass ({mask_name}):", mean_iou_multiclass)
    print(f"IoU per class multiclass ({mask_name}):", iou_per_class_multiclass)
    # diff = []
    # diff2 = []
    # for i in range(len(ximages)):
    #     plot_sample(ximages, mask_plot, pred_plot, ix=i, number=number)
    #     # df1, df2 = video_sample(mask_plot, pred_plot, ix=i)
    #     # diff.append(df1)
    #     # if class_diff:
    #     #     diff2.append(df2)
    #     print('sample {} saved'.format(i))
    # diff = np.array(diff)
    # wandb.log({f"video_pdiff({video_name})": wandb.Video(diff, fps=1)})
    # if class_diff:
    #     diff2 = np.array(diff2)
    #     wandb.log({f"video_cdiff({video_name})": wandb.Video(diff2, fps=1)})

    return mean_iou_oneclass, iou_per_class_oneclass, mean_iou_multiclass, iou_per_class_multiclass,ious_multiclass,ious_oneclass

def calculate_optimal_threshold(x_images,mask_numpy,softmask_multiclass,softmask_oneclass,pred_multiclass,name, intersection):

    ids1 = transform_batch(mask_numpy)
    ids2 = transform_batch2(mask_numpy)

    inter_multi = transform_batch(intersection)
    inter_one = transform_batch2(intersection)


    unique_values = np.unique(pred_multiclass)
    multiclass = len(unique_values[unique_values != 0]) > 1

    softmask_multiclass_np = np.array(softmask_multiclass)
    softmask_oneclass_np = np.array(softmask_oneclass)
    softmask_oneclass_np = softmask_oneclass_np.transpose((0, 2, 3, 1))
    softmask_multiclass_np = softmask_multiclass_np.transpose((0, 2, 3, 1))
    images_np = np.array(x_images)


    folder_path = os.path.join(yaml_config['save_soft_mask_path'], name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # if not os.path.exists(folder_path+'/GT'):
    #     os.makedirs(folder_path+'/GT')
    # if not os.path.exists(folder_path + '/GT_head'):
    #     os.makedirs(folder_path + '/GT_head')
    # if not os.path.exists(folder_path + '/GT_tail'):
    #     os.makedirs(folder_path + '/GT_tail')
    if not os.path.exists(folder_path+'/images'):
        os.makedirs(folder_path+'/images')
    if not os.path.exists(folder_path+'/head'):
        os.makedirs(folder_path+'/head')
    if not os.path.exists(folder_path+'/tail'):
        os.makedirs(folder_path+'/tail')
    if not os.path.exists(folder_path + '/multiclass'):
        os.makedirs(folder_path + '/multiclass')

    for i in range(softmask_oneclass_np.shape[0]):
        img = 255*softmask_oneclass_np[i]
        softmaskOneclass = img[:, :, 1]
        GT = 255*ids2[i]
        # GT_head = mask_numpy[i,2,:,:]*255
        # GT_tail = mask_numpy[i,1,:,:]*255
        realImage = images_np[i]*255
        if multiclass:
            softmaskMulticlass = 255*softmask_multiclass_np[i]
            img_head = softmaskMulticlass[:, :, 2]
            img_tail = softmaskMulticlass[:, :, 1]
            img_multiclass = img_head + img_tail
            cv2.imwrite(os.path.join(folder_path + '/head', f'{i}.png'), img_head)
            cv2.imwrite(os.path.join(folder_path + '/tail', f'{i}.png'), img_tail)
            cv2.imwrite(os.path.join(folder_path + '/multiclass', f'{i}.png'), img_multiclass)

        cv2.imwrite(os.path.join(folder_path, f'{i}.png'), softmaskOneclass)
        cv2.imwrite(os.path.join(folder_path+'/GT', f'{i}.png'), GT)
        cv2.imwrite(os.path.join(folder_path + '/images', f'{i}.png'), realImage)
        # cv2.imwrite(os.path.join(folder_path + '/GT_head', f'{i}.png'), GT_head)
        # cv2.imwrite(os.path.join(folder_path + '/GT_tail', f'{i}.png'), GT_tail)


    thresholds = torch.linspace(0, 1, steps=255)
    iou_scores_multiclass = []
    iou_scores_oneclass = []
    iou_scores_head = []
    iou_scores_tail = []
    iou_scores_merge = []

    if multiclass:
        softmask_tail_np1 = np.array(softmask_multiclass_np[:,:,:,1])
        softmask_head_np1 = np.array(softmask_multiclass_np[:,:,:,2])
        softmask_merge_multiclass = softmask_head_np1 + softmask_tail_np1

    i = 0
    for threshold in thresholds:
        if multiclass:
            pred_merge_multiclass = (softmask_merge_multiclass > threshold.numpy()).astype(float)
            if config.mode == 'intersection_and_union_inference':
                pred_merge_multiclass = np.maximum(pred_merge_multiclass, inter_multi)
            iou_merge_multiclass, i_merge_multiclass = calculate_iou(pred_merge_multiclass, ids2, 2)

            pred_masks_multiclass = (softmask_multiclass > threshold.numpy()).astype(float)
            pred_masks_multiclass_tail = pred_masks_multiclass[:, 1, :, :]
            pred_masks_multiclass_head = pred_masks_multiclass[:, 2, :, :]
            final_mask = np.zeros_like(pred_masks_multiclass_head)
            conflict = (pred_masks_multiclass_head == 1) & (pred_masks_multiclass_tail == 1)
            tail_conflict_values = softmask_multiclass_np[:, :, :, 1][conflict]
            head_conflict_values = softmask_multiclass_np[:, :, :, 2][conflict]
            final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(int)  # 1 dla tail, 2 dla head
            final_mask[(pred_masks_multiclass_tail == 1) & ~conflict] = 1
            final_mask[(pred_masks_multiclass_head == 1) & ~conflict] = 2
            if config.mode == 'intersection_and_union_inference':
                final_mask = np.maximum(final_mask, inter_multi)
                pred_masks_multiclass_head = np.maximum(pred_masks_multiclass_head, intersection[:,2,:,:])
                pred_masks_multiclass_tail = np.maximum(pred_masks_multiclass_tail, intersection[:,1,:,:])

            iou_multiclass, i_multiclass = calculate_iou(final_mask, ids1, 3)
            iou_head, i_head = calculate_iou(pred_masks_multiclass_head, mask_numpy[:,2,:,:], 2)
            iou_tail, i_tail = calculate_iou(pred_masks_multiclass_tail, mask_numpy[:,1,:,:], 2)
        else:
            iou_multiclass = 0
            i_multiclass = 0
            iou_head = 0
            i_head = 0
            iou_tail = 0
            i_tail = 0
            iou_merge_multiclass = 0
        pred_masks_oneclass = (softmask_oneclass > threshold.numpy()).astype(float)
        pred_masks_oneclass = pred_masks_oneclass[:, 1, :, :]

        if config.mode == 'intersection_and_union_inference':
            pred_masks_oneclass = np.maximum(pred_masks_oneclass, inter_one)
        iou_oneclass,i_oneclass = calculate_iou(pred_masks_oneclass, ids2,2)

        iou_scores_multiclass.append(iou_multiclass)
        iou_scores_oneclass.append(iou_oneclass)
        iou_scores_head.append(iou_head)
        iou_scores_tail.append(iou_tail)
        iou_scores_merge.append(iou_merge_multiclass)
        i+=1
        print(f"Iou {i} calculated")

    ap_score = calculate_ap_for_segmentation(softmask_oneclass_np[:, :, :,1], ids2)

    optimal_threshold_multiclass = thresholds[torch.argmax(torch.tensor(iou_scores_multiclass))]
    optimal_iou_multiclass = max(iou_scores_multiclass)
    print(f"Optimal threshold for multiclass: {optimal_threshold_multiclass}, IoU: {optimal_iou_multiclass}")

    optimal_threshold_oneclass = thresholds[torch.argmax(torch.tensor(iou_scores_oneclass))]
    optimal_iou_oneclass = max(iou_scores_oneclass)
    print(f"Optimal threshold for oneclass: {optimal_threshold_oneclass}, IoU: {optimal_iou_oneclass}")

    optimal_iou_head = max(iou_scores_head)
    optimal_iou_head_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_head))]
    print(f"Optimal threshold for head: {optimal_iou_head_threshold}, IoU: {optimal_iou_head}")

    optimal_iou_tail = max(iou_scores_tail)
    optimal_iou_tail_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_tail))]
    print(f"Optimal threshold for tail: {optimal_iou_tail_threshold}, IoU: {optimal_iou_tail}")

    optimal_iou_merge = max(iou_scores_merge)
    optimal_iou_merge_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_merge))]
    print(f"Optimal threshold for merge: {optimal_iou_merge_threshold}, IoU: {optimal_iou_merge}")


    if multiclass:
        softmask_head = softmask_multiclass_np[:, :, :, 2]
        pred_head = (softmask_head > optimal_iou_head_threshold.numpy()).astype(float)
        softmask_tail = softmask_multiclass_np[:, :, :, 1]
        pred_tail = (softmask_tail > optimal_iou_tail_threshold.numpy()).astype(float)
        softmask_merge_multiclass = softmask_tail + softmask_head


        final_mask = np.zeros_like(pred_head)
        conflict = (pred_head == 1) & (pred_tail == 1)
        tail_conflict_values = softmask_multiclass_np[:, :, :, 1][conflict]
        head_conflict_values = softmask_multiclass_np[:, :, :, 2][conflict]
        final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(int)  # 1 dla tail, 2 dla head
        final_mask[(pred_tail == 1) & ~conflict] = 1
        final_mask[(pred_head == 1) & ~conflict] = 2



        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # for i in range(final_mask.shape[0]):
        #     final_mask[i] = cv2.morphologyEx(final_mask[i], cv2.MORPH_CLOSE, kernel)

        pred_head2 = (softmask_head > optimal_threshold_multiclass.numpy()).astype(float)
        pred_tail2 = (softmask_tail > optimal_threshold_multiclass.numpy()).astype(float)
        pred_merge_multiclass = (softmask_merge_multiclass > optimal_iou_merge_threshold.numpy()).astype(float)
        final_mask2 = np.zeros_like(pred_head2)
        conflict2 = (pred_head2 == 1) & (pred_tail2 == 1)
        tail_conflict_values2 = softmask_multiclass_np[:, :, :, 1][conflict2]
        head_conflict_values2 = softmask_multiclass_np[:, :, :, 2][conflict2]
        final_mask2[conflict2] = 2 - (tail_conflict_values2 > head_conflict_values2).astype(int)  # 1 dla tail, 2 dla head
        final_mask2[(pred_tail2 == 1) & ~conflict2] = 1
        final_mask2[(pred_head2 == 1) & ~conflict2] = 2

        if config.mode == 'intersection_and_union_inference':
            final_mask = np.maximum(final_mask, inter_multi)
            pred_head = np.maximum(pred_head, intersection[:, 2, :, :])
            pred_tail = np.maximum(pred_tail, intersection[:, 1, :, :])
            final_mask2 = np.maximum(final_mask2, inter_multi)
            pred_merge_multiclass = np.maximum(pred_merge_multiclass, inter_multi)



        if not os.path.exists(folder_path + '/masks/best_tresholds_multiclass'):
            os.makedirs(folder_path + '/masks/best_tresholds_multiclass')
        for i in range(final_mask.shape[0]):
            slice = final_mask[i]
            color_image = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
            color_image[slice == 1] = [0, 255, 0]
            color_image[slice == 2] = [0, 0, 255]
            cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_multiclass', f'{i}.png'), color_image)

        if not os.path.exists(folder_path + '/masks/best_for_multiclass'):
            os.makedirs(folder_path + '/masks/best_for_multiclass')
        for i in range(final_mask2.shape[0]):
            slice = final_mask2[i]
            color_image = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
            color_image[slice == 1] = [0, 255, 0]
            color_image[slice == 2] = [0, 0, 255]
            cv2.imwrite(os.path.join(folder_path + '/masks/best_for_multiclass', f'{i}.png'), color_image)


        if not os.path.exists(folder_path + '/masks/best_tresholds_head'):
            os.makedirs(folder_path + '/masks/best_tresholds_head')
        for i in range(pred_head.shape[0]):
            cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_head', f'{i}.png'),
                        pred_head[i] * 255)

        if not os.path.exists(folder_path + '/masks/best_tresholds_tail'):
            os.makedirs(folder_path + '/masks/best_tresholds_tail')
        for i in range(pred_tail.shape[0]):
            cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_tail', f'{i}.png'),
                        pred_tail[i] * 255)

        if not os.path.exists(folder_path + '/masks/best_tresholds_merge'):
            os.makedirs(folder_path + '/masks/best_tresholds_merge')
        for i in range(pred_merge_multiclass.shape[0]):
            cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_merge', f'{i}.png'),
                        pred_merge_multiclass[i] * 255)

        iou_multiclass_best_tresholds, i_multiclass_best_thresholds = calculate_iou(final_mask, ids1, 3)
    else:
        iou_multiclass_best_tresholds = 0
        i_multiclass2_best_thresholds = 0

    softmask_oneclass2 = softmask_oneclass_np[:, :, :, 1]
    pred_oneclass2 = (softmask_oneclass2 > optimal_threshold_oneclass.numpy()).astype(float)
    if config.mode == 'intersection_and_union_inference':
        pred_oneclass2 = np.maximum(pred_oneclass2, inter_one)
    if not os.path.exists(folder_path + '/masks/best_tresholds_oneclass'):
        os.makedirs(folder_path + '/masks/best_tresholds_oneclass')
    for i in range(pred_oneclass2.shape[0]):
        cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_oneclass', f'{i}.png'), pred_oneclass2[i]*255)

    if multiclass:
        full_agreement_mask = np.zeros_like(pred_masks_multiclass_head)
        full_agreement_mask[(pred_head == 1) & (pred_oneclass2 == 1)] = 2
        full_agreement_mask[(pred_tail == 1) & (pred_oneclass2 == 1)] = 1

        print(f"Full agreement mask:")
        if not os.path.exists(folder_path + '/masks/best_tresholds_full_agreement'):
            os.makedirs(folder_path + '/masks/best_tresholds_full_agreement')
        for i in range(full_agreement_mask.shape[0]):
            cv2.imwrite(os.path.join(folder_path + '/masks/best_tresholds_full_agreement', f'{i}.png'),full_agreement_mask[i] * 255)
        iou_full_agreement, i_full_agreement = calculate_iou(full_agreement_mask, ids1, 3)

    ious_multiclass = []
    ious_oneclass = []
    ious_head = []
    ious_tail = []
    ious_full_agreement = []

    if multiclass:
        for i in range(ids1.shape[0]):
            iou_multiclass, iou_multiclasssss = calculate_iou(ids1[i], final_mask[i], 3)
            iou_head, i_head = calculate_iou(mask_numpy[i, 2, :, :],pred_head[i], 2)
            iou_tail, i_tail = calculate_iou(mask_numpy[i, 1, :, :],pred_tail[i], 2)
            iou = iou_multiclass
            ious_multiclass.append(iou)
            ious_head.append(iou_head)
            ious_tail.append(iou_tail)
    else:
        mean_iou_multiclass = 0
        iou_per_class_multiclass = 0
        ious_multiclass = 0
        ious_head = 0
        ious_tail = 0

    for i in range(ids2.shape[0]):
        iou_oneclass, iou_oneclasssss = calculate_iou(ids2[i], pred_oneclass2[i], 2)
        ious_oneclass.append(iou_oneclass)

    if not multiclass:
        optimal_iou_head = 0
        optimal_iou_tail = 0
        optimal_iou_head_threshold = 0
        optimal_iou_tail_threshold = 0
        iou_multiclass_best_tresholds = 0
        i_multiclass2_best_thresholds = 0

    save_numpy_files(name + "multiclass",optimal_threshold_multiclass,optimal_iou_multiclass)
    save_numpy_files(name + "oneclass",optimal_threshold_oneclass,optimal_iou_oneclass)

    # # Wykres IoU od threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds.numpy(), iou_scores_multiclass, label='IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')
    plt.title('IoU Score as a function of Threshold (multiclass)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds.numpy(), iou_scores_oneclass, label='IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')
    plt.title('IoU Score as a function of Threshold (one class)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return optimal_threshold_multiclass,optimal_iou_multiclass,optimal_threshold_oneclass,optimal_iou_oneclass,ap_score,optimal_iou_head,optimal_iou_tail,optimal_iou_head_threshold,optimal_iou_tail_threshold,iou_multiclass_best_tresholds,ious_oneclass,ious_multiclass,ious_head,ious_tail, optimal_iou_merge, optimal_iou_merge_threshold,iou_full_agreement

def predict(model, test_loader):
    model.eval() 
    input_images = []
    predicted_masks_oneclass = []
    predicted_masks_multiclass = []
    predicted_softmasks_oneclass = []
    predicted_softmasks_multiclass = []
    true_masks = []
    true_masks2 = []
    unions_list =[]
    intersections_list = []
    feelings_list = []
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 2:
                inputs, ids = data
            elif len(data) == 3:
                if config.mode == 'intersection_and_union' or config.mode == 'intersection': 
                    inputs, intersections, unions = data
                else:
                    inputs, ids1, ids2 = data
            elif len(data) == 6:
                inputs,intersections, unions,feelings,ids1, ids2 = data

            inputs = inputs.to(device)
            outputs = model(inputs)
            output1 = outputs[:, :3, :, :]
            output2 = outputs[:, [0, -1], :, :]

            if len(data) == 2:
                true_masks.append(ids.cpu())  
            elif len(data) == 3:
                if config.mode == 'intersection_and_union':
                    true_masks.append(unions.cpu())  
                    true_masks2.append(intersections.cpu())
                elif config.mode == 'intersection':
                    true_masks.append(intersections.cpu())
                elif config.mode == 'both':
                    true_masks.append(ids1.cpu())
                    true_masks2.append(ids2.cpu())
            elif len(data) == 6:
                if config.mode == 'intersection_and_union_inference' or config.mode == 'intersection_inference':
                    true_masks.append(ids1.cpu())  
                    true_masks2.append(ids2.cpu())
                    unions_list.append(unions.cpu())
                    intersections_list.append(intersections.cpu())
                    feelings_list.append(feelings.cpu())


            input_images.append(inputs.cpu())
            preds1 = torch.argmax(output1, dim=1)
            preds2 = torch.argmax(output2, dim=1)
            softs1 = torch.softmax(output1, dim=1)
            softs1 = softs1.squeeze(0)
            softs2 = torch.softmax(output2, dim=1)
            softs2 = softs2.squeeze(0)

            predicted_masks_multiclass.append(preds1.cpu())
            predicted_masks_oneclass.append(preds2.cpu())
            predicted_softmasks_multiclass.append(softs1.cpu())
            predicted_softmasks_oneclass.append(softs2.cpu())
            #predicted_masks.append(outputs_binary.cpu())

    input_images = np.concatenate(input_images, axis=0)
    true_masks = np.concatenate(true_masks, axis=0)
    #predicted_masks = np.concatenate(predicted_masks, axis=0)
    predicted_masks_multiclass = np.concatenate(predicted_masks_multiclass, axis=0)
    predicted_masks_oneclass = np.concatenate(predicted_masks_oneclass, axis=0)
    print(predicted_masks_multiclass.shape)


    x_images = input_images.transpose((0, 2, 3, 1))
    true = true_masks.transpose((0, 2, 3, 1))
    pred1 = predicted_masks_multiclass #.transpose((0, 2, 3, 1))
    pred2 = predicted_masks_oneclass #.transpose((0, 2, 3, 1))
    soft1 = predicted_softmasks_multiclass
    soft2 = predicted_softmasks_oneclass


    if config.mode == 'intersection_inference' or config.mode == 'intersection_and_union_inference':
        true_masks2 = np.concatenate(true_masks2, axis=0)
        true2 = true_masks2.transpose((0, 2, 3, 1))
        unions_numpy = np.concatenate(unions_list, axis=0)
        unions_plot = unions_numpy.transpose((0, 2, 3, 1))
        intersections_numpy = np.concatenate(intersections_list, axis=0)
        intersections_plot = intersections_numpy.transpose((0, 2, 3, 1))
        feeling_numpy = np.concatenate(feelings_list, axis=0)
        feeling_plot = feeling_numpy.transpose((0, 2, 3, 1))
        if config.mode == 'intersection_and_union_inference':
            inter_multi = transform_batch(intersections_numpy)
            inter_one = transform_batch2(intersections_numpy)
            pred1 = np.maximum(inter_multi, pred1)
            pred2 = np.maximum(inter_one, pred2)

        mean_iou_ids1_oneclass, iou_ids1_oneclass,mean_iou_ids1_multiclass, iou_ids1_multiclass, ious_multiclass_annotator1, ious_oneclass_annotator1 = inference(x_images, true_masks, true, pred1, pred2, number='2',
                                            mask_name='annotator1', video_name='ids1', class_diff=False)



        mean_iou_ids2_oneclass, iou_ids2_oneclass,mean_iou_ids2_multiclass, iou_ids2_multiclass, ious_multicass_annotator2, ious_oneclass_annotator2 = inference(x_images, true_masks2, true2, pred1, pred2, number='3',
                                            mask_name='annotator2', video_name='ids2', class_diff=False)

        mean_iou_intersection_oneclass, iou_intersection_oneclass, mean_iou_intersection_multiclass, iou_intersection_multiclass, ious_multiclass_intersection, ious_oneclass_intersection = inference(
            x_images, intersections_numpy, intersections_plot,
            pred1, pred2, number='1',
            mask_name='intersection', video_name='inters',
            class_diff=False)

        mean_iou_union_oneclass, iou_union_oneclass,mean_iou_union_multiclass, iou_union_multiclass, ious_multiclass_union, ious_oneclass_union = inference(x_images, unions_numpy, unions_plot, pred1, pred2, number='4',
                                              mask_name='union', video_name='union', class_diff=True)

        mean_iou_feeling_oneclass, iou_feeling_oneclass,mean_iou_feeling_multiclass, iou_feeling_multiclass,ious_multiclass_lucky,ious_oneclass_lucky = inference(x_images, feeling_numpy, feeling_plot, pred1, pred2,
                                                  number='5',
                                                  mask_name='feeling lucky', video_name='feelings', class_diff=False)

        optimal_threshold_multiclass,optimal_iou_multiclass,optimal_threshold_oneclass,optimal_iou_oneclass,ap_score_oneclass,optimal_iou_head,optimal_iou_tail,optimal_head_treshold,optimal_tail_threshold,iou_multiclass_best_tresholds,ious_opt_oneclass_annotator1, ious_opt_multiclass_annotator1,ious_opt_head_annotator1,ious_opt_tail_annotator1, optimal_merge_iou, optimal_merge_treshold,iou_full_agreement = calculate_optimal_threshold(x_images,true_masks,soft1, soft2,pred1,saved_model_name+'_',intersections_numpy)

        name = saved_model_name + '_'
        folder_path = os.path.join(yaml_config['save_soft_mask_path'], name)

        if not os.path.exists(folder_path + '/plots'):
            os.makedirs(folder_path + '/plots')

        if isinstance(ious_opt_multiclass_annotator1, (list, tuple, set, dict)):
            plt.figure(figsize=(30, 6))
            plt.bar(range(len(ious_opt_multiclass_annotator1)), ious_opt_multiclass_annotator1, color='blue')
            plt.xticks(range(len(ious_opt_multiclass_annotator1)),
                       [str(i + 1) for i in range(len(ious_opt_multiclass_annotator1))], rotation=90)
            plt.title("IoU Scores for Multiclass")
            plt.savefig(os.path.join(folder_path + '/plots', f'multiclass.png'))

            plt.figure(figsize=(30, 6))
            plt.bar(range(len(ious_opt_head_annotator1)), ious_opt_head_annotator1, color='blue')
            plt.xticks(range(len(ious_opt_head_annotator1)),
                       [str(i + 1) for i in range(len(ious_opt_head_annotator1))], rotation=90)
            plt.title("IoU Scores for Head")
            plt.savefig(os.path.join(folder_path + '/plots', f'head.png'))

            plt.figure(figsize=(30, 6))
            plt.bar(range(len(ious_opt_tail_annotator1)), ious_opt_tail_annotator1, color='blue')
            plt.xticks(range(len(ious_opt_tail_annotator1)),
                       [str(i + 1) for i in range(len(ious_opt_tail_annotator1))], rotation=90)
            plt.title("IoU Scores for Tail")
            plt.savefig(os.path.join(folder_path + '/plots', f'tail.png'))

            df = pd.DataFrame(ious_opt_multiclass_annotator1, columns=['IoU Scores'])
            df.to_csv(os.path.join(folder_path + '/plots', f'multiclass.csv'), index=False)

            df = pd.DataFrame(ious_opt_head_annotator1, columns=['IoU Scores'])
            df.to_csv(os.path.join(folder_path + '/plots', f'head.csv'), index=False)

            df = pd.DataFrame(ious_opt_tail_annotator1, columns=['IoU Scores'])
            df.to_csv(os.path.join(folder_path + '/plots', f'tail.csv'), index=False)


        plt.figure(figsize=(30, 6))
        plt.bar(range(len(ious_opt_oneclass_annotator1)), ious_opt_oneclass_annotator1, color='blue')
        plt.xticks(range(len(ious_opt_oneclass_annotator1)),
                   [str(i + 1) for i in range(len(ious_opt_oneclass_annotator1))], rotation=90)
        plt.title("IoU Scores for Oneclass")
        plt.savefig(os.path.join(folder_path + '/plots', f'oneclass.png'))

        df = pd.DataFrame(ious_opt_oneclass_annotator1, columns=['IoU Scores'])
        df.to_csv(os.path.join(folder_path + '/plots', f'oneclass.csv'), index=False)




        test_metrics = {"inference/Mean Iou oneclass (annotator1)": mean_iou_ids1_oneclass,
                     "inference/Iou for each class oneclass (annotator1)": iou_ids1_oneclass,
                     "inference/Mean Iou oneclass (annotator2)": mean_iou_ids2_oneclass,
                     "inference/Iou for each class oneclass (annotator2)": iou_ids2_oneclass,
                     "inference/Mean Iou oneclass (intersection)": mean_iou_intersection_oneclass,
                    "inference/Iou for each class oneclass (intersection)": iou_intersection_oneclass,
                    "inference/Mean Iou oneclass (union)": mean_iou_union_oneclass,
                    "inference/Iou for each class oneclass (union)": iou_union_oneclass,
                    "inference/Mean Iou oneclass (feeling lucky)": mean_iou_feeling_oneclass,
                    "inference/Iou for each class oneclass (feeling lucky)": iou_feeling_oneclass,
                    "inference/Mean Iou multiclass (annotator1)": mean_iou_ids1_multiclass,
                    "inference/Iou for each class multiclass (annotator1)": iou_ids1_multiclass,
                    "inference/Mean Iou multiclass (annotator2)": mean_iou_ids2_multiclass,
                    "inference/Iou for each class multiclass (annotator2)": iou_ids2_multiclass,
                    "inference/Mean Iou multiclass (intersection)": mean_iou_intersection_multiclass,
                    "inference/Iou for each class multiclass (intersection)": iou_intersection_multiclass,
                    "inference/Mean Iou multiclass (union)": mean_iou_union_multiclass,
                    "inference/Iou for each class multiclass (union)": iou_union_multiclass,
                    "inference/Mean Iou multiclass (feeling lucky)": mean_iou_feeling_multiclass,
                    "inference/Iou for each class multiclass (feeling lucky)": iou_feeling_multiclass,
                    "inference/Optimal threshold multiclass": optimal_threshold_multiclass,
                    "inference/Optimal IoU multiclass": optimal_iou_multiclass,
                    "inference/Optimal threshold oneclass": optimal_threshold_oneclass,
                    "inference/Optimal IoU oneclass": optimal_iou_oneclass,
                    "inference/AP oneclass": ap_score_oneclass,
                    "inference/Optimal IoU head": optimal_iou_head,
                    "inference/Optimal IoU tail": optimal_iou_tail,
                    "inference/Optimal threshold head": optimal_head_treshold,
                    "inference/Optimal threshold tail": optimal_tail_threshold,
                    "inference/IoU multiclass best tresholds": iou_multiclass_best_tresholds,
                    "inference/IoU merge treshold": optimal_merge_treshold,
                    "inference/Optimal IoU merge": optimal_merge_iou,
                    "inference/IoU full agreement": iou_full_agreement
                       }

    wandb.log(test_metrics)
   
    

num_classes = 4

path_dict ={'laptop':'/home/nitro/Studia/Praca Dyplomowa/noisy_labels/Kod/config/config_laptop.yaml',
            'lab':'/media/cal314-1/9E044F59044F3415/Marcin/noisy_labels/Kod/config/config_lab.yaml',
            'komputer':'/media/marcin/Dysk lokalny/Programowanie/Python/Magisterka/Praca Dyplomowa/noisy_labels/Kod/config/config.yaml'
            } 

model_dict = {'myUNet': UNet(3,num_classes),
              'smpUNet': smp.Unet(in_channels = 3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels = 3, classes=num_classes,encoder_name="resnet18",encoder_weights=None),
}

mode_dict = {'normal': 'mixed',
             'intersection': 'intersection',
             'intersection_and_union': 'intersection_and_union',
             'both': 'both',
             'intersection_and_union_inference': 'intersection_and_union_inference',
             'intersection_inference': 'intersection_inference'

}


wandb.init(project="Noisy_label", entity="noisy_label",
            config={
            "model": "smpUNet++",
            "batch_size": 1,
            "annotator": 1,
            "place": 'lab',
            "mode": "intersection_and_union_inference"
            })

config = wandb.config

with open(path_dict[config.place], 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

saved_model_name = 'Annotator_1_Model_smpUNet++_Augmentation_False_Modeintersection_and_union_Optimizer_Adam_Scheduler_CosineAnnealingLR_Epochs_300_Batch_Size_6_Start_lr_0.001_Loss_CrossEntropyLoss_Timestamp_2024-05-19-14-36_best_model_iou_multiclass'
model_path = yaml_config['save_model_path'] + '/' + saved_model_name
k = 0
name = (f'Two_segment_loss alfa = {k}_Inference: Model_name: {saved_model_name}')

wandb.run.name = name

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)  
    print("GPU dostępne:", gpu_name ) 
    device = torch.device("cuda")
else:
    raise Exception("Brak dostępnej karty GPU.")

model = model_dict[config.model]
model.load_state_dict(torch.load(model_path)) 
model.to(device)
batch_maker = BatchMaker(config_path=path_dict[config.place], batch_size=config.batch_size, mode = 'test',segment = mode_dict[config.mode],annotator= config.annotator)
test_loader = batch_maker.test_loader
predict(model, test_loader)