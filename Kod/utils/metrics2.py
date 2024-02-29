import numpy as np


def calculate_iou(labels,preds):

    num_classes = preds.shape[1]
    ious = []

    for cls in range(num_classes):

        cls_preds = preds[:,cls,:,:]
        cls_labels = labels[:,cls,:,:]

        intersection = np.logical_and(cls_preds, cls_labels).sum()
        union = np.logical_or(cls_preds, cls_labels).sum()
        union = union + 1e-6
        iou = intersection / union
        mean_iou = iou.mean().item()
        ious.append(mean_iou)

    average_iou = (ious[1]+ious[2])/2

    return average_iou, ious