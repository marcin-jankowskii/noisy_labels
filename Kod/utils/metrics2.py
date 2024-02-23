import numpy as np

def simple_iou(y_true, y_pred):
     # Flatten the arrays
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Calculate intersection and union
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    # Calculate IoU
    iou = intersection / union

    return iou

def iou_per_class(y_true, y_pred, num_classes):
    iou_scores = []
    for c in range(num_classes):
        iou = simple_iou(y_true[..., c], y_pred[..., c])
        iou_scores.append(iou)
    return iou_scores

def mean_iou(y_true, y_pred, num_classes):
    return np.mean(iou_per_class(y_true, y_pred, num_classes)[1:3])
