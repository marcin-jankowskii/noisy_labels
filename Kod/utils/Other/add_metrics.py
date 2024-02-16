import numpy as np
from sklearn.metrics import average_precision_score


def IoU(y_true, y_pred_bin):
    intersection = np.count_nonzero(y_true * y_pred_bin)
    if (np.count_nonzero(y_true) + np.count_nonzero(y_pred_bin) - intersection) == 0:
        return 0
    else:
        return intersection / (np.count_nonzero(y_true) + np.count_nonzero(y_pred_bin) - intersection)

def Opt_Th(list_gt, list_softmasks):
    Max_iou = 0
    Th_max_iou = 0

    for th in range(0,255):
        th = th/255
        list_iou = []
        for i in range(0,len(list_gt)):
            gt = list_gt[i]
            pr = list_softmasks[i]
            th_pr = np.zeros_like(list_softmasks[i])
            th_pr[pr >= th] = 1
            list_iou.append(IoU(gt,th_pr))
        if np.mean(list_iou) > Max_iou:
            Max_iou = np.mean(list_iou)
            Th_max_iou = th

    return Th_max_iou, Max_iou

def ThIoU(gt, softmask, th):
    pr = np.zeros_like(softmask)
    pr[softmask >= th] = 1
    return IoU(gt, pr)

def metrics(list_gt, list_softmasks):
    list_iou_0_25 = []
    list_iou_0_4 = []
    list_iou_0_6 = []
    list_iou_0_75 = []

    for i in range(0, len(list_gt)):
        gt = list_gt[i]
        softmask = list_softmasks[i]

        list_iou_0_25.append(ThIoU(gt, softmask, th=0.25))
        list_iou_0_4.append(ThIoU(gt, softmask, th=0.4))
        list_iou_0_6.append(ThIoU(gt, softmask, th=0.6))
        list_iou_0_75.append(ThIoU(gt, softmask, th=0.75))


        if i==0:
            y_true = list_gt[i]
            y_scores = list_softmasks[i]
        else:
            y_true = np.vstack([y_true, list_gt[i]])
            y_scores = np.vstack([y_scores, list_softmasks[i]])


    AP_score = average_precision_score((y_true).flat, (y_scores).flat)
    opt_th, _ = Opt_Th(list_gt, list_softmasks)
    Iou_0_25 = np.mean(list_iou_0_25)
    Iou_0_4 = np.mean(list_iou_0_4)
    Iou_0_6 = np.mean(list_iou_0_6)
    Iou_0_75 = np.mean(list_iou_0_75)

    return AP_score, Iou_0_25, Iou_0_4, Iou_0_6, Iou_0_75, opt_th