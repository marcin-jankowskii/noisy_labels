import numpy as np

class SegmentationMetrics:
    def __init__(self, true_labels, predicted_labels, num_classes):
        self.true_labels = true_labels.flatten()
        self.predicted_labels = predicted_labels.flatten()
        self.num_classes = num_classes

    def confusion_matrix(self):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                cm[i, j] = np.sum((self.true_labels == i) & (self.predicted_labels == j))
        return cm

    def tp_tn_fp_fn_per_class(self):
        cm = self.confusion_matrix()
        metrics = np.zeros((self.num_classes, 4))  # 4 columns for TP, TN, FP, FN
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            metrics[i] = [tp, tn, fp, fn]
        return metrics

    def accuracy(self):
        metrics = self.tp_tn_fp_fn_per_class()
        class_accuracies = (metrics[:, 0] + metrics[:, 1]) / np.sum(metrics, axis=1)
        return np.mean(class_accuracies)

    def precision(self):
        metrics = self.tp_tn_fp_fn_per_class()
        class_precision = metrics[:, 0] / (metrics[:, 0] + metrics[:, 2])
        return np.nanmean(np.where(class_precision >= 0, class_precision, np.nan))

    def recall(self):
        metrics = self.tp_tn_fp_fn_per_class()
        class_recall = metrics[:, 0] / (metrics[:, 0] + metrics[:, 3])
        return np.nanmean(np.where(class_recall >= 0, class_recall, np.nan))

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall + 1e-7)

    def iou(self):
        metrics = self.tp_tn_fp_fn_per_class()
        class_iou = metrics[:, 0] / (metrics[:, 0] + metrics[:, 2] + metrics[:, 3])
        return np.nanmean(np.where(class_iou >= 0, class_iou, np.nan))

    def dice(self):
        metrics = self.tp_tn_fp_fn_per_class()
        class_dice = 2 * metrics[:, 0] / (2 * metrics[:, 0] + metrics[:, 2] + metrics[:, 3])
        return np.nanmean(np.where(class_dice >= 0, class_dice, np.nan))

# Usage:
# metrics = SegmentationMetrics(true_labels, predicted_labels, num_classes)
# print(metrics.tp_tn_fp_fn_per_class())
# print(metrics.accuracy())
# print(metrics.precision())
# ... and so on ...
