import numpy as np

class SegmentationMetrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels.flatten()
        self.predicted_labels = predicted_labels.flatten()

    def confusion_matrix(self):
        tp = np.sum((self.true_labels == 1) & (self.predicted_labels == 1))
        tn = np.sum((self.true_labels == 0) & (self.predicted_labels == 0))
        fp = np.sum((self.true_labels == 0) & (self.predicted_labels == 1))
        fn = np.sum((self.true_labels == 1) & (self.predicted_labels == 0))
        return tp, tn, fp, fn

    def accuracy(self):
        tp, tn, fp, fn = self.confusion_matrix()
        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self):
        tp, _, fp, _ = self.confusion_matrix()
        return tp / (tp + fp)

    def recall(self):
        tp, _, _, fn = self.confusion_matrix()
        return tp / (tp + fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall)

    def iou(self):
        tp, _, fp, fn = self.confusion_matrix()
        return tp / (tp + fp + fn)
    
    def dice(self):
        tp, _, fp, fn = self.confusion_matrix()
        return 2*tp / (2*tp + fp + fn)
    
    
    def average_precision(self):
        # Initialize variables
        tp, _, fp, fn = self.confusion_matrix()
        precision = self.precision()
        recall = self.recall()

        # Sort predictions by confidence
        sorted_indices = np.argsort(self.predicted_labels)
        sorted_true_labels = self.true_labels[sorted_indices]

        # Initialize variables
        precision_list = [precision]
        recall_list = [recall]

        # Calculate precision and recall for each threshold
        for i in range(len(sorted_true_labels) - 1, -1, -1):
            if sorted_true_labels[i] == 1:
                tp -= 1
            else:
                fp -= 1
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)

        # Calculate area under the precision-recall curve
        ap = 0
        for i in range(1, len(precision_list)):
            if recall_list[i] != recall_list[i-1]:
                ap += precision_list[i] * (recall_list[i] - recall_list[i-1])
        return ap

# Usage:
# metrics = SegmentationMetrics(true_labels, predicted_labels)
# print(metrics.accuracy())
# print(metrics.precision())
# print(metrics.recall())
# print(metrics.f1())
# print(metrics.iou())