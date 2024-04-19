import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update_confusion_matrix(self, y_true, y_pred):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.logical_and(y_true == i, y_pred == j).sum()

    def calculate_iou_per_class(self):
        iou_scores = []
        for class_id in range(self.num_classes):
            true_positive = self.confusion_matrix[class_id, class_id]
            false_positive = self.confusion_matrix[:, class_id].sum() - true_positive
            false_negative = self.confusion_matrix[class_id, :].sum() - true_positive
            union = true_positive + false_positive + false_negative
            iou = true_positive / union if union > 0 else 0
            iou_scores.append(iou)
        return iou_scores
    
    def intersection_and_union(self):
        unions = []
        intersections = []
        for class_id in range(self.num_classes):
            true_positive = self.confusion_matrix[class_id, class_id]
            false_positive = self.confusion_matrix[:, class_id].sum() - true_positive
            false_negative = self.confusion_matrix[class_id, :].sum() - true_positive
            union = true_positive + false_positive + false_negative
            unions.append(union)
            intersections.append(true_positive)
        return unions, intersections

    def mean_iou(self):
        iou_scores = self.calculate_iou_per_class()
        return np.mean(iou_scores[1:self.num_classes])
    
    def calculate_accuracy(self):
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0

    def calculate_precision_recall_f1_per_class(self):
        precision = []
        recall = []
        f1_score = []

        for class_id in range(self.num_classes):
            tp = self.confusion_matrix[class_id, class_id]
            fp = self.confusion_matrix[:, class_id].sum() - tp
            fn = self.confusion_matrix[class_id, :].sum() - tp

            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0

            precision.append(class_precision)
            recall.append(class_recall)
            f1_score.append(class_f1)

        return precision, recall, f1_score

    def mean_precision_recall_f1(self):
        precision, recall, f1 = self.calculate_precision_recall_f1_per_class()
        mean_precision = np.mean(precision[1:])
        mean_recall = np.mean(recall[1:])
        mean_f1 = np.mean(f1[1:])
        return mean_precision, mean_recall, mean_f1
    


    def calculate_dice_score_per_class(self):
        dice_scores = []
        for class_id in range(self.num_classes):
            tp = self.confusion_matrix[class_id, class_id]
            fp = self.confusion_matrix[:, class_id].sum() - tp
            fn = self.confusion_matrix[class_id, :].sum() - tp

            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            dice_scores.append(dice)
        return dice_scores

    def mean_dice_score(self):
        dice_scores = self.calculate_dice_score_per_class()
        return np.mean(dice_scores)