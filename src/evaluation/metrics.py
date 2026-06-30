from sklearn.metrics import classification_report, confusion_matrix

def get_classification_report(y_true, y_pred, target_names=None):
    """
    Generate classification report.
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

def get_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
