from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# ======================================================================================================================
# Task A
def display_metric_results_A(training_labels, training_predictions, val_labels, val_predictions, test_labels, test_predictions):
    print(f"{'Training Set Accuracy Score:':<31}{accuracy_score(training_labels, training_predictions):>10.4f}")
    print(f"{'Validation Set Accuracy Score:':<31}{accuracy_score(val_labels, val_predictions):>10.4f}")
    print(f"{'Validation Set Precision Score:':<31}{precision_score(val_labels, val_predictions):>10.4f}")
    print(f"{'Validation Set Recall Score:':<31}{recall_score(val_labels, val_predictions):>10.4f}")
    print(f"{'Validation Set F1 Score:':<31}{f1_score(val_labels, val_predictions):>10.4f}")
    print(f"{'Test Set Accuracy Score:':<31}{accuracy_score(test_labels, test_predictions):>10.4f}")
    print(f"{'Test Set Precision Score:':<31}{precision_score(test_labels, test_predictions):>10.4f}")
    print(f"{'Test Set Recall Score:':<31}{recall_score(test_labels, test_predictions):>10.4f}")
    print(f"{'Test Set F1 Score:':<31}{f1_score(test_labels, test_predictions):>10.4f}")

# ======================================================================================================================
# Task B
def display_metric_results_B(training_labels, training_predictions, val_labels, val_predictions, test_labels, test_predictions):
    print(f"{'Training Set Accuracy Score:':<31}{accuracy_score(training_labels, training_predictions):>10.4f}")
    print(f"{'Validation Set Accuracy Score:':<31}{accuracy_score(val_labels, val_predictions):>10.4f}")
    print(f"{'Validation Set Precision Score:':<31}{precision_score(val_labels, val_predictions, average='micro'):>10.4f}")
    print(f"{'Validation Set Recall Score:':<31}{recall_score(val_labels, val_predictions, average='micro'):>10.4f}")
    print(f"{'Validation Set F1 Score:':<31}{f1_score(val_labels, val_predictions, average='micro'):>10.4f}")
    print(f"{'Test Set Accuracy Score:':<31}{accuracy_score(test_labels, test_predictions):>10.4f}")
    print(f"{'Test Set Precision Score:':<31}{precision_score(test_labels, test_predictions, average='micro'):>10.4f}")
    print(f"{'Test Set Recall Score:':<31}{recall_score(test_labels, test_predictions, average='micro'):>10.4f}")
    print(f"{'Test Set F1 Score:':<31}{f1_score(test_labels, test_predictions, average='micro'):>10.4f}")

