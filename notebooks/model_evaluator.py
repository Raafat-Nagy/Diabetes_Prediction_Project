import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    # Static DataFrame to hold evaluation metrics across models
    metrics_df = pd.DataFrame(
        columns=[
            "Model",
            "Accuracy",
            "F1-Score (Macro)",
            "Precision (Macro)",
            "Recall (Macro)",
        ]
    )

    def __init__(self, model, target_labels="auto"):
        self.model = model
        self.target_labels = target_labels

    def _predict(self, X_train, X_test):
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        return y_train_pred, y_test_pred

    def _calculate_metrics(self, y_train, y_train_pred, y_test, y_test_pred):
        return {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "train_f1": f1_score(y_train, y_train_pred, average="macro"),
            "test_f1": f1_score(y_test, y_test_pred, average="macro"),
            "test_precision": precision_score(y_test, y_test_pred, average="macro"),
            "test_recall": recall_score(y_test, y_test_pred, average="macro"),
        }

    def _display_metrics(self, metrics):
        metrics_df = pd.DataFrame(
            {
                "Dataset": ["Train", "Test"],
                "Accuracy": [metrics["train_accuracy"], metrics["test_accuracy"]],
                "F1-Score (Macro)": [metrics["train_f1"], metrics["test_f1"]],
            }
        )
        print(metrics_df)

    def _plot_confusion_matrix(self, y_test, y_test_pred):
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            square=True,
            xticklabels=self.target_labels,
            yticklabels=self.target_labels,
        )
        plt.title("Confusion Matrix for Test Data")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    @staticmethod
    def update_metrics_dataframe(
        model_name, test_accuracy, test_f1, test_precision, test_recall
    ):
        new_metrics = pd.DataFrame(
            {
                "Model": [model_name],
                "Accuracy": [test_accuracy],
                "F1-Score (Macro)": [test_f1],
                "Precision (Macro)": [test_precision],
                "Recall (Macro)": [test_recall],
            }
        )
        ModelEvaluator.metrics_df = pd.concat(
            [ModelEvaluator.metrics_df, new_metrics], ignore_index=True
        )

    def evaluate(self, X_train, X_test, y_train, y_test):
        # Get predictions
        y_train_pred, y_test_pred = self._predict(X_train, X_test)

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        self._display_metrics(metrics)

        # Update the static DataFrame with test metrics
        ModelEvaluator.update_metrics_dataframe(
            self.model.__class__.__name__,
            metrics["test_accuracy"],
            metrics["test_f1"],
            metrics["test_precision"],
            metrics["test_recall"],
        )

        # Display classification report
        print("\nClassification Report for Test Data:")
        target_names = self.target_labels if self.target_labels != "auto" else None
        print(classification_report(y_test, y_test_pred, target_names=target_names))

        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_test_pred)
