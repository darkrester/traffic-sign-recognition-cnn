import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from dataset import get_dataloaders
from model import TrafficSignCNN
from config import *


def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.set_num_threads(8)

    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        train_csv="../data/GTSRB/Train.csv",
        train_root="../data/GTSRB",
        test_csv="../data/GTSRB/Test.csv",
        test_root="../data/GTSRB"
    )

    model = TrafficSignCNN(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_accuracy = 0.0

    # История обучения
    train_losses = []

    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        progress_bar = tqdm(train_loader)

        for images, labels in progress_bar:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}"
            )

        # Средний loss за эпоху
        epoch_loss = running_loss / len(train_loader)

        train_losses.append(epoch_loss)

        # Метрики
        metrics = evaluate(
            model,
            test_loader,
            device
        )


        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1_scores.append(f1)

        print(
            f"\nEpoch: {epoch + 1}"
            f"\nLoss: {epoch_loss:.4f}"
            f"\nAccuracy: {accuracy:.2f}%"
            f"\nPrecision: {precision:.4f}"
            f"\nRecall: {recall:.4f}"
            f"\nF1-score: {f1:.4f}"
        )

        scheduler.step(accuracy)

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print("Best model saved")

    # Финальная матрица ошибок
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Построение графиков
    plot_training_curves(
        train_losses,
        test_accuracies,
        test_precisions,
        test_recalls,
        test_f1_scores
    )


def evaluate(model, loader, device):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    ) * 100

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=[
            f"class_{i}" for i in range(NUM_CLASSES)
        ],
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }


def plot_training_curves(
    losses,
    accuracies,
    precisions,
    recalls,
    f1_scores
):

    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 8))

    # Loss
    plt.subplot(2, 2, 1)

    plt.plot(epochs, losses)

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy
    plt.subplot(2, 2, 2)

    plt.plot(epochs, accuracies)

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    # Precision / Recall
    plt.subplot(2, 2, 3)

    plt.plot(epochs, precisions, label="Precision")
    plt.plot(epochs, recalls, label="Recall")

    plt.title("Precision / Recall")
    plt.xlabel("Epoch")

    plt.legend()

    # F1
    plt.subplot(2, 2, 4)

    plt.plot(epochs, f1_scores)

    plt.title("F1-score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")

    plt.tight_layout()

    os.makedirs("../plots", exist_ok=True)

    plt.savefig("../plots/training_curves.png")

    plt.show()


if __name__ == "__main__":

    train()