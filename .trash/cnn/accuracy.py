import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Training Loop with Accuracy Tracking ---------------- #
train_accuracies = []
test_accuracies = []

num_epochs = 10  # your number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # ---------------- Evaluate Training Accuracy ---------------- #
    model.eval()
    y_true_train, y_pred_train = [], []
    with torch.no_grad():
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(targets.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_accuracies.append(train_acc)

    # ---------------- Evaluate Test Accuracy ---------------- #
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(targets.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_accuracies.append(test_acc)

    print(f"\nEpoch {epoch+1}/{num_epochs} - "
          f"Loss: {running_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# ---------------- Plot Training vs Test Accuracy ---------------- #
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), train_accuracies, label="Train Accuracy", marker='o')
plt.plot(range(1, num_epochs+1), test_accuracies, label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
