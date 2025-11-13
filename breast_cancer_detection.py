import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Preprocessing
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build model
model = nn.Sequential(
    nn.Linear(30, 30),
    nn.ReLU(),
    nn.Linear(30, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Print model summary
print(model)

# Prepare data for PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

# Train model
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(100):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_losses.append(epoch_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor).item()
        val_losses.append(val_loss)
        val_predicted = (val_outputs > 0.5).float()
        val_correct = (val_predicted == y_test_tensor).sum().item()
        val_accuracies.append(val_correct / y_test_tensor.size(0))

history = {'loss': train_losses, 'val_loss': val_losses, 'accuracy': train_accuracies, 'val_accuracy': val_accuracies}

# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor).item()
    test_acc = ((test_outputs > 0.5).float() == y_test_tensor).float().mean().item()
    print(f'Test Accuracy: {test_acc}')

    y_pred = (test_outputs > 0.5).float().squeeze().numpy().astype(int)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

# Test prediction with a sample
sample_features = X_test[0]  # First test sample
sample_label = y_test.iloc[0]
with torch.no_grad():
    sample_pred = model(torch.tensor(sample_features, dtype=torch.float32).unsqueeze(0))
    sample_pred_class = int((sample_pred > 0.5).float().item())
print(f'Sample Prediction: Actual={sample_label}, Predicted={sample_pred_class} ({ "Malignant" if sample_pred_class == 1 else "Benign" })')

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Save model
torch.save(model.state_dict(), 'breast_cancer_model.h5')

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Prediction function
def predict_tumor():
    features = []
    for name in data.feature_names:
        val = float(input(f'Enter {name}: '))
        features.append(val)
    features_df = pd.DataFrame([features], columns=data.feature_names)
    features_scaled = scaler.transform(features_df)
    with torch.no_grad():
        pred_tensor = model(torch.tensor(features_scaled, dtype=torch.float32))
        pred_class = int((pred_tensor > 0.5).float().item())
    if pred_class == 1:
        print('Malignant')
    else:
        print('Benign')

# Run prediction (uncomment to use)
# predict_tumor()
