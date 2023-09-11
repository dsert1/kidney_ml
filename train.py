from model import MLPClassifier
from dataloader import get_dataloader
import numpy as np
from utils import simple_accuracy
import matplotlib.pyplot as plt

# Define hyperparameters and paths
DATA_PATH = './data/kidney_data_final_9.10.23.csv'
LABEL_COLUMN = 'GDvsTI'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3

# Get dataloader
print("DATA_PATH: ", DATA_PATH)
print("LABEL_COLUMN: ", LABEL_COLUMN)
print("BATCH SIZE: ", BATCH_SIZE)
features = ['eGFR_CKDEPI2021', 'age', "SeqId_10011_65", "SeqId_10024_44", "SeqId_10040_63", "SeqId_10042_8"]
dataloader = get_dataloader(DATA_PATH, LABEL_COLUMN, batch_size=BATCH_SIZE, features=features)

# Initialize model
model = MLPClassifier(hidden_dims=[256, 128], num_epochs=EPOCHS, lr=LR, verbose=True, lambda_l1=1e-4, lambda_l2=1e-3)

# Convert dataloader outputs to numpy arrays for training
X = []
y = []
for batch_input, batch_label in dataloader:
  X.append(batch_input.numpy())
  y.append(batch_label.numpy())
X = np.vstack(X)
y = np.hstack(y)


# Train the model
model.fit(X, y)

# Save the model
# model.save('./model_weights_sample.pt')
predictions = model.predict(X)
accuracy = (predictions == y).mean()
print(f"Training accuracy: {accuracy:.2f}")

# importances = model.permutation_importance(X, y)
# print(importances)
importances = model.permutation_importance(X, y, metric=simple_accuracy, features=features)
ordered_features = sorted(importances.keys(), key=lambda x: importances[x])
ordered_importances = [importances[feat] for feat in ordered_features]

plt.barh(ordered_features, ordered_importances)
# plt.show()


# If you'd like to visualize the importances


# plt.barh([f"Feature {i}" for i in range(X.shape[1])], importances)
# plt.barh(features, importances)
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.show()


