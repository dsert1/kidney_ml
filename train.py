# Imports
from model import MLPClassifier
from dataloader import get_dataloader
import numpy as np
from utils import simple_accuracy
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import matthews_corrcoef
from data import dataset
import itertools
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve


# Constants
DATA_PATH = dataset
LABEL_COLUMN = 'GDvsTI'
EPOCHS = 100
N_BOOTSTRAPS = 100
features = ['eGFR_CKDEPI2021', 'age', "SeqId_10011_65", "SeqId_10024_44", "SeqId_10040_63", "SeqId_10042_8"]
param_grid = {
    'lr': [1e-2, 1e-3, 1e-4],
    'batch_size': [8, 16, 32],
    'hidden_dims': [(32,), (64, 32), (128, 64, 32)]
}


# Hyperparameter grid
param_grid = {
    'lr': [1e-2, 1e-3, 1e-4],
    'batch_size': [8, 16, 32],
    # Add other hyperparameters here...
}

# Load the data
def load_data():
    print(f"Loading data from: {DATA_PATH}")
    dataloader = get_dataloader(DATA_PATH, LABEL_COLUMN, batch_size=32, features=features)

    X, y = [], []
    for batch_input, batch_label in dataloader:
        X.append(batch_input.numpy())
        y.append(batch_label.numpy())
    
    return np.vstack(X), np.hstack(y)

def grid_search_train(X, y):
    best_mcc = -np.inf
    best_params = None
    
    # Lists to store metrics for each bootstrap iteration
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    mccs = []
    rocs = []
    pr_curves = []

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for i in tqdm(range(N_BOOTSTRAPS)):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

        for params in param_combinations:
            model = MLPClassifier(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            # Compute metrics
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions)
            recall = recall_score(y_val, predictions)
            f1 = f1_score(y_val, predictions)
            mcc = matthews_corrcoef(y_val, predictions)
            fpr, tpr, _ = roc_curve(y_val, predictions)
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, predictions)
            
            # Store metrics
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            mccs.append(mcc)
            rocs.append((fpr, tpr))
            pr_curves.append((precision_curve, recall_curve))
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_params = params
                
    # Compute mean and standard deviation for each metric
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    mean_mcc = np.mean(mccs)
    std_mcc = np.std(mccs)
    
    return best_mcc, best_params, mean_accuracy, std_accuracy, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, mean_mcc, std_mcc

# Train the model with grid search
# def grid_search_train(X, y):
#     best_mcc = -np.inf
#     best_params = None
    
#     # Generate all combinations of hyperparameters
#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
#     for i in tqdm(range(N_BOOTSTRAPS)):
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

#         for params in param_combinations:
#             model = MLPClassifier(**params)
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_val)
#             mcc = matthews_corrcoef(y_val, predictions)
            
#             if mcc > best_mcc:
#                 best_mcc = mcc
#                 best_params = params
                
#     return best_mcc, best_params

# Plot the importances
def plot_importances(model, X, y):
    importances = model.permutation_importance(X, y, metric=simple_accuracy, features=features)
    ordered_features = sorted(importances.keys(), key=lambda x: importances[x])
    ordered_importances = [importances[feat] for feat in ordered_features]

    plt.barh(ordered_features, ordered_importances)
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.show()

# def grid_search_train(X, y):
#     # Define the model and the grid search
#     model = MLPClassifier()
#     grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    
#     # Run the grid search
#     grid_search.fit(X, y)

#     # Return best model and its parameters
#     return grid_search.best_score_, grid_search.best_params_


def main():
    X, y = load_data()
    best_score, best_params, mean_accuracy, std_accuracy, mean_precision, std_precision, mean_recall, std_recall, mean_f1, std_f1, mean_mcc, std_mcc = grid_search_train(X, y)
    
    print(f"Best Score (MCC): {best_score}")
    print(f"Best Parameters: {best_params}")
    print(f"Mean Accuracy: {mean_accuracy} ± {std_accuracy}")
    print(f"Mean Precision: {mean_precision} ± {std_precision}")
    print(f"Mean Recall: {mean_recall} ± {std_recall}")
    print(f"Mean F1 Score: {mean_f1} ± {std_f1}")
    print(f"Mean MCC: {mean_mcc} ± {std_mcc}")
    
    # Train final model with best parameters on the entire dataset
    model = MLPClassifier(**best_params)
    model.fit(X, y)
    
    # Optional: Save the model
    # model.save('./model_weights_sample.pt')
    
    # Plot feature importances
    plot_importances(model, X, y)



if __name__ == '__main__':
    main()
