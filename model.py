# Imports
import torch
import torch.nn as nn
import numpy as np
import tqdm
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from utils import simple_accuracy
from data import dataset

# Constants
__all__ = ['MLPClassifier']

# Functions
def l1_regularizer(model, lambda_l1=0.01):
    ''' LASSO '''
    lossl1 = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith('weight'):
            lossl1 += lambda_l1 * model_param_value.abs().sum()
    return lossl1

# Classes
class HelperDataset(torch.utils.data.Dataset):
    ''' Helper Dataset for PyTorch DataLoader '''
    def __init__(self, x, y=None):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MLP(nn.Module):
    ''' Multi-Layer Perceptron Implementation '''
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLP, self).__init__()

        hidden_layers = []
        hidden_dims = [n for n in hidden_dims]
        hidden_dims.insert(0, in_dim)
        for _in, _out in zip(hidden_dims, hidden_dims[1:]):
            hidden_layers.append(nn.Linear(_in, _out))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(0.5))

        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], out_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x

class MLPClassifier(BaseEstimator):
    ''' ... '''
    def __init__(
        self,
        hidden_dims = (32,),
        num_epochs = 32,
        batch_size = 8,
        lr = 1e-2,
        lambda_l1 = 1e-3,     # weight for L1 regularization
        lambda_l2 = 1e-2,     # weight for L2 regularization
        device = 'cpu',
        verbose = False,
    ):  
        ''' ... '''
        # training parameters
        params = locals()
        params.pop('self')
        for k, v in params.items():
            setattr(self, k, v)

    def fit(self, X, y):
        ''' ... '''
        # input validation
        X, y = self._validate_inputs(X, y)

        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # initialize model
        self.net_ = MLP(X.shape[1], self.hidden_dims, 1)
        self.net_.to(self.device)

        # initialize dataset and dataloader
        ldr = torch.utils.data.DataLoader(
            HelperDataset(X, y),
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = True,
        )

        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.net_.parameters(),
            lr = self.lr,
            weight_decay = self.lambda_l2
        )
        
        # initialize loss function (binary cross entropy)
        loss_fn = torch.nn.BCELoss()

        # set model to train mode
        torch.set_grad_enabled(True)
        self.net_.train()

        # training loop
        iter_ = tqdm.tqdm(range(self.num_epochs)) if self.verbose else range(self.num_epochs)
        for e in iter_:        
            for X, y in ldr:
                # convert inputs to proper dtype and device
                X = X.to(torch.float).to(self.device)
                y = y.to(torch.float).to(self.device)

                # feed inputs to model, forward
                outputs = self.net_(X)

                # calculate train loss and add 
                loss = loss_fn(outputs, y)
                if self.lambda_l1 > 0:
                    loss += l1_regularizer(self.net_, lambda_l1=self.lambda_l1)            
            
                # backward and update parameters
                optimizer.zero_grad() 
                loss.backward()        
                optimizer.step()

        return self

    def predict(self, X):
        ''' ... '''
        proba = self.predict_proba(X)
        return (proba[:, 1] > proba[:, 0]).astype(np.int32)
    
    def score(self, X, y):
      y_pred = self.predict(X)
      return simple_accuracy(y, y_pred)

        
    def predict_proba(self, X):
        ''' ... '''
        # input validation
        check_is_fitted(self)
        self._validate_inputs(X)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # convert to PyTorch tensor
        X = torch.tensor(X, dtype = torch.float, device = self.device)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # collect model outputs/probabilities 
        # and make them consistence with the outputs of sklearn models
        proba_pos = self.net_(X).detach().cpu().numpy()
        proba_neg = 1 - proba_pos
        return np.array(list(zip(proba_neg, proba_pos)))

    def load(self, filepath):
        ''' ... '''
        state_dict = torch.load(filepath, map_location='cpu')
        num_features = len(state_dict['module.0.running_mean'])
        self.net_ = MLP(num_features, self.hidden_dims, 1)
        self.net_.load_state_dict(state_dict)
        self.net_.to(self.device)

    def save(self, filepath):
        ''' ... '''
        check_is_fitted(self)
        torch.save(self.net_.state_dict(), filepath)

    def estimate_shap(self, X):
        ''' ... '''
        check_is_fitted(self)
        self._validate_inputs(X)
        raise NotImplementedError

    def _validate_inputs(self, X, y=None):
        ''' ... '''
        if y is None:
            check_array(X)

        else:
            X, y = check_X_y(X, y)

            # y shall always be 0 or 1
            for label in unique_labels(y):
                assert label in (0, 1), 'Label must be 0 or 1. {} is detected.'.format(label)

        return X, y
    
    def permutation_importance(self, X, y, metric=simple_accuracy, features=None):
      # If features are not provided, default to generic names
      if features is None:
          features = [f"Feature {i}" for i in range(X.shape[1])]
      
      # Validate that the length of features matches X's shape
      if len(features) != X.shape[1]:
          raise ValueError("Length of features does not match number of columns in X.")

      # Evaluate the original metric
      original_preds = self.predict(X)
      original_score = metric(y, original_preds)

      importances_dict = {}

      for i in range(X.shape[1]):
          feature_name = features[i]
          X_shuffled = X.copy()
          np.random.shuffle(X_shuffled[:, i])
          shuffled_preds = self.predict(X_shuffled)
          shuffled_score = metric(y, shuffled_preds)
          importances_dict[feature_name] = original_score - shuffled_score

      return importances_dict


# Grid Search Example with LDA
param_grid = {
    'solver': ['svd', 'lsqr', 'eigen']
}
grid_search = GridSearchCV(LDA(), param_grid, scoring='accuracy', cv=5)

# Main Function
if __name__ == '__main__':
    ''' for testing purpose only '''
    # cls = MLPClassifier(hidden_dims=[32], num_epochs=64, verbose=True)
    # X = np.random.randn(128, 8)
    # y = np.random.randint(0, 3, (128,))

    # cls.fit(X, y)
    # print([name for name, _ in cls.net_.named_parameters()])
    # print(cls.predict(X))

    # print(cls.net_.state_dict())
    # cls.save('./tmp.pt')

    # cls_tmp = MLPClassifier(hidden_dims=[32], verbose=True)
    # cls_tmp.load('./tmp.pt')
    # print(cls_tmp.predict(X)) 



    # cls = MLPClassifier(hidden_dims=[32], num_epochs=64, verbose=True)
    # X = np.random.randn(128, 8)
    # y = np.random.randint(0, 2, (128,))  # Ensure labels are 0 and 1

    # cls.fit(X, y)

    # importances = cls.permutation_importance(X, y)
    # print(importances)


    # import matplotlib.pyplot as plt

    # plt.barh([f"Feature {i}" for i in range(X.shape[1])], importances)
    # plt.xlabel('Permutation Importance')
    # plt.ylabel('Features')
    # plt.show()



    # v2

    df = pd.read_csv(dataset)
    label_column = "GDvsTI"
    print("df: ", df)
    X = df.drop(label_column, axis=1).values
    y = np.array([1 if i == 2 else 0 for i in df[label_column].values]) # 1 if Disease state 2 and 0 for Disease state 1
    # print(X)
    # print(y)
    clf = MLPClassifier(hidden_dims=[32], num_epochs=64, verbose=True)
    clf.fit(X, y)

    # clf = MLPClassifier(hidden_dims=[32])
    # clf.load('./model_weights.pt')

    predictions = clf.predict(X)
    accuracy = (predictions == y).mean()
    print(f"Training accuracy: {accuracy:.2f}")

    importances = clf.permutation_importance(X, y)
    print(importances)

    # If you'd like to visualize the importances
    import matplotlib.pyplot as plt

    plt.barh([f"Feature {i}" for i in range(X.shape[1])], importances)
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.show()

