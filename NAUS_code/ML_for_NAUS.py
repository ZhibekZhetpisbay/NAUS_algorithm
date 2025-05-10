#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from importlib.resources import path

#Function for dividing data in a given class ratio
#Input data: data - dataset, number - number 0f elements of each class in test data, i - random state
#Output data: Train feature values, test feature values, train target values, test target values
def split_test(data,number,i):
    class_0 = data[data.iloc[:,-1] == 0]
    class_1 = data[data.iloc[:,-1] == 1]
    test_size_per_class = number
    if class_0.empty or class_1.empty:
        raise ValueError("One of the classes is empty. Please ensure the dataset contains both classes.")
    class_0_test = class_0.sample(test_size_per_class, random_state=i)
    class_1_test = class_1.sample(test_size_per_class, random_state=i)
    test_set = pd.concat([class_0_test, class_1_test])
    train_set = data.drop(test_set.index)
    X_train = train_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1]
    X_test = test_set.iloc[:,:-1]
    y_test = test_set.iloc[:,-1]
    print("Training set:")
    print(y_train.value_counts())
    print("\nTest set:")
    print(y_test.value_counts())
    return X_train,X_test,y_train,y_test

#Function for training Random Forest algorithm data in pre- and post-balancing modes using the undersampling method
#Input data: data - dataset, test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split, mode - training mode 'original' for before undersampling and 'undersampled' for after, n_splits - number of splits for StratifiedKFold  
def rndf_weights(data, test_min, rnd1, rnd2,mode, n_splits):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(data, test_min, rnd1)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    if mode == 'original':
        class_weights='balanced_subsample'
    else:
        class_weights=None
    for train_index, val_index in kf.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=7,
            criterion='gini',
            min_samples_leaf=10,
            min_samples_split=18,
            n_jobs=-1,
            random_state=42,
            class_weight=class_weights
        )
        model.fit(X_train, y_train)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_pred_binary = (val_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val, val_pred_binary)
        auc_score = roc_auc_score(y_val, val_pred_proba)
        precision = precision_score(y_val, val_pred_binary)
        recall = recall_score(y_val, val_pred_binary)
        
        evals.append({
            'fold': fold,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall
        })
        
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    
    test_pred_proba = model.predict_proba(X_test1)[:, 1]
    test_pred_binary = (test_pred_proba > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test1, test_pred_binary)
    test_auc_score = roc_auc_score(y_test1, test_pred_proba)
    test_precision = precision_score(y_test1, test_pred_binary)
    test_recall = recall_score(y_test1, test_pred_binary)
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    return test_accuracy,test_auc_score,test_precision,test_recall

#Function for training Light GBM algorithm data in pre- and post-balancing modes using the undersampling method
#Input data: data - dataset, test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split, mode - training mode 'original' for before undersampling and 'undersampled' for after, n_splits - number of splits for StratifiedKFold  
def lgbm_weights(data,test_min, rnd1=42,rnd2=42,mode = 'original', n_splits=2):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(data, test_min, rnd1)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    if mode == 'original':
        params = {
            'objective': 'binary',       
            'metric': 'binary_logloss',  
            'verbosity': -1,
            'is_unbalance': True,
            'class_label': 'is_unbalance'}
    else :
        params = {
            'objective': 'binary',       
            'metric': 'binary_logloss',  
            'verbosity': -1}
    for train_index, val_index in kf.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
        )
        val_pred = model.predict(X_val)
        val_pred_binary = (val_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, val_pred_binary)
        auc_score = roc_auc_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred_binary)
        recall = recall_score(y_val, val_pred_binary)
        evals.append({
            'fold': fold,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall
        })
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    test_pred = model.predict(X_test1)
    test_pred_binary = (test_pred > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test1, test_pred_binary)
    test_auc_score = roc_auc_score(y_test1, test_pred)
    test_precision = precision_score(y_test1, test_pred_binary)
    test_recall = recall_score(y_test1, test_pred_binary)
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")


#Function for training MLP algorithm data in pre- and post-balancing modes using the undersampling method
#Input data: data - dataset, test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split, epochs - number of epochs for model learning, batch_size - samples per training step, lr - learning rate 
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def mlp_weight(data, test_min, rnd1=42, rnd2=42, epochs=50, batch_size=32, lr=0.001):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(data, test_min, rnd1)
    
    
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_full.shape[1]
    
    for train_index, val_index in kf.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full.iloc[train_index].values, X_train_full.iloc[val_index].values
        y_train, y_val = y_train_full.iloc[train_index].values, y_train_full.iloc[val_index].values
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        # Определяем MLP-модель внутри функции
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).cpu().numpy().flatten()
            val_preds_binary = (val_preds > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val, val_preds_binary)
        auc_score = roc_auc_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds_binary)
        recall = recall_score(y_val, val_preds_binary)
        evals.append({'fold': fold, 'accuracy': accuracy, 'auc': auc_score, 'precision': precision, 'recall': recall})
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    
    X_test1_tensor = torch.tensor(X_test1.values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test1_tensor).cpu().numpy().flatten()
        test_preds_binary = (test_preds > 0.5).astype(int)
    
    test_accuracy = accuracy_score(y_test1, test_preds_binary)
    test_auc_score = roc_auc_score(y_test1, test_preds)
    test_precision = precision_score(y_test1, test_preds_binary)
    test_recall = recall_score(y_test1, test_preds_binary)
    
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    return test_accuracy, test_auc_score, test_precision, test_recall

#Function for training Light GBM algorithm data after oversampling method
#Input data: overs_data - oversampled dataset, n - number of new samples,test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split
def lgbm_overs1(overs_data,n,test_min,rnd1=42,rnd2=42):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(overs_data.iloc[:-n,:], test_min, rnd1)
    X_train_full1 = pd.concat([X_train_full, overs_data.iloc[-n:, :-1]], ignore_index=True)
    y_train_full1 = pd.concat([y_train_full, overs_data.iloc[-n:, -1]], ignore_index=True)
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    params = {
        'objective': 'binary',       
        'metric': 'binary_logloss',  
        'verbosity': -1
    }
    for train_index, val_index in kf.split(X_train_full1, y_train_full1):
        X_train, X_val = X_train_full1.iloc[train_index], X_train_full1.iloc[val_index]
        y_train, y_val = y_train_full1.iloc[train_index], y_train_full1.iloc[val_index]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
        )
        val_pred = model.predict(X_val)
        val_pred_binary = (val_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, val_pred_binary)
        auc_score = roc_auc_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred_binary)
        recall = recall_score(y_val, val_pred_binary)
        evals.append({
            'fold': fold,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall
        })
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    test_pred = model.predict(X_test1)
    test_pred_binary = (test_pred > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test1, test_pred_binary)
    test_auc_score = roc_auc_score(y_test1, test_pred)
    test_precision = precision_score(y_test1, test_pred_binary)
    test_recall = recall_score(y_test1, test_pred_binary)
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    return test_accuracy,test_auc_score,test_precision,test_recall

#Function for training Random Forest algorithm data after oversampling method
#Input data: overs_data - oversampled dataset, n - number of new samples,test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split
def rndf_overs1(overs_data,n,test_min,rnd1=42,rnd2=42):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(overs_data.iloc[:-n,:], test_min, rnd1)
    X_train_full1 = pd.concat([X_train_full, overs_data.iloc[-n:, :-1]], ignore_index=True)
    y_train_full1 = pd.concat([y_train_full, overs_data.iloc[-n:, -1]], ignore_index=True)
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    for train_index, val_index in kf.split(X_train_full1, y_train_full1):
        X_train, X_val = X_train_full1.iloc[train_index], X_train_full1.iloc[val_index]
        y_train, y_val = y_train_full1.iloc[train_index], y_train_full1.iloc[val_index]        
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=7,
            criterion='gini',
            min_samples_leaf=10,
            min_samples_split=18,
            n_jobs=-1,
            random_state=42)
        model.fit(X_train, y_train)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_pred_binary = (val_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val, val_pred_binary)
        auc_score = roc_auc_score(y_val, val_pred_proba)
        precision = precision_score(y_val, val_pred_binary)
        recall = recall_score(y_val, val_pred_binary)
        
        evals.append({
            'fold': fold,
            'accuracy': accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall
        })
        
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    
    test_pred_proba = model.predict_proba(X_test1)[:, 1]
    test_pred_binary = (test_pred_proba > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test1, test_pred_binary)
    test_auc_score = roc_auc_score(y_test1, test_pred_proba)
    test_precision = precision_score(y_test1, test_pred_binary)
    test_recall = recall_score(y_test1, test_pred_binary)
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    return test_accuracy,test_auc_score,test_precision,test_recall

#Function for training MLP algorithm data after oversampling method
#Input data: overs_data - oversampled dataset, n - number of new samples,test_min - number of elements of each class in test data, rnd1 - random state for test split, rnd2 - random state for validation split, epochs - number of epochs for model learning, batch_size - samples per training step, lr - learning rate 
def mlp_overs1(overs_data, n, test_min, rnd1=42, rnd2=42, epochs=50, batch_size=32, lr=0.001):
    X_train_full, X_test1, y_train_full, y_test1 = split_test(overs_data.iloc[:-n, :], test_min, rnd1)
    X_train_full1 = pd.concat([X_train_full, overs_data.iloc[-n:, :-1]], ignore_index=True)
    y_train_full1 = pd.concat([y_train_full, overs_data.iloc[-n:, -1]], ignore_index=True)
    
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rnd2)
    fold = 1
    evals = []
    set_random_seed(42)
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_full1.shape[1]
    
    for train_index, val_index in kf.split(X_train_full1, y_train_full1):
        X_train, X_val = X_train_full1.iloc[train_index].values, X_train_full1.iloc[val_index].values
        y_train, y_val = y_train_full1.iloc[train_index].values, y_train_full1.iloc[val_index].values
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = MLP(input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor).cpu().numpy().flatten()
            val_preds_binary = (val_preds > 0.5).astype(int)
        
        accuracy = accuracy_score(y_val, val_preds_binary)
        auc_score = roc_auc_score(y_val, val_preds)
        precision = precision_score(y_val, val_preds_binary)
        recall = recall_score(y_val, val_preds_binary)
        evals.append({'fold': fold, 'accuracy': accuracy, 'auc': auc_score, 'precision': precision, 'recall': recall})
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold += 1
    
    X_test1_tensor = torch.tensor(X_test1.values, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test1_tensor).cpu().numpy().flatten()
        test_preds_binary = (test_preds > 0.5).astype(int)
    
    test_accuracy = accuracy_score(y_test1, test_preds_binary)
    test_auc_score = roc_auc_score(y_test1, test_preds)
    test_precision = precision_score(y_test1, test_preds_binary)
    test_recall = recall_score(y_test1, test_preds_binary)
    
    print(f"\nTest Data Evaluation - Accuracy: {test_accuracy:.4f}, AUC: {test_auc_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    return test_accuracy, test_auc_score, test_precision, test_recall

