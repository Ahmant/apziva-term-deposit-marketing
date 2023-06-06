import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import pandas as pd
import pandas as pd
import numpy as np
import src.features.build_features as build_features
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.under_sampling import RandomUnderSampler 


# Load dataset
dataset_path = './data/raw/data.csv'
dataset = pd.read_csv(dataset_path)

# Get prepared data for modeling
X_train, X_test, y_train, y_test = build_features.prepare_for_modeling(
    dataset,
    target_column='y',
    drop_columns=['default'],
    scaler=StandardScaler(),
    one_hot_encode=True,
    convert_binary_to_numeric=True,
    sampler=RandomUnderSampler(sampling_strategy = 'majority'),
    save_path='./data/processed'
)


def create_prediction_model(X_train, X_test, y_train, y_test, models, save_path = None):
    # TODO: Add "pipeline"

    results = []
    for model, params in models:
        print('🟡 STARTED: ' + model.__name__)
        cv = GridSearchCV(model(), param_grid=params, cv=5)
        cv.fit(X_train, y_train)
        
        y_pred = cv.predict(X_test)
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        best_params = cv.best_params_
        best_score = cv.best_score_
        
        print('🟢 COMPLETED: ' + model.__name__)

        results.append({
            'model': {
                'instance': cv,
                'name': model.__name__
            },
            'metrics': {
                'f1': f1,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'best_params': best_params,
                'best_score': best_score,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        })

        if save_path is not None:
            best_model = cv.best_estimator_
            joblib.dump(best_model, save_path + '/' + model.__name__ + '.pkl')
        
    return results

models = [
    (xgb.XGBClassifier, {'n_estimators': [100, 150, 200], 'subsample': [0.8, 0.9, 1]}),
    (RandomForestClassifier, {'n_estimators': np.arange(5, 10, 1)}),
]

create_prediction_model(
    X_train,
    X_test,
    y_train,
    y_test,
    models,
    save_path='./models'
)