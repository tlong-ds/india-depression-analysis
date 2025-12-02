import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    NUMERIC_FEATURES_COMMON, CATEGORICAL_FEATURES_COMMON, ORDINAL_FEATURES,
    STUDENT_SPECIFIC_NUMERIC, WORKER_SPECIFIC_NUMERIC, WORKER_SPECIFIC_CATEGORICAL,
    SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER
)
from src.transformers import GarbageCleaner

def create_student_pipeline():
    """Creates the pipeline for student data."""
    numeric_features = NUMERIC_FEATURES_COMMON + STUDENT_SPECIFIC_NUMERIC
    categorical_features = CATEGORICAL_FEATURES_COMMON
    ordinal_features = ORDINAL_FEATURES

    # 1. Cleaning Step (Custom Transformer)
    cleaner = GarbageCleaner(columns=numeric_features)

    # 2. Preprocessing Steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER]))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    # 3. Full Pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', cleaner),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline

def create_worker_pipeline():
    """Creates the pipeline for working professional data."""
    numeric_features = NUMERIC_FEATURES_COMMON + WORKER_SPECIFIC_NUMERIC
    categorical_features = CATEGORICAL_FEATURES_COMMON + WORKER_SPECIFIC_CATEGORICAL
    ordinal_features = ORDINAL_FEATURES

    # 1. Cleaning Step
    cleaner = GarbageCleaner(columns=numeric_features)

    # 2. Preprocessing Steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=[SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER]))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

    # 3. Full Pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', cleaner),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return pipeline

class DepressionAnalysisSystem:
    def __init__(self):
        self.student_pipeline = create_student_pipeline()
        self.worker_pipeline = create_worker_pipeline()

    def fit(self, data, target_col='Depression'):
        # Split data
        student_data = data[data['Working Professional or Student'] == 'Student'].copy()
        worker_data = data[data['Working Professional or Student'] == 'Working Professional'].copy()

        # Fit Student Pipeline
        if not student_data.empty:
            X_student = student_data.drop(columns=[target_col])
            y_student = student_data[target_col]
            self.student_pipeline.fit(X_student, y_student)

        # Fit Worker Pipeline
        if not worker_data.empty:
            X_worker = worker_data.drop(columns=[target_col])
            y_worker = worker_data[target_col]
            self.worker_pipeline.fit(X_worker, y_worker)
            
        return self

    def predict(self, data):
        # Initialize result array
        results = pd.Series(index=data.index, dtype=int)
        
        # Split data
        student_mask = data['Working Professional or Student'] == 'Student'
        worker_mask = data['Working Professional or Student'] == 'Working Professional'
        
        student_data = data[student_mask].copy()
        worker_data = data[worker_mask].copy()

        # Predict Student
        if not student_data.empty:
            student_preds = self.student_pipeline.predict(student_data)
            results.loc[student_mask] = student_preds

        # Predict Worker
        if not worker_data.empty:
            worker_preds = self.worker_pipeline.predict(worker_data)
            results.loc[worker_mask] = worker_preds
            
        return results

    def predict_proba(self, data):
        # Initialize result dataframe
        # Assuming binary classification (0 and 1)
        results = pd.DataFrame(index=data.index, columns=[0, 1], dtype=float)
        
        # Split data
        student_mask = data['Working Professional or Student'] == 'Student'
        worker_mask = data['Working Professional or Student'] == 'Working Professional'
        
        student_data = data[student_mask].copy()
        worker_data = data[worker_mask].copy()

        # Predict Student
        if not student_data.empty:
            student_probs = self.student_pipeline.predict_proba(student_data)
            results.loc[student_mask] = student_probs

        # Predict Worker
        if not worker_data.empty:
            worker_probs = self.worker_pipeline.predict_proba(worker_data)
            results.loc[worker_mask] = worker_probs
            
        return results
