import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
import warnings
import joblib
import os

from src.config import (
    NUMERIC_FEATURES_COMMON, CATEGORICAL_FEATURES_COMMON, ORDINAL_FEATURES,
    STUDENT_SPECIFIC_NUMERIC, WORKER_SPECIFIC_NUMERIC, WORKER_SPECIFIC_CATEGORICAL,
    SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER
)
from src.transformers import (
    GarbageCleaner, OutlierToNaN, ScaledKNNImputer, OutlierClipper, DataFrameSelector,
    SequentialImputer, FeatureEngineer, PandasOrdinalEncoder,
    RoleCorrector, CityCleaner, DegreeCleaner, ProfessionCleaner, 
    SleepDurationCleaner, WorkStudyHoursOutlierClipper, AggregatedFeatureSelector,
    CategoricalTypeCaster, FrequencyEncoder
)

def create_student_pipeline():
    """Creates the pipeline for student data."""
    # Base features
    numeric_features = NUMERIC_FEATURES_COMMON + STUDENT_SPECIFIC_NUMERIC
    # Remove City from categorical features for OneHotEncoding, as we will use FrequencyEncoding
    categorical_features = [f for f in CATEGORICAL_FEATURES_COMMON if f != 'City']
    high_card_features = ['City']
    ordinal_features = ORDINAL_FEATURES

    # Engineered features (Numeric)
    engineered_features = [
        'Physical_Health', 'Work_Sleep', 'Suicidal_Family_Interaction', 
        'Sleep_Vulnerability', 'Financial_Work_Interaction', 'Age_Pressure', 
        'Burnout_Load', 'Effort_Reward_Imbalance', 'Performance_Anxiety_Score'
    ]
    
    # All numeric features for the final scaler (Base Numeric + Ordinal (Encoded) + Engineered)
    # Note: Ordinal features become numeric after PandasOrdinalEncoder
    # all_numeric_features = numeric_features + ordinal_features + engineered_features

    # 1. Cleaning Step
    city_cleaner = CityCleaner()
    degree_cleaner = DegreeCleaner()
    profession_cleaner = ProfessionCleaner()
    sleep_cleaner = SleepDurationCleaner()
    cleaner = GarbageCleaner(columns=numeric_features)
    hours_clipper = WorkStudyHoursOutlierClipper()
    type_caster = CategoricalTypeCaster(columns=categorical_features)

    cleaning_pipeline = Pipeline(steps=[
        ('city_cleaner', city_cleaner),
        ('degree_cleaner', degree_cleaner),
        ('profession_cleaner', profession_cleaner),
        ('sleep_cleaner', sleep_cleaner),
        ('cleaner', cleaner),
        ('hours_clipper', hours_clipper),
        ('type_caster', type_caster)
    ])

    # 2. Sequential Imputation (Handles CGPA -> Likert -> Age)
    imputation_pipeline = SequentialImputer(n_neighbors=5)

    # 3. Outlier Clipping (Moved to before encoding, matching notebook)
    outlier_clipper = OutlierClipper(ranges={
        'Age': (10, 90),
        'CGPA': (0, 10),
        'Academic Pressure': (0, 5),
        'Study Satisfaction': (0, 5),
        'Financial Stress': (0, 5)
    })

    # 4. Master Encoding (Ordinal + OneHot + Frequency)
    # This replaces the separate ordinal and onehot/frequency steps and handles feature selection
    ordinal_encoder = OrdinalEncoder(
        categories=[SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER],
        handle_unknown='use_encoded_value',
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )

    master_encoder = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_encoder, ordinal_features),
            ('high_card', FrequencyEncoder(), high_card_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop', # Drop Name, Role, etc.
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # 5. Feature Engineering (Works on encoded data)
    feature_engineer = FeatureEngineer()

    # 6. Final Preprocessing (Scaling)
    final_scaler = StandardScaler().set_output(transform="pandas")

    # 7. Feature Selection (Dynamic)
    feature_selector = AggregatedFeatureSelector(n_features=10)

    # 8. Full Pipeline
    pipeline = Pipeline([
        ('cleaning', cleaning_pipeline),
        ('imputation', imputation_pipeline),
        ('outlier_clipping', outlier_clipper),
        ('encoding', master_encoder),
        ('feature_engineering', feature_engineer),
        ('scaling', final_scaler),
        ('feature_selection', feature_selector),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def create_worker_pipeline():
    """Creates the pipeline for working professional data."""
    # Base features
    # Base features
    numeric_features = NUMERIC_FEATURES_COMMON + WORKER_SPECIFIC_NUMERIC
    # Remove City from categorical features for OneHotEncoding
    categorical_features = [f for f in (CATEGORICAL_FEATURES_COMMON + WORKER_SPECIFIC_CATEGORICAL) if f != 'City']
    high_card_features = ['City']
    ordinal_features = ORDINAL_FEATURES

    # Engineered features (Numeric)
    engineered_features = [
        'Physical_Health', 'Work_Sleep', 'Suicidal_Family_Interaction', 
        'Sleep_Vulnerability', 'Financial_Work_Interaction', 'Age_Pressure', 
        'Burnout_Load', 'Effort_Reward_Imbalance', 'Trapped_Score'
    ]
    
    # All numeric features
    # all_numeric_features = numeric_features + ordinal_features + engineered_features

    # 1. Cleaning Step
    city_cleaner = CityCleaner()
    degree_cleaner = DegreeCleaner()
    profession_cleaner = ProfessionCleaner()
    sleep_cleaner = SleepDurationCleaner()
    cleaner = GarbageCleaner(columns=numeric_features)
    hours_clipper = WorkStudyHoursOutlierClipper()
    type_caster = CategoricalTypeCaster(columns=categorical_features)

    cleaning_pipeline = Pipeline(steps=[
        ('city_cleaner', city_cleaner),
        ('degree_cleaner', degree_cleaner),
        ('profession_cleaner', profession_cleaner),
        ('sleep_cleaner', sleep_cleaner),
        ('cleaner', cleaner),
        ('hours_clipper', hours_clipper),
        ('type_caster', type_caster)
    ])

    # 2. Sequential Imputation
    imputation_pipeline = SequentialImputer(n_neighbors=5)

    # 3. Outlier Clipping
    outlier_clipper = OutlierClipper(ranges={
        'Age': (10, 90),
        'Work Pressure': (0, 5),
        'Job Satisfaction': (0, 5),
        'Financial Stress': (0, 5)
    })

    # 4. Master Encoding
    ordinal_encoder = OrdinalEncoder(
        categories=[SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER],
        handle_unknown='use_encoded_value',
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )

    master_encoder = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_encoder, ordinal_features),
            ('high_card', FrequencyEncoder(), high_card_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # 5. Feature Engineering
    feature_engineer = FeatureEngineer()

    # 6. Final Preprocessing (Scaling)
    final_scaler = StandardScaler().set_output(transform="pandas")

    # 7. Feature Selection (Dynamic)
    feature_selector = AggregatedFeatureSelector(n_features=10)

    # 8. Full Pipeline
    pipeline = Pipeline([
        ('cleaning', cleaning_pipeline),
        ('imputation', imputation_pipeline),
        ('outlier_clipping', outlier_clipper),
        ('encoding', master_encoder),
        ('feature_engineering', feature_engineer),
        ('scaling', final_scaler),
        ('feature_selection', feature_selector),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

class DepressionAnalysisSystem:
    def __init__(self):
        self.student_pipeline = create_student_pipeline()
        self.worker_pipeline = create_worker_pipeline()
        self.role_corrector = RoleCorrector()

    def fit(self, data, target_col='Depression'):
        # Correct Roles first
        data = self.role_corrector.transform(data)

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
        
        # Correct Roles first
        data = self.role_corrector.transform(data)
        
        # Split data
        student_mask = data['Working Professional or Student'] == 'Student'
        worker_mask = data['Working Professional or Student'] == 'Working Professional'
        
        student_data = data[student_mask].copy()
        worker_data = data[worker_mask].copy()

        # Predict Student
        if not student_data.empty:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
                student_preds = self.student_pipeline.predict(student_data)
            results.loc[student_mask] = student_preds

        # Predict Worker
        if not worker_data.empty:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
                worker_preds = self.worker_pipeline.predict(worker_data)
            results.loc[worker_mask] = worker_preds
            
        return results

    def predict_proba(self, data):
        # Initialize result dataframe
        # Assuming binary classification (0 and 1)
        results = pd.DataFrame(index=data.index, columns=[0, 1], dtype=float)
        
        # Correct Roles first
        data = self.role_corrector.transform(data)
        
        # Split data
        student_mask = data['Working Professional or Student'] == 'Student'
        worker_mask = data['Working Professional or Student'] == 'Working Professional'
        
        student_data = data[student_mask].copy()
        worker_data = data[worker_mask].copy()


        # Predict Student
        if not student_data.empty:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
                student_probs = self.student_pipeline.predict_proba(student_data)
            results.loc[student_mask] = student_probs

        # Predict Worker
        if not worker_data.empty:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="This Pipeline instance is not fitted yet")
                worker_probs = self.worker_pipeline.predict_proba(worker_data)
            results.loc[worker_mask] = worker_probs
            
        return results

    def save_model(self, filepath):
        """Saves the trained model to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Loads a trained model from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
