import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from src.config import (
    INVALID_VALUES, TEXT_TO_NUMERIC_MAPPING,
    SLEEP_DURATION_ORDER, DIETARY_HABITS_ORDER
)

class PandasOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Applies OrdinalEncoder to specific columns and returns the entire DataFrame.
    """
    def __init__(self, mapping):
        """
        Args:
            mapping (dict): Dictionary of {column: categories_list}
        """
        self.mapping = mapping
        self.encoders = {}

    def fit(self, X, y=None):
        for col, categories in self.mapping.items():
            if col in X.columns:
                encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=np.nan)
                encoder.fit(X[[col]])
                self.encoders[col] = encoder
        return self

    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[[col]]).flatten()
        return X

from src.config import INVALID_VALUES, TEXT_TO_NUMERIC_MAPPING

class GarbageCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean garbage values from specific columns.
    - Replaces 'Low'/'High' with numeric values.
    - Replaces invalid strings with NaN.
    - Coerces to numeric.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            return X
            
        for col in self.columns:
            if col in X.columns:
                # Map text to numbers if applicable
                X[col] = X[col].replace(TEXT_TO_NUMERIC_MAPPING)
                
                # Replace invalid values with NaN
                X[col] = X[col].replace(INVALID_VALUES, np.nan)
                
                # Coerce to numeric
                X[col] = pd.to_numeric(X[col], errors='coerce')
        return X

class RoleCorrector(BaseEstimator, TransformerMixin):
    """
    Corrects 'Working Professional or Student' role based on data completeness.
    Implements strict_role_correction_v3 logic from notebook.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Working Professional or Student' not in X.columns:
            return X

        # Calculate scores
        s_cols = ['Academic Pressure', 'Study Satisfaction', 'CGPA']
        w_cols = ['Work Pressure', 'Job Satisfaction']
        
        # Ensure columns exist
        s_cols = [c for c in s_cols if c in X.columns]
        w_cols = [c for c in w_cols if c in X.columns]
        
        # Score: count non-nulls
        s_score = X[s_cols].notna().sum(axis=1)
        w_score = X[w_cols].notna().sum(axis=1)
        
        # Vectorized logic for efficiency
        # Default: keep original
        new_role = X['Working Professional or Student'].copy()
        
        # Logic for Student
        student_mask = X['Working Professional or Student'] == 'Student'
        
        # Switch to Worker: (S=0, W=2) or (S=1, W=2)
        switch_to_worker = student_mask & ((s_score == 0) & (w_score == 2) | (s_score == 1) & (w_score == 2))
        new_role.loc[switch_to_worker] = 'Working Professional'
        
        # Logic for Worker
        worker_mask = X['Working Professional or Student'] == 'Working Professional'
        
        # Switch to Student: (W=0, S=3), (W=1, S=3), (W=1, S=2), (W=0, S=2)
        switch_to_student = worker_mask & (
            ((w_score == 0) & (s_score == 3)) | 
            ((w_score == 1) & (s_score == 3)) | 
            ((w_score == 1) & (s_score == 2)) |
            ((w_score == 0) & (s_score == 2))
        )
        new_role.loc[switch_to_student] = 'Student'
        
        # Update Role
        X['Working Professional or Student'] = new_role
        
        # Cleanup based on new role
        # If Student -> Set Work cols to NaN
        # If Worker -> Set Student cols to NaN
        
        final_student_mask = X['Working Professional or Student'] == 'Student'
        final_worker_mask = X['Working Professional or Student'] == 'Working Professional'
        
        # Clean Student: Set Work cols to NaN
        for col in w_cols:
            X.loc[final_student_mask, col] = np.nan
            
        # Clean Worker: Set Student cols to NaN
        for col in s_cols:
            X.loc[final_worker_mask, col] = np.nan
            
        return X

class CityCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans 'City' column: standardizes text, fixes typos, filters valid cities.
    """
    def __init__(self):
        self.valid_cities = [
            'Ahmedabad', 'Agra', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi', 
            'Faridabad', 'Ghaziabad', 'Gurgaon', 'Hyderabad', 'Indore', 
            'Jaipur', 'Kalyan', 'Kanpur', 'Kolkata', 'Lucknow', 'Ludhiana', 
            'Meerut', 'Mumbai', 'Nagpur', 'Nashik', 'Patna', 'Pune', 
            'Rajkot', 'Srinagar', 'Surat', 'Thane', 'Vadodara', 'Varanasi', 
            'Vasai-Virar', 'Visakhapatnam'
        ]
        self.typo_map = {
            'Molkata': 'Kolkata', 'Tolkata': 'Kolkata',
            'Khaziabad': 'Ghaziabad', 'Galesabad': 'Ghaziabad',
            'Less Delhi': 'Delhi',
            'Nalyan': 'Kalyan', 'Less Than 5 Kalyan': 'Kalyan',
            'Ishanabad': 'Ghaziabad' 
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'City' not in X.columns:
            return X
            
        def normalize_text(val):
            val = str(val).strip()
            if val.lower() == 'nan': return np.nan
            # Remove special chars at start
            val = re.sub(r'^[^a-zA-Z]+', '', val)
            return val.title()

        X['City'] = X['City'].apply(normalize_text)
        X['City'] = X['City'].replace(self.typo_map)
        
        # Filter valid
        X.loc[~X['City'].isin(self.valid_cities), 'City'] = np.nan
        
        return X

class DegreeCleaner(BaseEstimator, TransformerMixin):
    """
    Groups 'Degree' into 4 categories: Bachelor, Master, Doctorate, High School.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Degree' not in X.columns:
            return X
            
        def group_degree(val):
            val = str(val).lower().strip()
            if 'phd' in val: return 'Doctorate'
            if any(x in val for x in ['m.tech', 'mtech', 'me', 'm.sc', 'msc', 'm.s', 'm.com', 'mcom', 'mba', 'mca', 'm.ed', 'med', 'llm', 'ma', 'm.a', 'm.arch', 'march', 'm.pharm', 'mpharm', 'md', 'doctor', 'ca', 'aca','mhm']): return 'Master'
            if any(x in val for x in ['b.tech', 'btech', 'b tech', 'be', 'bachelor', 'b.sc', 'bsc', 'b.com', 'bcom', 'bba', 'bb', 'b ba', 'bca', 'mbbs', 'b.pharm', 'bpharm', 'b.ph', 'b.arch', 'barch', 'b.ed', 'bed', 'llb', 'll b', 'ba', 'b.a', 'bhm']): return 'Bachelor'
            if any(x in val for x in ['class 12', '12th', 'hsc', 'class 11']): return 'High School'
            return np.nan

        X['Degree'] = X['Degree'].apply(group_degree)
        return X

class ProfessionCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans 'Profession' column: standardizes, fixes typos, groups into domains.
    """
    def __init__(self):
        self.typo_map = {
            'Finanancial Analyst': 'Financial Analyst',
            'Ux/Ui Designer': 'UX/UI Designer',
            'Hr Manager': 'HR Manager'
        }
        self.garbage_values = [
            'Yogesh', 'Pranav', 'Yuvraj', 'Dev', 'Nalini', 'Vivaan', 'Ritik', 'Bhavesh',
            'Patna', 'Nagpur', 'Visakhapatnam', 'Familyvirar', 'Bhopal', 'Kalyan', 
            'B.Com', 'Be', 'Mba', 'Llm', 'Bca', 'Bba', 'Mbbs', 'B.Ed', 'M.Ed', 'Phd', 'Degree',
            'Unveil', 'Moderate', 'Null', 'Error', 'Academic', 'Profession', 'Working Professional'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Profession' not in X.columns:
            return X
            
        # Standardize
        X['Profession'] = X['Profession'].astype(str).str.strip().str.title()
        X['Profession'] = X['Profession'].replace('Nan', np.nan)
        X['Profession'] = X['Profession'].replace(self.typo_map)
        
        # Remove garbage
        X.loc[X['Profession'].isin(self.garbage_values), 'Profession'] = np.nan
        
        # Impute Student
        if 'Working Professional or Student' in X.columns:
            X.loc[X['Working Professional or Student'] == 'Student', 'Profession'] = 'Student'
            
            # Fix Worker claiming to be Student -> Unspecified
            mask_worker_wrong = (X['Working Professional or Student'] == 'Working Professional') & (X['Profession'] == 'Student')
            X.loc[mask_worker_wrong, 'Profession'] = 'Unspecified'
            
            # Fill missing Worker -> Unspecified
            mask_worker_missing = (X['Working Professional or Student'] == 'Working Professional') & (X['Profession'].isnull())
            X.loc[mask_worker_missing, 'Profession'] = 'Unspecified'

        # Group Domains
        def group_domain(val):
            val = str(val).lower().strip()
            if val == 'student': return 'Student'
            if any(x in val for x in ['teacher', 'educational consultant', 'academic', 'researcher']): return 'Education/Research'
            if any(x in val for x in ['doctor', 'pharmacist', 'chemist', 'medical']): return 'Medical/Health'
            if any(x in val for x in ['data scientist', 'software engineer', 'developer', 'dev']): return 'Tech/IT'
            if any(x in val for x in ['manager', 'analyst', 'accountant', 'consultant', 'entrepreneur', 'sales', 'marketing', 'investment banker']): return 'Business/Finance'
            if any(x in val for x in ['writer', 'designer', 'marketer', 'digital']): return 'Creative/Media'
            if any(x in val for x in ['engineer', 'architect']): return 'Engineering/Architecture'
            if any(x in val for x in ['lawyer', 'judge']): return 'Legal'
            if any(x in val for x in ['chef', 'pilot', 'support', 'travel', 'electrician', 'plumber']): return 'Service/Operations'
            return 'Other Professions'

        X['Profession'] = X['Profession'].apply(group_domain)
        return X

class SleepDurationCleaner(BaseEstimator, TransformerMixin):
    """
    Categorizes 'Sleep Duration' into 3 groups.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'Sleep Duration' not in X.columns:
            return X
            
        def categorize(val):
            val = str(val).lower().strip()
            if any(x in val for x in ['more than 8', '8-9', '9-11', '10-11']): return 'More than 8 hours'
            if any(x in val for x in ['7-8', '6-7', '6-8', '8 hours']): return '6-8 hours'
            if any(x in val for x in ['less than 5', '5-6', '3-4', '4-5', '1-6', '4-6', '2-3', '3-6', 'than 5', '1-2', '1-3']): return 'Less than or equal 6 hours'
            return np.nan

        X['Sleep Duration'] = X['Sleep Duration'].apply(categorize)
        return X

class WorkStudyHoursOutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips 'Work/Study Hours' using IQR method from training data.
    """
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        self.median_val = None

    def fit(self, X, y=None):
        if 'Work/Study Hours' in X.columns:
            col = X['Work/Study Hours']
            # Coerce to numeric first
            col = pd.to_numeric(col, errors='coerce').dropna()
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bound = Q1 - 1.5 * IQR
            self.upper_bound = Q3 + 1.5 * IQR
            self.median_val = col.median()
        return self

    def transform(self, X):
        X = X.copy()
        if 'Work/Study Hours' not in X.columns or self.median_val is None:
            return X
            
        X['Work/Study Hours'] = pd.to_numeric(X['Work/Study Hours'], errors='coerce')
        mask = (X['Work/Study Hours'] < self.lower_bound) | (X['Work/Study Hours'] > self.upper_bound)
        X.loc[mask, 'Work/Study Hours'] = self.median_val
        return X

class OutlierToNaN(BaseEstimator, TransformerMixin):
    """
    Replaces values outside the specified range with NaN.
    """
    def __init__(self, ranges=None):
        """
        Args:
            ranges (dict): Dictionary where key is column name and value is (min, max) tuple.
                           Values outside [min, max] will be set to NaN.
        """
        self.ranges = ranges

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.ranges is None:
            return X

        for col, (min_val, max_val) in self.ranges.items():
            if col in X.columns:
                mask = (X[col] < min_val) | (X[col] > max_val)
                X.loc[mask, col] = np.nan
        return X

class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    """
    Applies KNNImputer on scaled data but returns data in original scale.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Store feature names if available (for pandas output)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns
            
        # 1. Scale
        X_scaled = self.scaler.fit_transform(X)
        # 2. Fit Imputer
        self.imputer.fit(X_scaled)
        return self

    def transform(self, X):
        # 1. Scale
        X_scaled = self.scaler.transform(X)
        # 2. Impute
        X_imputed_scaled = self.imputer.transform(X_scaled)
        # 3. Inverse Scale
        X_imputed = self.scaler.inverse_transform(X_imputed_scaled)
        
        # Return DataFrame if input was DataFrame
        if self.feature_names_in_ is not None:
            return pd.DataFrame(X_imputed, columns=self.feature_names_in_, index=X.index)
        return X_imputed

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clips values in specified columns to a given range.
    """
    def __init__(self, ranges=None):
        """
        Args:
            ranges (dict): Dictionary where key is column name and value is (min, max) tuple.
                           Example: {'Age': (10, 90), 'CGPA': (0, 10)}
        """
        self.ranges = ranges

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.ranges is None:
            return X

        for col, (min_val, max_val) in self.ranges.items():
            if col in X.columns:
                # Coerce to numeric first
                X[col] = pd.to_numeric(X[col], errors='coerce')
                mask = (X[col] < min_val) | (X[col] > max_val)
                X.loc[mask, col] = np.nan
        return X

class CategoricalTypeCaster(BaseEstimator, TransformerMixin):
    """
    Casts specified columns to string type to ensure uniform input for Encoders.
    Handles NaN values by converting them to 'Missing' or keeping as is (if we want OneHotEncoder to handle NaNs).
    Here we convert everything to string, so NaNs become 'nan'.
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            return X
            
        for col in self.columns:
            if col in X.columns:
                # Convert to string, but keep real NaNs if possible? 
                # Sklearn OneHotEncoder prefers uniform types. 
                # Simplest is to convert EVERYTHING to string, including NaNs -> 'nan'
                X[col] = X[col].astype(str)
                
                # Optional: Map 'nan' string back to np.nan if we want OneHotEncoder to treat it as missing
                # But 'nan' as a category is also fine for RandomForest
                # Let's keep it as 'nan' string to avoid mixed type error (float nan vs string)
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from a DataFrame.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

class SequentialImputer(BaseEstimator, TransformerMixin):
    """
    Implements the sequential KNN imputation logic from the notebooks.
    1. Impute CGPA (Student only).
    2. Impute Likert scales (Academic/Work Pressure, Satisfaction, Financial Stress).
    3. Impute Age.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer_cgpa = ScaledKNNImputer(n_neighbors=n_neighbors)
        self.imputer_likert_student = ScaledKNNImputer(n_neighbors=n_neighbors)
        self.imputer_likert_worker = ScaledKNNImputer(n_neighbors=n_neighbors)
        self.imputer_age_student = ScaledKNNImputer(n_neighbors=n_neighbors)
        self.imputer_age_worker = ScaledKNNImputer(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        # We need to fit each internal imputer on the relevant subset of data
        # Note: This assumes X contains 'Working Professional or Student' column
        # If used inside a ColumnTransformer that drops it, this will fail.
        # So this transformer expects the full dataframe or at least one with the role column.
        
        role_col = 'Working Professional or Student'
        if role_col not in X.columns:
            # Fallback if role column is missing (should not happen in our pipeline design)
            return self

        # 1. CGPA (Student only)
        # Features: ['Academic Pressure', 'Study Satisfaction', 'Age', 'CGPA']
        student_mask = X[role_col] == 'Student'
        if student_mask.any():
            cgpa_cols = ['Academic Pressure', 'Study Satisfaction', 'Age', 'CGPA']
            # Filter columns that exist
            cgpa_cols = [c for c in cgpa_cols if c in X.columns]
            self.imputer_cgpa.fit(X.loc[student_mask, cgpa_cols])

        # 2. Likert
        # Student
        student_likert_cols = ['Age', 'CGPA', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress']
        student_likert_cols = [c for c in student_likert_cols if c in X.columns]
        if student_mask.any():
            self.imputer_likert_student.fit(X.loc[student_mask, student_likert_cols])
            
        # Worker
        worker_mask = X[role_col] == 'Working Professional'
        worker_likert_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Financial Stress']
        worker_likert_cols = [c for c in worker_likert_cols if c in X.columns]
        if worker_mask.any():
            self.imputer_likert_worker.fit(X.loc[worker_mask, worker_likert_cols])

        # 3. Age
        # Student
        student_age_cols = ['CGPA', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress', 'Age']
        student_age_cols = [c for c in student_age_cols if c in X.columns]
        if student_mask.any():
            self.imputer_age_student.fit(X.loc[student_mask, student_age_cols])
            
        # Worker
        worker_age_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress', 'Age']
        worker_age_cols = [c for c in worker_age_cols if c in X.columns]
        if worker_mask.any():
            self.imputer_age_worker.fit(X.loc[worker_mask, worker_age_cols])
            
        return self

    def transform(self, X):
        X = X.copy()
        role_col = 'Working Professional or Student'
        if role_col not in X.columns:
            return X
            
        student_mask = X[role_col] == 'Student'
        worker_mask = X[role_col] == 'Working Professional'

        # --- 1. CGPA ---
        # Hard Rule: Worker CGPA = 0
        if 'CGPA' in X.columns:
            X.loc[worker_mask, 'CGPA'] = 0
        
        # Impute Student CGPA
        cgpa_cols = ['Academic Pressure', 'Study Satisfaction', 'Age', 'CGPA']
        cgpa_cols = [c for c in cgpa_cols if c in X.columns]
        
        if student_mask.any() and 'CGPA' in X.columns:
            # We need to temporarily fill other NaNs to allow KNN to work best? 
            # Or just rely on KNNImputer handling NaNs in features. KNNImputer handles it.
            imputed = self.imputer_cgpa.transform(X.loc[student_mask, cgpa_cols])
            # Update CGPA column
            if isinstance(imputed, pd.DataFrame):
                X.loc[student_mask, 'CGPA'] = imputed['CGPA']
            else:
                # Assuming order is preserved and last col is CGPA if list matches
                # But ScaledKNNImputer returns array or DF. 
                # Let's rely on column name matching if DF, else index
                cgpa_idx = cgpa_cols.index('CGPA')
                X.loc[student_mask, 'CGPA'] = imputed[:, cgpa_idx]

        # --- 2. Likert ---
        # Hard Rules
        if 'Work Pressure' in X.columns: X.loc[student_mask, 'Work Pressure'] = 0
        if 'Job Satisfaction' in X.columns: X.loc[student_mask, 'Job Satisfaction'] = 0
        
        if 'Academic Pressure' in X.columns: X.loc[worker_mask, 'Academic Pressure'] = 0
        if 'Study Satisfaction' in X.columns: X.loc[worker_mask, 'Study Satisfaction'] = 0
        # Worker CGPA already 0

        # Impute Student Likert
        student_likert_cols = ['Age', 'CGPA', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress']
        student_likert_cols = [c for c in student_likert_cols if c in X.columns]
        student_targets = ['Academic Pressure', 'Study Satisfaction', 'Financial Stress']
        
        if student_mask.any():
            imputed = self.imputer_likert_student.transform(X.loc[student_mask, student_likert_cols])
            for col in student_targets:
                if col in X.columns:
                    if isinstance(imputed, pd.DataFrame):
                        X.loc[student_mask, col] = imputed[col]
                    else:
                        idx = student_likert_cols.index(col)
                        X.loc[student_mask, col] = imputed[:, idx]

        # Impute Worker Likert
        worker_likert_cols = ['Age', 'Work Pressure', 'Job Satisfaction', 'Financial Stress']
        worker_likert_cols = [c for c in worker_likert_cols if c in X.columns]
        worker_targets = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']
        
        if worker_mask.any():
            imputed = self.imputer_likert_worker.transform(X.loc[worker_mask, worker_likert_cols])
            for col in worker_targets:
                if col in X.columns:
                    if isinstance(imputed, pd.DataFrame):
                        X.loc[worker_mask, col] = imputed[col]
                    else:
                        idx = worker_likert_cols.index(col)
                        X.loc[worker_mask, col] = imputed[:, idx]

        # --- 3. Age ---
        # Impute Student Age
        student_age_cols = ['CGPA', 'Academic Pressure', 'Study Satisfaction', 'Financial Stress', 'Age']
        student_age_cols = [c for c in student_age_cols if c in X.columns]
        
        if student_mask.any() and 'Age' in X.columns:
            imputed = self.imputer_age_student.transform(X.loc[student_mask, student_age_cols])
            if isinstance(imputed, pd.DataFrame):
                X.loc[student_mask, 'Age'] = imputed['Age'].astype(int)
            else:
                idx = student_age_cols.index('Age')
                X.loc[student_mask, 'Age'] = imputed[:, idx].astype(int)

        # Impute Worker Age
        worker_age_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress', 'Age']
        worker_age_cols = [c for c in worker_age_cols if c in X.columns]
        
        if worker_mask.any() and 'Age' in X.columns:
            imputed = self.imputer_age_worker.transform(X.loc[worker_mask, worker_age_cols])
            if isinstance(imputed, pd.DataFrame):
                X.loc[worker_mask, 'Age'] = imputed['Age'].astype(int)
            else:
                idx = worker_age_cols.index('Age')
                X.loc[worker_mask, 'Age'] = imputed[:, idx].astype(int)

        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates interaction features based on the notebook logic.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        role_col = 'Working Professional or Student'
        
        # Ensure numeric columns are numeric
        numeric_cols = ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
                       'Job Satisfaction', 'Financial Stress', 'Work/Study Hours', 'Age', 
                       'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
                       'Family History of Mental Illness']
        
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # --- Common Features ---
        if 'Sleep Duration' in X.columns and 'Dietary Habits' in X.columns:
            X['Physical_Health'] = X['Sleep Duration'] * X['Dietary Habits']
            
        if 'Work/Study Hours' in X.columns and 'Sleep Duration' in X.columns:
            X['Work_Sleep'] = X['Work/Study Hours'] * X['Sleep Duration']
            
        if 'Have you ever had suicidal thoughts ?' in X.columns and 'Family History of Mental Illness' in X.columns:
            X['Suicidal_Family_Interaction'] = X['Have you ever had suicidal thoughts ?'] * X['Family History of Mental Illness']
            
        if 'Financial Stress' in X.columns and 'Sleep Duration' in X.columns:
            X['Sleep_Vulnerability'] = X['Financial Stress'] * X['Sleep Duration']
            
        if 'Financial Stress' in X.columns and 'Work/Study Hours' in X.columns:
            X['Financial_Work_Interaction'] = X['Financial Stress'] * X['Work/Study Hours']

        # Age Pressure (Role dependent)
        if 'Age' in X.columns:
            # Avoid division by zero by adding 5 (as per notebook)
            age_denom = X['Age'] + 5
            
            if 'Academic Pressure' in X.columns and 'Work Pressure' in X.columns:
                # Vectorized condition
                X['Age_Pressure'] = np.where(
                    X[role_col] == 'Student',
                    X['Academic Pressure'] / age_denom,
                    X['Work Pressure'] / age_denom
                )
            elif 'Academic Pressure' in X.columns:
                 X['Age_Pressure'] = X['Academic Pressure'] / age_denom
            elif 'Work Pressure' in X.columns:
                 X['Age_Pressure'] = X['Work Pressure'] / age_denom

        # --- Student Features ---
        if 'Academic Pressure' in X.columns and 'Work/Study Hours' in X.columns:
            # We calculate Burnout_Load generically, then override or use specific logic
            # Notebook: Student Burnout = Academic * Hours
            # Notebook: Worker Burnout = Work * Hours
            # Let's use np.where again if both exist
            
            if 'Work Pressure' in X.columns:
                X['Burnout_Load'] = np.where(
                    X[role_col] == 'Student',
                    X['Academic Pressure'] * X['Work/Study Hours'],
                    X['Work Pressure'] * X['Work/Study Hours']
                )
            else:
                 X['Burnout_Load'] = X['Academic Pressure'] * X['Work/Study Hours']

        # Effort Reward Imbalance
        if 'Burnout_Load' in X.columns:
            if 'Study Satisfaction' in X.columns and 'Job Satisfaction' in X.columns:
                X['Effort_Reward_Imbalance'] = np.where(
                    X[role_col] == 'Student',
                    X['Burnout_Load'] * (6 - X['Study Satisfaction']),
                    X['Burnout_Load'] * (6 - X['Job Satisfaction'])
                )

        # Performance Anxiety (Student)
        if 'CGPA' in X.columns and 'Academic Pressure' in X.columns:
             # Only relevant for students, but we can calc for all and it will be 0 for workers (since CGPA=0)
             X['Performance_Anxiety_Score'] = X['CGPA'] * X['Academic Pressure']

        # Trapped Score (Worker)
        if 'Financial Stress' in X.columns and 'Job Satisfaction' in X.columns:
            # Only relevant for workers
            X['Trapped_Score'] = X['Financial Stress'] + (6 - X['Job Satisfaction'])
            # Zero out for students? Notebook doesn't explicitly say, but implies separation.
            # Let's keep it calculated for all, or mask it. 
            # If we follow notebook strict separation, we should probably mask.
            # But RandomForest handles 0/irrelevant well.
            if role_col in X.columns:
                 X.loc[X[role_col] == 'Student', 'Trapped_Score'] = 0



        return X.fillna(0) # Fill any NaNs created by engineering with 0
class AggregatedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10):
        self.n_features = n_features
        self.selected_features_ = None

    def fit(self, X, y):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Fill NaNs for selection methods that don't handle them
        X_filled = X.fillna(0)
        
        # 1. Pearson Correlation
        corrs = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
        
        # 2. Mutual Information
        mi = mutual_info_classif(X_filled, y, random_state=42)
        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        
        # 3. LassoCV
        lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
        lasso.fit(X_filled, y)
        coef_abs = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
        
        # 4. RandomForest Importance
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_filled, y)
        rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Aggregate Rankings
        rank_df = pd.DataFrame({
            'corr_rank': corrs.rank(ascending=False),
            'mi_rank': mi_series.rank(ascending=False),
            'lasso_rank': coef_abs.rank(ascending=False),
            'rf_rank': rf_imp.rank(ascending=False)
        }).fillna(1e6) # Treat missing as low importance
        
        rank_df['mean_rank'] = rank_df.mean(axis=1)
        self.selected_features_ = rank_df.sort_values('mean_rank').head(self.n_features).index.tolist()
        
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("AggregatedFeatureSelector has not been fitted yet.")
            
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # If input is numpy array, we can't select by name easily unless we tracked columns
            # Assuming input to transform is same structure as fit
            # Ideally this transformer is used where X is a DataFrame (e.g. after FeatureEngineer)
            raise ValueError("AggregatedFeatureSelector requires pandas DataFrame input.")
