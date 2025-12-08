
# Feature Lists
NUMERIC_FEATURES_COMMON = [
    'Age', 
    'Work/Study Hours', 
    'Financial Stress'
]

CATEGORICAL_FEATURES_COMMON = [
    'Gender', 
    'City', 
    'Degree', 
    'Have you ever had suicidal thoughts ?', 
    'Family History of Mental Illness'
]

ORDINAL_FEATURES = [
    'Sleep Duration', 
    'Dietary Habits'
]

# Student Specific Features
STUDENT_SPECIFIC_NUMERIC = [
    'Academic Pressure', 
    'CGPA', 
    'Study Satisfaction'
]
# Note: Profession is not relevant for students in the same way, but we might want to drop it or handle it.
# Based on notebooks, students have these specific columns populated.

# Worker Specific Features
WORKER_SPECIFIC_NUMERIC = [
    'Work Pressure', 
    'Job Satisfaction'
]
WORKER_SPECIFIC_CATEGORICAL = [
    'Profession'
]

# Ordinal Mappings
SLEEP_DURATION_ORDER = ['Less than or equal 6 hours', '6-8 hours', 'More than 8 hours']
DIETARY_HABITS_ORDER = ['Unhealthy', 'Moderate', 'Healthy']

# Garbage/Cleaning Constants
INVALID_VALUES = ['Error', 'Null', '??', '#VALUE!', 'Not Available', 'Twenty', 'nan']
TEXT_TO_NUMERIC_MAPPING = {
    'Low': 1,
    'High': 5
}

# Selected Features (Dynamic Selection Implemented in Pipeline)
# The lists below are removed as feature selection is now handled dynamically 
# by AggregatedFeatureSelector in the pipeline.
