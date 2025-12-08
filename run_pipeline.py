import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.pipeline import DepressionAnalysisSystem
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('raw/raw_depression_dataset.csv')
    except FileNotFoundError:
        print("Error: 'raw/raw_depression_dataset.csv' not found. Please ensure the dataset is in the correct location.")
        return

    print(f"Data loaded: {df.shape}")
    
    # 1. Split by Role
    student_df = df[df['Working Professional or Student'] == 'Student']
    worker_df = df[df['Working Professional or Student'] == 'Working Professional']
    
    # 2. Split each group into Train/Test (Stratified by Depression)
    # Student Split
    student_train, student_test = train_test_split(
        student_df, test_size=0.2, random_state=42, stratify=student_df['Depression']
    )
    
    # Worker Split
    worker_train, worker_test = train_test_split(
        worker_df, test_size=0.2, random_state=42, stratify=worker_df['Depression']
    )
    
    # 3. Combine to create final Train and Test sets
    train_df = pd.concat([student_train, worker_train], ignore_index=True)
    test_df = pd.concat([student_test, worker_test], ignore_index=True)
    
    print(f"Total Training set: {train_df.shape} (Students: {student_train.shape[0]}, Workers: {worker_train.shape[0]})")
    print(f"Total Test set: {test_df.shape} (Students: {student_test.shape[0]}, Workers: {worker_test.shape[0]})")

    # Initialize and Train System
    print("\nInitializing and Training DepressionAnalysisSystem...")
    system = DepressionAnalysisSystem()
    system.fit(train_df, target_col='Depression')
    
    # Save the model
    system.save_model('model/depression_analysis_model.joblib')

    # Evaluation
    y_pred = system.predict(test_df)

    print("\n--- Evaluation by Role ---")
    # Student
    student_mask = test_df['Working Professional or Student'] == 'Student'
    if student_mask.any():
        print("\nStudent Performance:")
        print(classification_report(test_df.loc[student_mask, 'Depression'], y_pred[student_mask]))
    
    # Worker
    worker_mask = test_df['Working Professional or Student'] == 'Working Professional'
    if worker_mask.any():
        print("\nWorker Performance:")
        print(classification_report(test_df.loc[worker_mask, 'Depression'], y_pred[worker_mask]))

if __name__ == "__main__":
    main()
