import pandas as pd
import numpy as np
import sys
import warnings
from src.pipeline import DepressionAnalysisSystem

# Suppress warnings
warnings.filterwarnings("ignore")

def get_input(prompt, type_func=str, valid_options=None, allow_nan=False):
    while True:
        user_input = input(prompt + ": ").strip()
        
        if not user_input:
            if allow_nan:
                return np.nan
            else:
                print("Input cannot be empty. Please try again.")
                continue

        if valid_options:
            # Case insensitive check
            matches = [opt for opt in valid_options if opt.lower() == user_input.lower()]
            if matches:
                return matches[0]
            else:
                print(f"Invalid input. Options: {', '.join(valid_options)}")
                continue
        
        try:
            return type_func(user_input)
        except ValueError:
            print(f"Invalid format. Expected {type_func.__name__}.")

def main():
    print("========================================")
    print("   Depression Risk Prediction System    ")
    print("========================================")
    print("Initializing system and training model...")
    print("Please wait, this may take a few seconds.")

    try:
        df = pd.read_csv('raw/raw_depression_dataset.csv')
    except FileNotFoundError:
        print("Error: 'raw/raw_depression_dataset.csv' not found.")
        return

    system = DepressionAnalysisSystem()
    system.fit(df, target_col='Depression')
    print("Model trained successfully!\n")

    print("Please answer the following questions to assess depression risk.")
    print("---------------------------------------------------------------")

    # 1. Role
    role = get_input("Are you a 'Student' or 'Working Professional'?", valid_options=['Student', 'Working Professional'])

    # 2. Common Inputs
    name = input("Name (Optional): ").strip()
    gender = get_input("Gender", valid_options=['Male', 'Female'])
    age = get_input("Age", type_func=int)
    city = input("City: ").strip()
    
    # Options that map well to the cleaner logic
    sleep_options = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
    sleep = get_input(f"Sleep Duration ({', '.join(sleep_options)})", valid_options=sleep_options)
    
    diet_options = ['Healthy', 'Moderate', 'Unhealthy']
    diet = get_input(f"Dietary Habits ({', '.join(diet_options)})", valid_options=diet_options)
    
    suicidal = get_input("Have you ever had suicidal thoughts?", valid_options=['Yes', 'No'])
    family_history = get_input("Family History of Mental Illness?", valid_options=['Yes', 'No'])
    
    hours = get_input("Average Work/Study Hours per day", type_func=float)
    financial_stress = get_input("Financial Stress (1-5)", type_func=float)

    # 3. Role Specific
    degree = np.nan
    profession = np.nan
    academic_pressure = np.nan
    work_pressure = np.nan
    cgpa = np.nan
    study_satisfaction = np.nan
    job_satisfaction = np.nan

    if role == 'Student':
        degree = input("Degree (e.g., B.Tech, B.Sc): ").strip()
        academic_pressure = get_input("Academic Pressure (1-5)", type_func=float)
        cgpa = get_input("CGPA (0-10)", type_func=float)
        study_satisfaction = get_input("Study Satisfaction (1-5)", type_func=float)
    else:
        degree = input("Degree (e.g., B.Tech, B.Sc): ").strip()
        profession = input("Profession: ").strip()
        work_pressure = get_input("Work Pressure (1-5)", type_func=float)
        job_satisfaction = get_input("Job Satisfaction (1-5)", type_func=float)

    # Construct Data
    input_data = pd.DataFrame([{
        'Name': name,
        'Gender': gender,
        'Age': age,
        'City': city,
        'Working Professional or Student': role,
        'Profession': profession,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'Degree': degree,
        'Have you ever had suicidal thoughts ?': suicidal,
        'Work/Study Hours': hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }])

    print("\nAnalyzing...")
    try:
        prediction = system.predict(input_data)[0]
        probability = system.predict_proba(input_data)[1].iloc[0]

        print("========================================")
        print("              RESULT                    ")
        print("========================================")
        if prediction == 1:
            print(f"Prediction: High Risk of Depression")
        else:
            print(f"Prediction: Low Risk of Depression")
        
        print(f"Probability Score: {probability:.2%}")
        print("========================================")
        print("Note: This is an AI-based assessment and not a medical diagnosis.")
        print("If you are feeling unwell, please consult a professional.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
