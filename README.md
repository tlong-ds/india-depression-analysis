# Depression Analysis Pipeline

This project implements a comprehensive machine learning pipeline to analyze and predict depression levels based on survey data. It is designed to handle the distinct characteristics of "Student" and "Working Professional" groups by applying separate data processing and modeling strategies.

## Project Information

**Subject**: Data Preparation and Visualization  
**Instructor**: Dr. Nguyen Tuan Long

**Team Members**:

| No. | Name | Student ID | Class |
|-----|------|------------|-------|
| 1 | Nguyễn Vân Anh | 11230603 | DSEB 65B |
| 2 | Nguyễn Mạnh Cường | 112305 | DSEB 65B |
| 3 | Trần Thu Hiền | 11230534 | DSEB 65B |
| 4 | Lý Thành Long | 11230561 | DSEB 65B |
| 5 | Nguyễn Thanh Mơ | 11230571 | DSEB 65B |

## Data Dictionary

The dataset includes information on demographics, work/study, lifestyle, and medical history.

| Column Name | Description | Data Type |
|---|---|---|
| Name | Name of the survey participant | Object |
| Gender | Gender | Object |
| City | Current City | Object |
| Working Professional or Student | Current Occupation (Working Professional or Student) | Object |
| Profession | Specific Profession | Object |
| Academic Pressure | Academic Pressure (scale 1-5) | Float |
| Work Pressure | Work Pressure (scale 1-5) | Float |
| CGPA | Cumulative Grade Point Average | Float |
| Study Satisfaction | Satisfaction with Study (scale 1-5) | Float |
| Job Satisfaction | Satisfaction with Job (scale 1-5) | Float |
| Sleep Duration | Average Sleep Duration | Object |
| Dietary Habits | Dietary Habits | Object |
| Degree | Degree | Object |
| Have you ever had suicidal thoughts ? | History of suicidal thoughts | Object |
| Work/Study Hours | Daily Work/Study Hours | Float |
| Financial Stress | Financial Stress (scale 1-5) | Float |
| Family History of Mental Illness | Family History of Mental Illness | Object |
| Depression | Depression Status (0: No, 1: Yes) | Int |

## Project Overview

The pipeline reproduces and operationalizes the logic found in the exploratory data analysis notebooks. It includes:
- **Data Cleaning**: Handling mixed data types and "garbage" values (e.g., "Error", "Null").
- **Feature Engineering**: Imputation, scaling, and encoding tailored to specific feature sets.
- **Modeling**: A `RandomForestClassifier` trained separately for students and working professionals.
- **Orchestration**: A unified system that automatically routes data to the correct sub-pipeline.

## Project Structure

```
.
├── dataset/
│   └── clean_depression_dataset.csv  # Input data
├── model/
│   └── full_depression_model.joblib  # Saved model artifact
├── notebooks/                        # Exploratory analysis and prototyping
├── src/
│   ├── config.py                     # Feature definitions and constants
│   ├── pipeline.py                   # Main pipeline logic and system class
│   └── transformers.py               # Custom scikit-learn transformers
├── main.py                           # Entry point script
├── requirements.txt                  # Project dependencies
└── README.md                         # This file
```

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the pipeline (train the model if it doesn't exist, and generate predictions):

```bash
python main.py
```

### What `main.py` does:
1. **Loads Data**: Reads the dataset from `dataset/clean_depression_dataset.csv`.
2. **Model Management**:
   - Checks if a trained model exists at `model/full_depression_model.joblib`.
   - If found, it loads the model.
   - If not found, it initializes the `DepressionAnalysisSystem`, fits it on the data, and saves the trained model.
3. **Prediction**: Generates class predictions and probabilities for the input data.

## Pipeline Details

The core logic is encapsulated in the `DepressionAnalysisSystem` class in `src/pipeline.py`.

### 1. Data Split
The system automatically splits the input data into two subsets based on the `Working Professional or Student` column.

### 2. Student Pipeline
- **Features**: Uses common features plus student-specific ones like `Academic Pressure`, `CGPA`, `Study Satisfaction`.
- **Processing**:
    - **Garbage Cleaning**: Coerces specific columns to numeric, handling errors.
    - **Imputation**: Median for numerics, Most Frequent for categoricals.
    - **Scaling/Encoding**: StandardScaler for numerics, OneHotEncoder for nominals, OrdinalEncoder for `Sleep Duration` and `Dietary Habits`.
- **Model**: RandomForestClassifier.

### 3. Worker Pipeline
- **Features**: Uses common features plus worker-specific ones like `Work Pressure`, `Job Satisfaction`, `Profession`.
- **Processing**: Similar strategies to the student pipeline but applied to the worker-specific feature set.
- **Model**: RandomForestClassifier.

### 4. Configuration
All feature lists and ordinal mappings are defined in `src/config.py` to ensure consistency and easy updates.
