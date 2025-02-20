# Predicting Exercise Manner Using Machine Learning

## Project Overview
This project aims to predict the manner in which participants performed exercises using accelerometer data. A machine learning model is trained to classify movement patterns (`classe` variable). The project includes data preprocessing, model training, cross-validation, and predictions on test cases.

## Dataset
- **Training Data:** [pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
- **Test Data:** [pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

## Requirements
This project requires R and the following packages:

```r
install.packages(c("caret", "randomForest", "rpart", "rpart.plot", "e1071"))
```

## Running the Project
### Step 1: Load and Preprocess Data
Run the following script to load and clean the data:

```r
source("exercise_manner_ml.R")
```

### Step 2: Train Model and Evaluate Performance
The script trains a **Random Forest** model and evaluates it using cross-validation.

### Step 3: Generate Predictions
Predictions for test cases are printed at the end of the script.

## Output Files
- `exercise_manner_report.Rmd`: R Markdown report describing methodology and results.
- `exercise_manner_ml.R`: R script containing model training and evaluation.

## Author
Your Name
