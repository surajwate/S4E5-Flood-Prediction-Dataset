# S4E5 Flood Prediction Dataset - Kaggle Playground Series

This repository contains the code and resources for the **Kaggle Playground Series S4E5: Flood Prediction** dataset competition. The goal of this project is to predict the probability of flooding in various regions using machine learning models, with a focus on regression techniques.

## Overview

In this project, we work with a synthetically generated dataset where each feature represents a potential cause or factor related to floods. The models are evaluated based on their **R² score**. We explored multiple models, including **CatBoost**, **Linear Regression**, and **Random Forest**, and built a structured pipeline for efficient model training and evaluation.

For more details about the challenge and the approach, please refer to the following resources:
- [Blog Post: Regression with a Flood Prediction Dataset](https://surajwate.com/blog/regression-with-a-flood-prediction-dataset/)
- [Kaggle Notebook: Flood Prediction](https://www.kaggle.com/code/surajwate/s4e5-flood-prediction)
- [Kaggle Competition: Playground Series - S4E5](https://www.kaggle.com/competitions/playground-series-s4e5)

## Project Structure

The project is structured as follows:

```
├── input
│   ├── train.csv
│   ├── test.csv
│   ├── train_folds.csv
│   └── sample_submission.csv
├── logs
│   └── logs.txt
├── notebooks
│   └── eda.ipynb
├── output
├── src
│   ├── config.py
│   ├── create_fold.py
│   ├── final_model.py
│   ├── model_dispatcher.py
│   ├── train.py
│   └── main.py
├── README.md
└── requirements.txt
```
All folders and files shown in the structure is not present in this repository because I added them to .gitignore file, to keep the repository lean. 

### Key Files

- **`src/create_fold.py`**: This script creates stratified K-fold cross-validation splits and outputs a `train_folds.csv` file. This needs to be run before training the models.
- **`src/main.py`**: The entry point to run different models. Executes the training process with a single command.
- **`src/train.py`**: Contains the core training logic, including data preprocessing, model fitting, and evaluation.
- **`src/model_dispatcher.py`**: A dictionary of all models used in the project.
- **`src/config.py`**: Configuration file containing paths and logging settings.
- **`notebooks/eda.ipynb`**: Jupyter notebook containing the exploratory data analysis (EDA).
- **`logs/logs.txt`**: Log file that stores information about the training process, including R² scores and training time for each model.
- **`input/train.csv`**: The training dataset used to build the model.
- **`input/test.csv`**: The test dataset for making final predictions.

## Installation

To run this project on your local machine, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/surajwate/S4E5-Flood-Prediction-Dataset.git
cd S4E5-Flood-Prediction-Dataset
pip install -r requirements.txt
```

### Downloading the Data

Before running the code, you will need to download the dataset from Kaggle:

1. Go to the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s4e5).
2. Download the `train.csv`, `test.csv`, and `sample_submission.csv` files and place them in the `input/` directory.

## Usage

### Step 1: Create Folds for Cross-Validation
Before training the models, you need to run `create_fold.py` to generate the K-fold splits. This script outputs the `train_folds.csv` file that is required for model training.

Run the following command:
```bash
python src/create_fold.py
```

This will generate a new `train_folds.csv` file in the `input/` directory, which will be used by the `train.py` script.

### Step 2: Train Models

Once the folds are created, you can train any of the models specified in the `model_dispatcher.py` by running the following command:

```bash
python src/main.py --model xgboost
```

This command will start the training process for the specified model (`xgboost` in this case) using the training dataset, and it will generate predictions for the test dataset.

To train with other models, replace `xgboost` with any of the following available models:
- `linear_regression`
- `random_forest`
- `catboost`
- `lightgbm`
- `svr`
- `kneighbors`
- `ridge`
- `lasso`

## Evaluation

All models are evaluated based on their **R² score**. The results for each fold, including training time, are logged in the `logs/logs.txt` file for each model.

### Results:

The best-performing model in our experiments was **CatBoostRegressor**, which achieved an average R² score of **0.846** across all folds.

For more details, check out the [blog post](https://surajwate.com/blog/regression-with-a-flood-prediction-dataset/).

## Related Links

- **Kaggle Competition**: [Playground Series - S4E5](https://www.kaggle.com/competitions/playground-series-s4e5)
- **Kaggle Notebook**: [S4E5 Flood Prediction](https://www.kaggle.com/code/surajwate/s4e5-flood-prediction)
- **Blog Post**: [Regression with a Flood Prediction Dataset](https://surajwate.com/blog/regression-with-a-flood-prediction-dataset/)
