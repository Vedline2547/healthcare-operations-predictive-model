# Healthcare Operations: Predictive Model Development & Evaluation

## 1. Project Overview

This project develops and evaluates machine learning models to predict recorded healthcare encounter duration using historical healthcare operations data.

The objective is to explore whether encounter information available at the start of an encounter can help estimate its recorded duration. Such estimates could potentially support operational planning, resource allocation and workflow management.

**Important:** This is an exploratory machine learning project using a synthetic healthcare dataset. Predictions are not clinically validated and must not be used for diagnosis, treatment or direct clinical decision-making.

## 2. Problem Statement

Healthcare organizations need to understand and plan for the time associated with different types of encounters. This project investigates the following question:

*Can historical encounter characteristics and start-time information be used to predict the recorded duration of a healthcare encounter?*

The target variable is `DURATION_MINUTES`, calculated as the difference between encounter `STOP` and `START` timestamps.

## 3. Dataset

The project uses the following datasets:

- `encounters.csv`
- `patients.csv`
- `procedures.csv`
- `payers.csv`
- `organizations.csv`
- `data_dictionary.csv`

The encounters dataset contains 27,891 original records. Following investigation of suspiciously long records, 14 records were excluded from modelling, leaving 27,877 records.

The duration target is highly right-skewed, with most encounters having short recorded durations and a small number having much longer durations.

## 4. Data Preparation and Feature Engineering

The preparation workflow included:

- Parsing encounter timestamps.
- Calculating encounter duration in minutes.
- Investigating unusually long encounter records.
- Checking duplicate records and foreign-key consistency.
- Creating calendar-based features from the encounter start timestamp.
- Excluding identifiers and fields that could introduce leakage.

The selected model features were:

- `ENCOUNTERCLASS`
- `CODE`
- `PAYER`
- `START_YEAR`
- `START_MONTH`
- `START_DAY`
- `START_HOUR`
- `START_DAYOFWEEK`

The `STOP` timestamp was excluded because it is used to calculate the target. Cost and claim-related fields, reason codes and descriptions were also excluded because they may contain information unavailable at prediction time or recorded after the encounter begins.

## 5. Modelling Approach

The data was sorted chronologically and divided into:

- Training set: 22,301 records.
- Testing set: 5,576 records.

Four approaches were evaluated:

1. Median baseline.
2. Decision Tree Regressor.
3. Random Forest Regressor.
4. Gradient Boosting Regressor with a log-transformed target.

Categorical variables were one-hot encoded, while numerical features were passed through the preprocessing pipeline.

## 6. Model Evaluation

Models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 102.16 | 981.94 | -0.0109 |
| Decision Tree | 60.36 | 882.55 | 0.1834 |
| Random Forest | 60.93 | 881.42 | 0.1854 |
| Gradient Boosting | 64.84 | 891.79 | 0.1662 |

The Random Forest achieved the best RMSE and R², while the Decision Tree achieved the lowest MAE. Both substantially improved on the baseline.

The Random Forest was selected provisionally for the inference prototype.

## 7. Error Analysis and Limitations

Performance varied considerably across encounter classes.

The model performed well on common short-duration categories, including outpatient, urgent care and wellness encounters. However, it struggled with longer emergency and inpatient encounters.

In particular, the model underestimated many long inpatient durations. This is consistent with the highly skewed target distribution and the relatively small number of long-duration examples.

Other limitations include:

- The dataset contains synthetic healthcare records.
- Only one organization is represented.
- Rare, extended encounters are difficult to predict.
- The chronological test period contains a different distribution of some long encounters from the training period.
- The selected features may not capture all operational factors influencing encounter duration.
- Exclusion of suspicious records was based on investigation and heuristics, not confirmation that those records were erroneous.

Consequently, the model should be viewed as an exploratory operational prediction prototype rather than a validated production system.

## 8. Model Inference and Testing

The selected Random Forest pipeline was saved using Joblib.

An inference script is provided in `src/predict.py`. It accepts encounter class, encounter code, payer identifier and start timestamp, then returns a predicted duration in minutes.

Automated tests using Pytest check that:

- Predictions are numeric.
- Predictions are positive.
- Unseen encounter categories are handled.

All three tests passed.

## 9. Project Structure

```text
healthcare-operations-predictive-model/
│
├── data/
│   └── raw/
│
├── models/
│   └── random_forest_duration_model.joblib
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   └── 02_model_development.ipynb
│
├── src/
│   └── predict.py
│
├── tests/
│   └── test_predict.py
│
└── README.md
```

## 10. Running the Project

Install the required dependencies:

```bash
python -m pip install pandas numpy scikit-learn==1.7.2 joblib pytest
```

Run the inference script:

```bash
python src/predict.py
```

Run automated tests:

```bash
python -m pytest tests -v
```

The raw dataset files must be placed in `data/raw/` to reproduce the notebook workflow.

## 11. Conclusion

This project demonstrates an end-to-end machine learning workflow for a healthcare operations prediction problem, covering data inspection, preprocessing, feature engineering, model development, evaluation, error analysis, model persistence and inference testing.

Although the models improved substantially over the baseline, the results also demonstrate the challenges of predicting highly variable encounter durations, particularly for rare and extended encounters.