# JobGuard: Fraudulent Online Job Advertisement Detection

## Overview

JobGuard is a machine learning project developed to identify potentially fraudulent online job advertisements using a Decision Tree classifier.

The model analyzes job advertisement information, including job titles, company profiles, descriptions, requirements, and other structured features to estimate the likelihood of an advertisement being fraudulent.

## Model Details

- **Model name:** JobGuard
- **Algorithm:** Decision Tree Classifier
- **Task:** Binary Classification
- **Framework:** Scikit-learn
- **Model format:** Joblib
- **Classification threshold:** 0.70
- **Dataset:** Employment Scam Aegean Dataset (EMSCAD)

## Performance

Final test results:

| Metric | Score |
|---|---:|
| Accuracy | 98.49% |
| Fraud Precision | 88% |
| Fraud Recall | 80% |
| Fraud F1-score | 84% |
| PR-AUC | 0.7527 |

## Intended Use

This model is intended for educational and research purposes. It can help identify suspicious job advertisements, but its predictions should not be treated as proof of fraud.

## Limitations

- The training dataset contains advertisements collected between 2012 and 2014.
- The dataset is imbalanced, with fewer fraudulent advertisements.
- The model may not recognize all modern recruitment scams.
- Predictions depend on the information provided by users.

## Files

- `final_decision_tree_model.joblib`: Trained preprocessing and classification pipeline with the selected threshold.

## Developer

Developed as an MSc Computer Science machine learning project at Addis Ababa University.
