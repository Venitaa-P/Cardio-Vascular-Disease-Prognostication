# Heart Disease Prediction System

## Model Performance Comparison
| Model               |   Accuracy |   ROC-AUC |   Precision |   Recall |       F1 |      MCC |   Training Time (s) |
|:--------------------|-----------:|----------:|------------:|---------:|---------:|---------:|--------------------:|
| SVM                 |   0.901639 |  0.94181  |    0.933333 |  0.875   | 0.903226 | 0.805167 |          0.00678444 |
| Logistic Regression |   0.885246 |  0.919181 |    0.878788 |  0.90625 | 0.892308 | 0.76998  |          0.00284672 |
| Random Forest       |   0.901639 |  0.942349 |    0.933333 |  0.875   | 0.903226 | 0.805167 |          0.0845506  |
| XGBoost             |   0.868852 |  0.912716 |    0.9      |  0.84375 | 0.870968 | 0.739505 |          0.0526755  |
| KNN                 |   0.918033 |  0.95528  |    0.935484 |  0.90625 | 0.920635 | 0.836384 |          0.00107217 |

## Key Visualizations
### ROC Curves Comparison
![ROC Curves](results/roc_curves.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)

## Detailed Metrics
### SVM
- **Accuracy**: 90.16%
- **ROC-AUC**: 0.942
- **Precision**: 0.93
- **Recall**: 0.88
- **F1 Score**: 0.90
- **MCC**: 0.81
- **Training Time**: 0.01s

#### Confusion Matrix
![SVM CM](results/confusion_matrix_svm.png)

#### Precision-Recall Curve
![SVM PR](results/pr_curve_svm.png)

### Logistic Regression
- **Accuracy**: 88.52%
- **ROC-AUC**: 0.919
- **Precision**: 0.88
- **Recall**: 0.91
- **F1 Score**: 0.89
- **MCC**: 0.77
- **Training Time**: 0.00s

#### Confusion Matrix
![Logistic Regression CM](results/confusion_matrix_logistic_regression.png)

#### Precision-Recall Curve
![Logistic Regression PR](results/pr_curve_logistic_regression.png)

### Random Forest
- **Accuracy**: 90.16%
- **ROC-AUC**: 0.942
- **Precision**: 0.93
- **Recall**: 0.88
- **F1 Score**: 0.90
- **MCC**: 0.81
- **Training Time**: 0.08s

#### Confusion Matrix
![Random Forest CM](results/confusion_matrix_random_forest.png)

#### Precision-Recall Curve
![Random Forest PR](results/pr_curve_random_forest.png)

### XGBoost
- **Accuracy**: 86.89%
- **ROC-AUC**: 0.913
- **Precision**: 0.90
- **Recall**: 0.84
- **F1 Score**: 0.87
- **MCC**: 0.74
- **Training Time**: 0.05s

#### Confusion Matrix
![XGBoost CM](results/confusion_matrix_xgboost.png)

#### Precision-Recall Curve
![XGBoost PR](results/pr_curve_xgboost.png)

### KNN
- **Accuracy**: 91.80%
- **ROC-AUC**: 0.955
- **Precision**: 0.94
- **Recall**: 0.91
- **F1 Score**: 0.92
- **MCC**: 0.84
- **Training Time**: 0.00s

#### Confusion Matrix
![KNN CM](results/confusion_matrix_knn.png)

#### Precision-Recall Curve
![KNN PR](results/pr_curve_knn.png)

