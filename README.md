# Credit Card Fraud Detection Project

## Overview
This project focuses on detecting fraudulent credit card transactions using three machine learning models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**

Each model is trained and evaluated for its performance, with a comparison of their accuracies and classification reports.

---

## Features
1. **Data Preparation**:
   - Dataset: `creditcard.csv`
   - Features: 28 anonymized variables (`V1` to `V28`), along with `Time` and `Amount`.
   - Target: `Class` (0 = Non-Fraud, 1 = Fraud).

2. **Exploratory Data Analysis**:
   - Missing values check.
   - Visualizations:
     - Fraud vs. Non-Fraud transactions.
     - Correlation heatmap.

3. **Machine Learning Models**:
   - Logistic Regression
   - Random Forest Classifier
   - Decision Tree Classifier

4. **Evaluation Metrics**:
   - Confusion Matrix
   - Classification Report
   - Accuracy Score

---

## Requirements

### Libraries Used
The project uses the following Python libraries:
- `pandas` (data manipulation)
- `scikit-learn` (machine learning tools)
- `seaborn` (data visualization)
- `matplotlib` (plotting)

### Installation
Install the required libraries using:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

---

## How to Run
1. **Dataset**: Place the `creditcard.csv` file in the project directory.  
   (Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).  

2. **Run the Code**: Execute the script:
   ```bash
   python main.py
   ```

3. **Outputs**:
   - Visualizations (class distribution and correlation heatmap).
   - Confusion matrix and classification reports for all three models.
   - Accuracy comparison for Logistic Regression, Random Forest, and Decision Tree models.

---

## Results
- **Logistic Regression**:
  - Suitable for linear relationships and simple datasets.
- **Random Forest Classifier**:
  - High accuracy and robust performance.
- **Decision Tree Classifier**:
  - Easy to interpret but may overfit.

### Sample Output:
| Model                 | Accuracy  | Comments                |
|-----------------------|-----------|-------------------------|
| Logistic Regression   | ~98.5%   | Good baseline model     |
| Random Forest         | ~99.6%   | Best performance overall|
| Decision Tree         | ~99.0%   | Overfits slightly       |

---

## Visualizations

### Fraud vs Non-Fraud Transactions
This plot shows the imbalance in the dataset, with far fewer fraud cases compared to non-fraud cases.

### Correlation Heatmap
Displays the relationships between features to identify patterns.

---

## Future Enhancements
- Address data imbalance using techniques like **SMOTE**.
- Implement advanced models like **XGBoost**.
- Incorporate precision, recall, and F1-score for more detailed performance evaluation.

## Contributing
Feel free to fork this repository, make changes, and submit a pull request. Feedback is always welcome!

## Acknowledgements
- Dataset provided by ULB’s Machine Learning Group.
- Thanks to the Python community for tools like **Scikit-learn**, **Seaborn**, and **Matplotlib**.
