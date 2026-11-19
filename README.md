# 🩺 Ovarian Cancer Risk Prediction Using Logistic Regression

**A machine learning project that explores ovarian cancer risk classification using logistic regression implemented from scratch with Python and NumPy.**

---

## 📌 About the Project

Ovarian cancer is often difficult to detect in its early stages because its symptoms can be subtle or easily mistaken for other conditions. This project explores how machine learning can be used to analyze patient data and classify potential ovarian cancer cases.

The goal is to build a **multivariable logistic regression model from scratch**, using NumPy rather than relying on machine-learning libraries such as Scikit-learn for model training.

By implementing the algorithm manually, this project provides a practical understanding of how logistic regression works internally — from calculating predictions to computing gradients and updating model parameters through gradient descent.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical diagnostic tool and must not be used to make clinical decisions.

## 🎯 Project Objectives

- Understand and implement multivariable logistic regression from scratch.
- Explore and preprocess an ovarian cancer dataset.
- Apply feature standardization to improve model training.
- Implement the sigmoid activation function.
- Calculate binary cross-entropy (log loss).
- Train the model using gradient descent.
- Evaluate predictions using unseen test data.
- Understand how model coefficients influence predictions.

## 📂 Dataset

This project uses the **Ovarian Cancer Risk and Detection Dataset** from Kaggle.

**Dataset:** [Ovarian Cancer Risk and Detection Dataset](https://www.kaggle.com/datasets/gowtha69/ovarian-cancer-risk-and-detection-dataset)

The dataset contains patient-related information and a target variable:

- `Ovarian_Cancer_Status` — the classification target.
- `Partition` — identifies whether a record belongs to the training or testing partition.
- Other patient-related variables — used as model features after preprocessing.

The dataset file used in this project is:

`oc_risk50k_single_dataset.csv`

Patient identifiers are excluded from the predictive features because they identify records rather than provide meaningful clinical measurements.

## 🧠 How the Model Works

The model uses multivariable logistic regression to estimate the probability of a patient belonging to the positive class.

### 1. Linear Combination

The model first calculates a weighted sum of the input features:

$$
z = \beta_0 + X\beta
$$

Where:

- $X$ represents the input feature matrix.
- $\beta$ represents the feature coefficients.
- $\beta_0$ represents the intercept.Where:

- \(X\) represents the input feature matrix.
- \(\beta\) represents the feature coefficients.
- \(\beta_0\) represents the intercept.

### 2. Sigmoid function

The linear output is passed through the sigmoid function to produce a probability between 0 and 1:

$$
sigma(z) = \frac{1}{1+e^{-z}}
$$

### 3. Loss function

Binary cross-entropy measures the difference between the predicted probabilities and actual labels:

$$
L = -\frac{1}{m}\sum_{i=1}^{m}
[y_i\log(p_i)+(1-y_i)\log(1-p_i)]
$$

### 4. Gradient descent

The model calculates gradients and updates its parameters iteratively to minimize the loss.

The parameter updates are:

$$
\beta_0 := \beta_0-\alpha\frac{\partial L}{\partial\beta_0}
$$

$$
\beta := \beta-\alpha\frac{\partial L}{\partial\beta}
$$

Where \(\alpha\) is the learning rate.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Mathematical operations and model implementation |
| Pandas | Dataset loading and preprocessing |
| KaggleHub | Downloading the dataset |
| Jupyter Notebook | Experimentation and development |

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed.

Install the required libraries:

```bash
pip install numpy pandas kagglehub
```

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate into the project directory:

```bash
cd YOUR_REPOSITORY
```

### Run the project

Open the notebook in Jupyter:

```bash
jupyter notebook
```

Run the notebook cells in order to load the dataset, preprocess the data, train the model, and evaluate its predictions.

## 🔬 Model Training

The model is trained using gradient descent.

During training:

1. The model initializes its coefficients and intercept.
2. It calculates the linear combination of input features.
3. The sigmoid function converts the result into probabilities.
4. The loss function measures prediction error.
5. Gradients are calculated.
6. The coefficients and intercept are updated.
7. The process repeats for the specified number of epochs.

The training process uses standardized features to help ensure that variables with different numerical scales do not dominate the optimization process.

## 📊 Model Evaluation

The model should be evaluated using the dataset's held-out test partition.

Important evaluation metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall (Sensitivity):** Proportion of actual positive cases correctly identified.
- **Specificity:** Proportion of actual negative cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model distinguishes between the two classes.

For a medical classification problem, accuracy alone is not sufficient. False negatives and false positives should also be carefully examined.

Any reported performance should come from unseen test data, not just the training dataset.

## 💡 Key Learning Outcomes

Through this project, I explored:

- How logistic regression works mathematically.
- How to perform matrix operations with NumPy.
- How gradient descent optimizes model parameters.
- Why feature scaling matters in machine learning.
- How to separate training and testing data.
- Why model evaluation on unseen data is essential.
- How to build a machine-learning algorithm without relying on a prebuilt estimator.

## 🔮 Future Improvements

Possible improvements include:

- Add a confusion matrix visualization.
- Compare different learning rates and epoch counts.
- Experiment with different classification thresholds.
- Evaluate class imbalance.
- Add ROC and Precision-Recall curves.
- Compare the implementation against Scikit-learn's logistic regression.
- Experiment with regularization to reduce overfitting.
- Investigate feature importance and model interpretability.

## ⚠️ Important Note

This project is an educational exploration of machine learning applied to a healthcare-related dataset.

A high training accuracy does not necessarily mean that a model generalizes well. Clinical usefulness would require independent validation, assessment of data quality and bias, and rigorous evaluation by qualified medical professionals.

**This model should not be used to diagnose, screen, or guide treatment for ovarian cancer.**

## 👨‍💻 Author

Developed as a hands-on machine learning project to understand and implement logistic regression from first principles using Python and NumPy.

---

⭐ **If you find this project interesting, consider giving the repository a star!**
