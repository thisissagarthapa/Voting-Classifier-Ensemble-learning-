# Ensemble Learning with Voting Classifier

This project demonstrates the use of ensemble learning through the implementation of a **Voting Classifier** using three different machine learning algorithms: Decision Tree, Support Vector Machine (SVM), and Gaussian Naive Bayes.

## Project Overview

Ensemble learning is a technique where multiple models (often referred to as weak learners) are combined to improve the overall performance. In this project, we implement **Voting Classifier**, which combines the predictions of multiple models to make more accurate predictions.

### Models Used:
1. **Decision Tree Classifier**: A tree-like model of decisions and their possible consequences.
2. **Support Vector Classifier (SVC)**: A classifier that constructs hyperplanes to separate data points into different classes.
3. **Gaussian Naive Bayes (GNB)**: A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions.

## Dataset

The dataset used is generated using `make_moons` from `sklearn.datasets`. It contains two features (`x1`, `x2`) and a target label (`y`). This dataset is ideal for binary classification tasks.

- **Number of Samples**: 200
- **Noise Level**: 0.1

## Libraries Used

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For plotting.
- **Seaborn**: For enhanced visualizations.
- **Scikit-Learn**: For building the machine learning models.

## Project Workflow

1. **Data Generation**:
    - The dataset is generated using `make_moons(n_samples=200, noise=0.1)`.
    
2. **Data Visualization**:
    - The dataset is visualized using `Seaborn` to get an idea of the distribution of the two classes (`y`).

3. **Train-Test Split**:
    - The dataset is split into training and testing sets using `train_test_split` from `sklearn`.

4. **Model Training**:
    - **Decision Tree**, **SVM**, and **Naive Bayes** models are trained individually.
    - Each model's training and testing scores are evaluated.

5. **Voting Classifier**:
    - A **Voting Classifier** is implemented using the three models, combining their predictions to get a final output.
    - The classifier is tested on both the training and test datasets to evaluate its performance.

6. **Comparison of Individual Model Predictions**:
    - The predictions of individual models (Decision Tree, SVM, Naive Bayes) on the test dataset are compared.

## Results

| Model                | Training Score | Testing Score |
|----------------------|----------------|---------------|
| Decision Tree         | 100%           | 97.5%         |
| Support Vector Classifier | 97.5%         | 98.125%       |
| Gaussian Naive Bayes  | 85%            | 88.125%       |
| Voting Classifier     | 97.5%          | 97.5%         |

The **Voting Classifier** provides a balanced and robust performance by combining the predictions of all three models.

## Code

To run the code:

```python
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# Data Generation
x, y = make_moons(n_samples=200, noise=0.1)
df = {"x1": x[:, 0], "x2": x[:, 1], "y": y}
datasets = pd.DataFrame(df)

# Data Visualization
sns.scatterplot(x="x1", y="x2", hue="y", data=datasets)
plt.show()

# Splitting the dataset
x1 = datasets.iloc[:, :-1]
y1 = datasets["y"]
x_train, x_test, y_train, y_test = train_test_split(x1, y1, train_size=0.2, random_state=42)

# Model Initialization
dt = DecisionTreeClassifier()
sv = SVC()
gnb = GaussianNB()

# Training individual models
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)
gnb.fit(x_train, y_train)

# Voting Classifier
lst = [("dt", DecisionTreeClassifier()), ("svm", SVC()), ("gnb", GaussianNB())]
vc = VotingClassifier(estimators=lst)
vc.fit(x_train, y_train)

# Model evaluation
print("Voting Classifier Training Accuracy:", vc.score(x_train, y_train)*100)
print("Voting Classifier Testing Accuracy:", vc.score(x_test, y_test)*100)
