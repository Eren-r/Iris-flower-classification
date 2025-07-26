# 🌸 Iris Flower Classification

This project classifies Iris flowers into 3 species — *Setosa, Versicolor, Virginica* — using machine learning algorithms like Logistic Regression and KNN.

## 📊 Dataset
- Built-in Iris dataset from `sklearn.datasets`
- Features: Sepal length, Sepal width, Petal length, Petal width

## 🧠 Models Used
- Logistic Regression
- K-Nearest Neighbors (K=3)

## 🔍 Evaluation Metrics
- Accuracy Score
- Classification Report
- Confusion Matrix (with heatmap)

## 📈 Results
- Both models perform >95% accuracy
- Visual plots and confusion matrices included

## 🔧 Requirements
- Python 3.x
- scikit-learn
- pandas
- seaborn
- matplotlib

### 🔍 Why Logistic Regression and K-Nearest Neighbors?

We chose two popular classification algorithms :

#### ✅ Logistic Regression
- **Simple and interpretable** model.
- Works well when classes are linearly separable.
- Provides probability estimates, useful for understanding model confidence.
- Fast training even on large datasets.

#### ✅ K-Nearest Neighbors (KNN)
- **Instance-based learning**: Makes predictions based on closest training samples.
- Effective when decision boundaries are non-linear.
- No training phase — lazy learner, making it intuitive and easy to implement.
- Highly accurate on well-structured datasets like Iris.

These models help us explore both **linear** (Logistic Regression) and **non-linear** (KNN) decision boundaries — giving a well-rounded understanding of classification techniques.
