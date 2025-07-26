from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load CSV file
df = pd.read_csv('Iris.csv')  


X = df.iloc[:, :-1] 
y = df.iloc[:, -1]  


le = LabelEncoder()
y = le.fit_transform(y)
target_names = le.classes_
df['species_encoded'] = y

# 2) Data visualization
sns.pairplot(df.iloc[:, :-1].assign(species=df.iloc[:, -1]), hue="species")
plt.suptitle("Iris Flower Features Pairplot", y=1.02)
plt.show()

# 3) Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4) Train Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
log_preds = log_reg.predict(X_test)

# 5) Train K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)

# 6) Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("KNN Classifier", y_test, knn_preds)