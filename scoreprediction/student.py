import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset (make sure student_scores.csv is in the same folder)
df = pd.read_csv("student_scores.csv")

# Features and target
X = df[['Hours']]
y = df['Scores']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
print("Model Accuracy (R^2):", model.score(X_test, y_test))

# Interactive input
user_hours = float(input("Enter study hours: "))
user_pred = model.predict([[user_hours]])
print(f"Predicted score for {user_hours} study hours: {user_pred[0]:.2f}/100")

# Plot regression line
plt.scatter(X, y, color='blue', label="Actual scores")
plt.plot(X, model.predict(X), color='red', label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Score (out of 100)")
plt.title("Study Hours vs Exam Score")
plt.legend()
plt.show()
