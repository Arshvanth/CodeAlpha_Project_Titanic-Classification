import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'socio_economic_status': [1, 2, 1, 3, 2],  # 1 = Low, 2 = Medium, 3 = High
    'age': [22, 45, 33, 60, 25],
    'gender': [1, 0, 1, 0, 1],  # 1 = Male, 0 = Female
    'health_status': [1, 2, 1, 3, 1],  # 1 = Good, 2 = Fair, 3 = Poor
    'preparedness': [1, 2, 2, 1, 1],  # 1 = Low, 2 = High
    'survived': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Prepare the data
X = df[['socio_economic_status', 'age', 'gender', 'health_status', 'preparedness']]
y = df['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Prediction function
def predict_survival(socio_economic_status, age, gender, health_status, preparedness):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'socio_economic_status': [socio_economic_status],
        'age': [age],
        'gender': [gender],
        'health_status': [health_status],
        'preparedness': [preparedness]
    })

    # Make prediction
    prediction = model.predict_proba(input_data)[0][1]  # Probability of survival

    return prediction

# Example usage
prediction = predict_survival(2, 30, 1, 1, 2)
print(f"Probability of Survival: {prediction:.2f}")
